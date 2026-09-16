#!/usr/bin/env python3
"""Time-safe UCL goal model built on champions_league_results.py.

Dependencies: numpy, pandas, scipy, scikit-learn. See README.md for the
26-factor mapping, input contracts, assumptions, and command examples.
No network calls, guessed xG, fabricated injury data, or fixed team bonuses.
"""
from __future__ import annotations

import argparse
import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import PoissonRegressor

# These are INPUT fields, not numerical coefficients. Effects are fitted.
CONTEXT_FIELDS = (
    "squad_depth_score", "quality_dropoff", "key_absences_impact",
    "available_squad_pct", "continuity_minutes_share", "manager_cl_matches",
    "manager_cl_residual", "travel_km", "timezone_shift_hours", "eastbound",
    "distance_14d", "high_intensity_km_14d", "sprints_14d", "ppda",
    "line_height", "compactness", "progressive_passes", "progressive_carries",
    "cross_share", "set_piece_xg_for", "set_piece_xg_against",
    "xg_rate_leading", "xg_rate_trailing", "xg_rate_level",
    "xga_rate_leading", "xga_rate_trailing", "xga_rate_level",
    "domestic_position_percentile", "domestic_title_pressure",
    "uefa_club_points", "uefa_association_points", "wage_bill_eur_m",
    "revenue_eur_m", "crowd_capacity_fraction", "referee_cards_per_match",
    "referee_penalties_per_match", "cl_rank", "cl_points_to_eighth",
    "cl_points_to_twentyfourth", "cl_matches_remaining", "is_dead_rubber",
    "expected_rotation_share", "future_opponent_strength", "bracket_strength",
    "away_goals_legacy_exposure", "matchday",
)
STATS = ("shots", "shots_on_target", "corners", "possession_pct", "pass_accuracy",
         "fouls", "offsides", "saves", "yellow_cards", "red_cards", "xg")
ALIASES = {"atletico de madrid": "atletico madrid", "bayern munchen": "bayern munich",
           "internazionale": "inter milan", "inter": "inter milan", "as roma": "roma",
           "vfb stuttgart": "stuttgart", "psv": "psv eindhoven", "fc porto": "porto",
           "slavia praha": "slavia prague", "lask linz": "lask", "sabah fk": "sabah",
           "feyenoord rotterdam": "feyenoord", "viking fk": "viking"}
CODE_NAMES = dict(zip(
    "ARS AVL LIV MCI MUN ATM BAR BET RMA VIL BAY BVB LEI STU COM INT NAP ROM PSG LEN LIL POR SCP FEY PSV FEN GAL BRU BOD VIK SHK SLA SLO LAS AEK SAB".lower().split(),
    ["arsenal", "aston villa", "liverpool", "manchester city", "manchester united",
     "atletico madrid", "barcelona", "real betis", "real madrid", "villarreal",
     "bayern munich", "borussia dortmund", "rb leipzig", "stuttgart", "como",
     "inter milan", "napoli", "roma", "paris saint germain", "lens", "lille",
     "porto", "sporting cp", "feyenoord", "psv eindhoven", "fenerbahce",
     "galatasaray", "club brugge", "bodo glimt", "viking", "shakhtar donetsk",
     "slavia prague", "slovan bratislava", "lask", "aek athens", "sabah"]))


def canonical(value):
    value = str(value).replace("ø", "o").replace("Ø", "O").replace("ı", "i")
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    text = " ".join("".join(c if c.isalnum() else " " for c in text).split())
    return CODE_NAMES.get(text, ALIASES.get(text, text))


def utc(value):
    return pd.to_datetime(value, utc=True)


def finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def load_history(path: str | Path) -> pd.DataFrame:
    """Read our original JSON or the collector's output directory; deduplicate games.

    Returns ONE row per match. Results must represent regulation time (90 minutes).
    For older knockout data the caller must replace ET/penalty-inclusive scores.
    """
    path = Path(path)
    if path.is_dir():
        frames = [pd.read_csv(path / name) for name in
                  ("champions_league_results.csv", "domestic_league_results.csv")]
        records = pd.concat(frames, ignore_index=True).to_dict("records")
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
        rename = {"eventId": "event_id", "name": "team", "gf": "goals_for",
                  "ga": "goals_against", "shotsOnTarget": "shots_on_target",
                  "possession": "possession_pct", "passAccuracy": "pass_accuracy",
                  "yellowCards": "yellow_cards", "redCards": "red_cards"}
        records = []
        for collection, cl in (("championsLeague", True), ("domesticMatches", False)):
            for raw in data[collection]:
                row = {rename.get(k, k): v for k, v in raw.items()}
                row.update(competition="Champions League" if cl else raw["league"],
                           stage="league_phase" if cl else "domestic")
                records.append(row)
    else:
        # Canonical match-level CSV is also accepted; useful for historical expansion.
        result = pd.read_csv(path)
        result["date"] = utc(result["date"])
        for key in ("home", "away"):
            result[key] = result[key].map(canonical)
        return validate_history(result)
    games = {}
    for row in records:
        competition = row["competition"]
        key = f"{competition}:{row['event_id']}"
        team, opponent = canonical(row["team"]), canonical(row["opponent"])
        if row["venue"] not in ("Home", "Away"):
            raise ValueError("Venue must be Home or Away; supply neutral=1 separately")
        home_side = row["venue"] == "Home"
        home, away = (team, opponent) if home_side else (opponent, team)
        gf, ga = int(row["goals_for"]), int(row["goals_against"])
        hg, ag = (gf, ga) if home_side else (ga, gf)
        base = dict(match_id=key, date=utc(row["date"]), home=home, away=away,
                    home_goals=hg, away_goals=ag, competition=competition,
                    stage=row.get("stage") or "unknown", neutral=0, leg="single")
        if key in games:
            for field in ("home", "away", "home_goals", "away_goals", "date"):
                if games[key][field] != base[field]:
                    raise ValueError(f"Conflicting duplicated match {key}: {field}")
        else:
            games[key] = base
        prefix = "home_" if home_side else "away_"
        populated = any(finite(row.get(s)) > 0 for s in
                        ("shots", "possession_pct", "corners", "fouls"))
        for stat in STATS:
            val = finite(row.get(stat))
            if stat != "xg" and not populated:
                val = np.nan  # unpublished all-zero template
            if stat == "pass_accuracy" and val == 0:
                val = np.nan
            games[key][prefix + stat] = val
    return validate_history(pd.DataFrame(games.values()))


def validate_history(frame):
    required = {"match_id", "date", "home", "away", "home_goals", "away_goals", "competition"}
    if not required.issubset(frame):
        raise ValueError(f"Missing match columns: {sorted(required - set(frame))}")
    frame = frame.copy()
    frame["date"] = utc(frame["date"])
    if frame[list(required)].isna().any().any() or frame.match_id.duplicated().any():
        raise ValueError("Missing required values or duplicate match IDs")
    for field in ("home_goals", "away_goals"):
        values = pd.to_numeric(frame[field], errors="raise")
        if ((values < 0) | (values % 1 != 0)).any():
            raise ValueError("Goals must be non-negative integers")
        frame[field] = values.astype(int)
    if (frame.home == frame.away).any():
        raise ValueError("A team cannot play itself")
    for col, default in (("stage", "unknown"), ("neutral", 0), ("leg", "single")):
        if col not in frame:
            frame[col] = default
        frame[col] = frame[col].fillna(default)
    frame["stage"] = frame.stage.replace({"LP": "league_phase", "PO": "playoff",
        "R16": "round_of_16", "QF": "quarter_final", "SF": "semi_final", "F": "final"})
    frame["neutral"] = pd.to_numeric(frame.neutral, errors="raise")
    if not frame.neutral.isin([0, 1]).all():
        raise ValueError("neutral must be 0 or 1")
    return frame.sort_values(["date", "match_id"]).reset_index(drop=True)


def validate_fixtures(frame):
    required = {"match_id", "date", "home", "away", "competition"}
    if not required.issubset(frame) or frame.empty:
        raise ValueError("Provide at least one fixture with match_id,date,home,away,competition")
    frame = frame.copy()
    frame["date"] = utc(frame.date)
    if frame[list(required)].isna().any().any() or frame.match_id.duplicated().any():
        raise ValueError("Missing fixture fields or duplicate IDs")
    for field in ("home", "away"):
        frame[field] = frame[field].map(canonical)
    if (frame.home == frame.away).any():
        raise ValueError("A fixture needs two different teams")
    for field, default in (("stage", "unknown"), ("neutral", 0), ("leg", "single")):
        if field not in frame:
            frame[field] = default
        frame[field] = frame[field].fillna(default)
    frame["neutral"] = pd.to_numeric(frame.neutral, errors="raise")
    if not frame.neutral.isin([0, 1]).all():
        raise ValueError("neutral must be 0 or 1")
    if not frame.leg.isin(["single", "first", "second", "first_leg", "second_leg"]).all():
        raise ValueError("Unknown leg label")
    return frame


def from_phase1(matches) -> pd.DataFrame:
    """Bridge to the uploaded core.Match objects; only explicitly provided results.

    Original xG/physical fields default to zero: do NOT import these as observations.
    Supply verified historical metrics in canonical CSV or dated context instead.
    Pass only completed 90-minute matches, never the synthetic demo's fixtures.
    """
    rows = []
    for m in matches:
        value = lambda x: getattr(x, "value", x)
        rows.append(dict(match_id=str(m.match_id), date=utc(m.date),
            home=canonical(m.home_team), away=canonical(m.away_team),
            home_goals=m.home_goals, away_goals=m.away_goals,
            competition="Champions League" if value(m.competition) == "UCL" else str(value(m.competition)),
            is_domestic_league=int(value(m.competition) == "DOM"),
            stage=str(value(m.stage)) if m.stage else "unknown", leg=str(value(m.leg))))
    return validate_history(pd.DataFrame(rows))


class Context:
    """Match-specific pre-kickoff snapshots, keyed by match_id AND team.

    Each row is a full snapshot. Rows require available_at and source_url.
    Using today's injury/coefficient table for older fixtures is not permitted.
    """
    def __init__(self, path=None):
        self.rows = {}
        if path is not None:
            frame = pd.read_csv(path)
            required = {"match_id", "team", "available_at", "source_url"}
            if not required.issubset(frame) or frame[list(required)].isna().any().any():
                raise ValueError("Context needs match_id, team, available_at, source_url")
            frame["available_at"] = utc(frame.available_at)
            frame["team"] = frame.team.map(canonical)
            if frame.duplicated(["match_id", "team", "available_at"]).any():
                raise ValueError("Duplicate context snapshots")
            for key, group in frame.groupby(["match_id", "team"]):
                self.rows[key] = group.sort_values("available_at")

    def get(self, fixture, team):
        group = self.rows.get((fixture["match_id"], team))
        if group is None:
            return {}
        eligible = group[group.available_at < utc(fixture["date"])]
        if eligible.empty:
            return {}
        return eligible.iloc[-1].to_dict()


@dataclass(frozen=True)
class Config:
    # Hyperparameters, NOT empirically established feature effects.
    alpha: float = 1.0
    half_life_days: float = 120.0
    form_matches: int = 10
    shrinkage_matches: float = 5.0
    rho_prior_precision: float = 25.0
    min_train_matches: int = 30

    def __post_init__(self):
        if min(self.alpha, self.half_life_days, self.form_matches,
               self.shrinkage_matches, self.min_train_matches) <= 0:
            raise ValueError("Regularization, windows, shrinkage and minimum sample must be positive")
        if self.rho_prior_precision < 0:
            raise ValueError("rho_prior_precision must be non-negative")


class Features:
    def __init__(self, history, context=None, config=None):
        self.config = config or Config()
        self.context = context or Context()
        rows = []
        for m in history.to_dict("records"):
            for side, other in (("home", "away"), ("away", "home")):
                gf, ga = m[side + "_goals"], m[other + "_goals"]
                row = dict(team=m[side], opponent=m[other], date=m["date"],
                           gf=gf, ga=ga, points=3 if gf > ga else 1 if gf == ga else 0,
                           cl=m["competition"] == "Champions League", stage=m["stage"],
                           domestic=bool(m.get("is_domestic_league", m["competition"] != "Champions League")),
                           home=int(side == "home" and not m["neutral"]))
                row.update({s: finite(m.get(side + "_" + s)) for s in STATS})
                rows.append(row)
        self.history = pd.DataFrame(rows)
        self.by_team = {k: v.sort_values("date") for k, v in self.history.groupby("team")}

    def team(self, fixture, side):
        team = fixture[side]
        opponent = fixture["away" if side == "home" else "home"]
        cutoff = utc(fixture["date"]).normalize()  # conservative with date-only collector CSVs
        prior = self.by_team.get(team, self.history.iloc[:0])
        prior = prior[prior.date < cutoff]
        recent = prior.tail(self.config.form_matches)
        age = (cutoff - recent.date).dt.total_seconds() / 86400
        weights = np.exp2(-age.to_numpy() / self.config.half_life_days)
        num = {"observed_matches": float(len(prior)), "home_advantage":
               0.0 if fixture.get("neutral", 0) else (0.5 if side == "home" else -0.5)}
        for metric in ("gf", "ga", "points") + STATS:
            values = recent[metric].to_numpy(dtype=float)
            valid = np.isfinite(values)
            num["form_" + metric] = np.average(values[valid], weights=weights[valid]) if valid.any() else np.nan
        all_age = (cutoff - prior.date).dt.total_seconds() / 86400
        num["observed_rest_days"] = float(all_age.min()) if len(prior) else np.nan
        for days in (7, 14, 21, 30):
            num[f"observed_matches_{days}d"] = float((all_age <= days).sum())
        num["schedule_load"] = float(np.exp2(-all_age[all_age <= 30] / 5.0).sum())
        num["unbeaten_run"] = 0.0
        for p in reversed(prior.points.tolist()):
            if p == 0:
                break
            num["unbeaten_run"] += 1
        for label, subset in (("cl", prior[prior.cl]), ("domestic", prior[prior.domestic]),
                              ("stage", prior[(prior.stage == fixture.get("stage")) & prior.cl]),
                              ("h2h", prior[prior.opponent == opponent])):
            n = len(subset)
            # Shrink point differential toward zero; neutral prior is a modeling choice.
            num[label + "_form"] = float((subset.points - 1).sum() / (n + self.config.shrinkage_matches))
            num[label + "_sample"] = float(n)
        values = self.context.get(fixture, team)
        num.update({k: finite(values.get(k)) for k in CONTEXT_FIELDS})
        num["short_rest_depth"] = (max(0.0, 4 - num["observed_rest_days"])
                                   * (1 - num["squad_depth_score"]))
        num["press_vs_progression"] = np.nan  # paired below
        # Aggregate totals are explicitly from THIS fixture's home/away perspective.
        if fixture.get("leg") in ("second", "second_leg"):
            if "aggregate_home" not in fixture or "aggregate_away" not in fixture:
                raise ValueError("Second legs require aggregate_home and aggregate_away")
            lead = finite(fixture["aggregate_home"]) - finite(fixture["aggregate_away"])
            if not math.isfinite(lead):
                raise ValueError("Missing second-leg aggregate")
            num["aggregate_lead"] = lead if side == "home" else -lead
        else:
            num["aggregate_lead"] = 0.0
        is_cl = fixture["competition"] == "Champions League"
        cat = {"attack": team, "defence": opponent, "competition": fixture["competition"],
               "stage": fixture.get("stage", "unknown"), "leg": fixture.get("leg", "single"),
               "european_attack": team if is_cl else "domestic",
               "european_home": team if is_cl and side == "home" and not fixture.get("neutral", 0) else "none"}
        return num, cat

    def pair(self, fixture):
        h, hc = self.team(fixture, "home")
        a, ac = self.team(fixture, "away")
        pairs = []
        for own, opp, cat in ((h, a, hc), (a, h, ac)):
            values = dict(own)
            values.update({"opponent_" + k: v for k, v in opp.items() if k != "home_advantage"})
            values["press_vs_progression"] = own["ppda"] * opp["progressive_passes"]
            pairs.append((values, cat))
        return pairs

    def build(self, fixtures):
        pairs = [pair for fixture in fixtures.to_dict("records") for pair in self.pair(fixture)]
        return pd.DataFrame([p[0] for p in pairs]), [p[1] for p in pairs]


class Encoder:
    """Train-only median imputation + scaling; retain missingness indicators.

    Features that are never observed or never vary are not activated at prediction.
    This prevents new injury/coefficient inputs acquiring untrained effects.
    """
    def fit(self, numeric, categorical):
        self.columns = [c for c in numeric if numeric[c].nunique(dropna=True) > 1]
        self.unused = [c for c in numeric if c not in self.columns]
        self.medians = numeric[self.columns].median()
        filled = numeric[self.columns].fillna(self.medians)
        self.means = filled.mean()
        self.scales = filled.std(ddof=0).replace(0, 1)
        self.categories = DictVectorizer(sparse=True).fit(categorical)
        self.names = (self.columns + [c + "__missing" for c in self.columns]
                      + list(self.categories.get_feature_names_out()))
        return self

    def transform(self, numeric, categorical):
        raw = numeric.reindex(columns=self.columns)
        z = (raw.fillna(self.medians) - self.means) / self.scales
        return sparse.hstack([sparse.csr_matrix(z.to_numpy()),
                              sparse.csr_matrix(raw.isna().to_numpy(dtype=float)),
                              self.categories.transform(categorical)], format="csr")


def score_grid(home_rate, away_rate, rho=0.0, tail=1e-10):
    """Dixon-Coles corrected Poisson grid (not a general bivariate Poisson).

    Negative rho raises 0-0 and 1-1, lowers 1-0 and 0-1. Enforce positive
    correction factors before applying, and report the effective rho.
    """
    h, a = float(home_rate), float(away_rate)
    if not (0 < h <= 100 and 0 < a <= 100 and math.isfinite(rho)):
        raise ValueError("Finite goal rates must lie in (0, 100]")
    lo, hi = max(-1 / h, -1 / a) + 1e-9, min(1.0, 1 / (h * a)) - 1e-9
    effective = float(np.clip(rho, lo, hi))
    size = max(2, int(max(poisson.ppf(1 - tail, h), poisson.ppf(1 - tail, a))) + 1)
    goals = np.arange(size)
    grid = np.outer(poisson.pmf(goals, h), poisson.pmf(goals, a))
    grid[0, 0] *= 1 - h * a * effective
    grid[0, 1] *= 1 + h * effective
    grid[1, 0] *= 1 + a * effective
    grid[1, 1] *= 1 - effective
    mass = float(grid.sum())
    return grid / mass, effective, max(0.0, 1 - mass)


def probabilities(grid):
    h, a = np.indices(grid.shape)
    return {"p_home": float(grid[h > a].sum()), "p_draw": float(grid[h == a].sum()),
            "p_away": float(grid[h < a].sum()), "p_over_25": float(grid[h + a > 2].sum()),
            "p_btts": float(grid[(h > 0) & (a > 0)].sum())}


def market_probabilities(odds):
    odds = np.asarray(odds, dtype=float)
    if odds.shape != (3,) or not np.isfinite(odds).all() or (odds <= 1).any():
        raise ValueError("Provide three finite decimal odds greater than one")
    inverse = 1 / odds
    return inverse / inverse.sum()


class UCLModel:
    def __init__(self, config=None):
        self.config = config or Config()

    def fit(self, history, context=None):
        if len(history) < self.config.min_train_matches:
            raise ValueError(f"Need at least {self.config.min_train_matches} matches")
        self.history = validate_history(history)
        self.features = Features(self.history, context, self.config)
        dates = np.sort(self.history.date.dt.normalize().unique())
        split = dates[max(1, int(len(dates) * .8))] if len(dates) >= 10 else None
        calibration = self.history[self.history.date.dt.normalize() >= split] if split is not None else self.history.iloc[:0]
        train = self.history[self.history.date.dt.normalize() < split] if split is not None else self.history
        if len(train) < self.config.min_train_matches:
            train, calibration = self.history, self.history.iloc[:0]
        numeric, categorical = self.features.build(train)
        self.encoder = Encoder().fit(numeric, categorical)
        X = self.encoder.transform(numeric, categorical)
        y = train[["home_goals", "away_goals"]].to_numpy().ravel()
        ages = (self.history.date.max() - train.date).dt.total_seconds().to_numpy() / 86400
        weights = np.repeat(np.exp2(-ages / self.config.half_life_days), 2)
        self.regression = PoissonRegressor(alpha=self.config.alpha, max_iter=1500, tol=1e-7)
        self.regression.fit(X, y, sample_weight=weights)
        self.rho = 0.0
        if len(calibration) >= 10:
            rates = self.rates(calibration)
            goals = calibration[["home_goals", "away_goals"]].to_numpy()
            low = max(np.max(-1 / rates[:, 0]), np.max(-1 / rates[:, 1])) + 1e-6
            high = min(1.0, np.min(1 / rates.prod(axis=1))) - 1e-6
            def objective(rho):
                h, a = rates.T
                x, y = goals.T
                tau = np.ones(len(goals))
                mask = (x == 0) & (y == 0); tau[mask] = (1 - h * a * rho)[mask]
                mask = (x == 0) & (y == 1); tau[mask] = (1 + h * rho)[mask]
                mask = (x == 1) & (y == 0); tau[mask] = (1 + a * rho)[mask]
                mask = (x == 1) & (y == 1); tau[mask] = 1 - rho
                return -np.log(tau).sum() + self.config.rho_prior_precision * rho ** 2
            optimized = minimize_scalar(objective, bounds=(low, high), method="bounded")
            if optimized.success:
                self.rho = float(optimized.x)
        self.fit_info = {"goal_fit_matches": len(train), "rho_calibration_matches": len(calibration),
                         "rho": self.rho, "history_matches": len(history),
                         "history_cutoff": str(self.history.date.max()),
                         "goal_fit_cutoff": str(train.date.max()),
                         "active_numeric_features": self.encoder.columns,
                         "inactive_numeric_features": self.encoder.unused}
        return self

    def rates(self, fixtures):
        num, cat = self.features.build(fixtures)
        X = self.encoder.transform(num, cat)
        linear = np.asarray(X @ self.regression.coef_).ravel() + self.regression.intercept_
        # Numerical guard, not an empirically calibrated scoring limit.
        return np.exp(np.clip(linear, np.log(.05), np.log(8.0))).reshape(-1, 2)

    def predict(self, fixtures):
        fixtures = validate_fixtures(fixtures)
        rows = []
        if (fixtures.date.dt.normalize() <= self.history.date.max().normalize()).any():
            raise ValueError("Forecast fixtures must be after the training history; use backtest for past matches")
        for fixture, (h, a) in zip(fixtures.to_dict("records"), self.rates(fixtures)):
            grid, rho, tail = score_grid(h, a, self.rho)
            row = {"match_id": fixture["match_id"], "date": str(fixture["date"]),
                   "home": fixture["home"], "away": fixture["away"],
                   "home_goal_rate": float(h), "away_goal_rate": float(a),
                   **probabilities(grid), "effective_rho": rho, "omitted_tail_mass": tail}
            num, _ = self.features.build(pd.DataFrame([fixture]))
            row["missing_active_numeric_inputs"] = int(num[self.encoder.columns].isna().sum().sum())
            row["home_observed_matches"] = num.iloc[0]["observed_matches"]
            row["away_observed_matches"] = num.iloc[1]["observed_matches"]
            flat = np.argsort(grid.ravel())[-5:][::-1]
            row["top_scores"] = {f"{i // grid.shape[1]}-{i % grid.shape[1]}": float(grid.ravel()[i]) for i in flat}
            row["uncalibrated_research_model"] = True
            rows.append(row)
        return rows

    def coefficients(self):
        return pd.DataFrame({"feature": ["intercept"] + self.encoder.names,
                             "coefficient": [self.regression.intercept_] + list(self.regression.coef_)})

    def explain(self, fixture):
        num, cat = self.features.build(pd.DataFrame([fixture]))
        X = self.encoder.transform(num, cat).toarray()
        return {side: {"intercept": float(self.regression.intercept_),
                       "log_rate_contributions": dict(zip(self.encoder.names, (X[i] * self.regression.coef_).tolist()))}
                for i, side in enumerate(("home", "away"))}


def backtest(history, context=None, config=None, folds=3):
    """Expanding-window test; same-day games always stay in one fold.

    Held-out matches never enter model fitting, scaling, or rho calibration.
    Test features stay frozen at each fold's origin (batch forecasting).
    """
    config = config or Config()
    dates = np.sort(history.date.dt.normalize().unique())
    chunks = np.array_split(dates, folds + 1)
    predictions, summaries = [], []
    for i in range(1, len(chunks)):
        if not len(chunks[i]):
            continue
        train = history[history.date.dt.normalize() < chunks[i][0]]
        test = history[history.date.dt.normalize().isin(chunks[i])]
        if len(train) < config.min_train_matches:
            continue
        model = UCLModel(config).fit(train, context)
        rates = model.rates(test)
        baseline = train[["home_goals", "away_goals"]].mean().clip(.05, 8).to_numpy()
        records = []
        for fixture, (h, a) in zip(test.to_dict("records"), rates):
            grid, _, _ = score_grid(h, a, model.rho)
            p = probabilities(grid)
            bp = probabilities(score_grid(*baseline)[0])
            x, y = fixture["home_goals"], fixture["away_goals"]
            label = 0 if x > y else 1 if x == y else 2
            probs = np.array([p[k] for k in ("p_home", "p_draw", "p_away")])
            bprobs = np.array([bp[k] for k in ("p_home", "p_draw", "p_away")])
            record = dict(match_id=fixture["match_id"], date=str(fixture["date"]), fold=i,
                          competition=fixture["competition"], **p,
                          log_loss=float(-np.log(max(probs[label], 1e-15))),
                          brier=float(np.sum((probs - np.eye(3)[label]) ** 2)),
                          baseline_log_loss=float(-np.log(max(bprobs[label], 1e-15))),
                          baseline_brier=float(np.sum((bprobs - np.eye(3)[label]) ** 2)))
            snap = (context or Context()).get(fixture, fixture["home"])
            if all(k in snap and pd.notna(snap[k]) for k in ("odds_home", "odds_draw", "odds_away")):
                market = market_probabilities([snap[k] for k in ("odds_home", "odds_draw", "odds_away")])
                record["market_log_loss"] = float(-np.log(market[label]))
            records.append(record)
        predictions.extend(records)
        f = pd.DataFrame(records)
        summaries.append(dict(fold=i, train_matches=len(train), test_matches=len(test),
                              test_start=str(test.date.min()), test_end=str(test.date.max()),
                              **f[["log_loss", "brier", "baseline_log_loss", "baseline_brier"]].mean().to_dict()))
    if not predictions:
        raise ValueError("Insufficient chronological history for backtesting")
    return pd.DataFrame(predictions), summaries


def simulate_match(home_rate, away_rate, rho=0.0, n=10000, seed=42):
    """Samples the SAME corrected joint score distribution used in forecasts."""
    if n < 1:
        raise ValueError("n must be positive")
    grid, _, _ = score_grid(home_rate, away_rate, rho)
    indices = np.random.default_rng(seed).choice(grid.size, size=n, p=grid.ravel())
    return np.column_stack(np.unravel_index(indices, grid.shape))


def second_leg_qualification(grid, aggregate_home, aggregate_away, home_rate, away_rate,
                             penalty_home_probability=.5):
    """Home/away mean SECOND-LEG teams; aggregate contains the prior leg only.

    No away-goals rule. Ties: 30-minute independent-Poisson ET at 1/3 of
    regulation goal rates, then a supplied shootout probability (default 0.5).
    These extra-time assumptions are explicit, not learned from this dataset.
    """
    if not 0 <= penalty_home_probability <= 1:
        raise ValueError("Invalid shootout probability")
    h, a = np.indices(grid.shape)
    margin = h - a + aggregate_home - aggregate_away
    extra = probabilities(score_grid(home_rate / 3, away_rate / 3)[0])
    tied_win = extra["p_home"] + extra["p_draw"] * penalty_home_probability
    p = float(grid[margin > 0].sum() + grid[margin == 0].sum() * tied_win)
    return {"p_home_qualify": p, "p_away_qualify": 1 - p}


def simulate_bracket(entrants, rounds, predict_rates, n=10000, seed=42):
    """Conditional, fixed knockout bracket; adjacent winners meet next round.

    entrants: confirmed bracket order, a power of two. rounds: one dict per
    round with stage, date, legs=1|2, and second_date for two legs. First entrant
    hosts leg 1; second hosts leg 2. Single-leg final is neutral. Caller callback
    receives fixture and returns (home_rate, away_rate, rho). Callback must apply
    sampled aggregate totals in leg 2. Unknown draws are never invented here.
    Form/squad strength is frozen unless caller updates it. This is conditional
    Monte Carlo uncertainty, not parameter uncertainty.
    """
    size = len(entrants)
    if size < 2 or size & (size - 1) or len(set(entrants)) != size:
        raise ValueError("Distinct entrants in a power-of-two fixed bracket are required")
    if len(rounds) != int(math.log2(size)) or n < 1:
        raise ValueError("One round specification per elimination stage is required")
    rng = np.random.default_rng(seed)
    counts = {team: {"wins_tournament": 0, **{f"wins_round_{i+1}": 0 for i in range(len(rounds))}}
              for team in entrants}
    cache = {}
    def draw(fixture):
        key = json.dumps(fixture, sort_keys=True, default=str)
        if key not in cache:
            rates = predict_rates(fixture)
            cache[key] = (rates, score_grid(*rates)[0])
        rates, grid = cache[key]
        ix = rng.choice(grid.size, p=grid.ravel())
        return ix // grid.shape[1], ix % grid.shape[1], rates
    for _ in range(n):
        alive = list(entrants)
        for r, spec in enumerate(rounds):
            winners = []
            for j in range(0, len(alive), 2):
                first, second = alive[j:j+2]
                legs = spec["legs"]
                if legs not in (1, 2):
                    raise ValueError("legs must be 1 or 2")
                fixture = dict(match_id=f"sim-{r}-{j}-1", date=spec["date"],
                               home=first, away=second, competition="Champions League",
                               stage=spec["stage"], leg="first" if legs == 2 else "single",
                               neutral=int(legs == 1))
                h, a, rates = draw(fixture)
                agg_h, agg_a = h, a
                if legs == 2:
                    fixture.update(match_id=f"sim-{r}-{j}-2", date=spec["second_date"],
                                   home=second, away=first, leg="second", neutral=0,
                                   aggregate_home=a, aggregate_away=h)
                    sh, sa, rates = draw(fixture)
                    first, second = second, first
                    agg_h, agg_a = a + sh, h + sa
                if agg_h == agg_a:
                    agg_h += rng.poisson(rates[0] / 3)
                    agg_a += rng.poisson(rates[1] / 3)
                winner = first if agg_h > agg_a else second if agg_a > agg_h else rng.choice([first, second])
                winners.append(winner)
                counts[winner][f"wins_round_{r+1}"] += 1
            alive = winners
        counts[alive[0]]["wins_tournament"] += 1
    return {team: {key: value / n for key, value in stages.items()} for team, stages in counts.items()}


def simulate_league_phase(completed, remaining, model, teams, n=10000, seed=42):
    """Conditional league-phase Monte Carlo using a COMPLETE 36-team schedule.

    completed: match-level table with scores. remaining: fixture-level table.
    Both must contain ONLY league-phase games. Preserve completed results;
    sample remaining games from the corrected score grids once per fixture.
    Implements UEFA ranking through opponents' collective goals (criterion 8).
    Discipline and coefficient ties are randomized and counted explicitly,
    since future disciplinary totals are not supplied. Thus this is an
    APPROXIMATION, not official ranking when those final criteria are needed.
    Form/injuries and match probabilities are frozen, not path-updated.
    """
    teams = [canonical(team) for team in teams]
    if len(teams) != 36 or len(set(teams)) != 36 or n < 1:
        raise ValueError("Exactly 36 unique teams and a positive simulation count are required")
    schedule = pd.concat([completed, remaining], ignore_index=True)
    if len(schedule) != 144 or schedule.match_id.duplicated().any():
        raise ValueError("Supply all 144 league-phase matches exactly once")
    index = {t: i for i, t in enumerate(teams)}
    if not set(schedule.home).union(schedule.away).issubset(index):
        raise ValueError("Schedule contains an unknown team; canonicalize names first")
    opponents = np.zeros((36, 36), dtype=int)
    for game in schedule.to_dict("records"):
        h, a = index[game["home"]], index[game["away"]]
        if h == a or opponents[h, a]:
            raise ValueError("League phase requires eight distinct opponents, no self fixtures")
        opponents[h, a] = opponents[a, h] = 1
    if not (opponents.sum(axis=1) == 8).all():
        raise ValueError("Each team must have eight opponents")
    if any((schedule.home == t).sum() != 4 for t in teams):
        raise ValueError("Each team must have four home fixtures")
    if len(remaining) and (remaining.date.dt.normalize() <= model.history.date.max().normalize()).any():
        raise ValueError("Remaining fixtures must be after the model's history cutoff")
    rng = np.random.default_rng(seed)
    points, gf, ga, away_goals, wins, away_wins = [np.zeros((n, 36), dtype=int) for _ in range(6)]
    future_rates = model.rates(remaining) if len(remaining) else np.empty((0, 2))
    for i, game in enumerate(schedule.to_dict("records")):
        h, a = index[game["home"]], index[game["away"]]
        if i < len(completed):
            hg = np.full(n, int(game["home_goals"])); ag = np.full(n, int(game["away_goals"]))
        else:
            grid = score_grid(*future_rates[i - len(completed)], model.rho)[0]
            sample = rng.choice(grid.size, n, p=grid.ravel())
            hg, ag = np.unravel_index(sample, grid.shape)
        points[:, h] += 3 * (hg > ag) + (hg == ag)
        points[:, a] += 3 * (ag > hg) + (hg == ag)
        gf[:, h] += hg; ga[:, h] += ag
        gf[:, a] += ag; ga[:, a] += hg
        away_goals[:, a] += ag
        wins[:, h] += hg > ag; wins[:, a] += ag > hg
        away_wins[:, a] += ag > hg
    gd = gf - ga
    criteria = [points, gd, gf, away_goals, wins, away_wins,
                points @ opponents, gd @ opponents, gf @ opponents]
    # np.lexsort uses the LAST key as the primary key.
    order = np.lexsort(tuple([rng.random((n, 36))] + [-x for x in reversed(criteria)]), axis=1)
    positions = np.empty((n, 36), dtype=int)
    positions[np.arange(n)[:, None], order] = np.arange(1, 37)
    unresolved = np.ones((n, 35), dtype=bool)
    for values in criteria:
        sorted_values = np.take_along_axis(values, order, axis=1)
        unresolved &= sorted_values[:, 1:] == sorted_values[:, :-1]
    summary = {team: {"average_points": float(points[:, i].mean()),
                      "average_position": float(positions[:, i].mean()),
                      "p_top8": float((positions[:, i] <= 8).mean()),
                      "p_playoff": float(((positions[:, i] >= 9) & (positions[:, i] <= 24)).mean()),
                      "p_eliminated": float((positions[:, i] > 24).mean())}
               for i, team in enumerate(teams)}
    return {"teams": summary, "simulation_count": n,
            "simulations_with_unresolved_ties": int(unresolved.any(axis=1).sum()),
            "unresolved_qualification_boundary_simulations": int(unresolved[:, [7, 23]].any(axis=1).sum()),
            "tie_policy": "random after UEFA criterion 8; discipline/coefficient omitted",
            "path_assumption": "fixed pre-simulation probabilities; no dynamic rotation/incentive update"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("model_output"))
    args = parser.parse_args()
    if args.alpha <= 0:
        parser.error("--alpha must be positive")
    history = load_history(args.history)
    context = Context(args.context)
    config = Config(alpha=args.alpha)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    fixtures = None
    if args.fixtures:
        fixtures = validate_fixtures(pd.read_csv(args.fixtures))
        history = history[history.date < fixtures.date.min().normalize()]
    if args.backtest:
        test_rows, summary = backtest(history, context, config)
        test_rows.to_csv(out / "backtest_predictions.csv", index=False)
        (out / "backtest_summary.json").write_text(json.dumps(summary, indent=2))
    model = UCLModel(config).fit(history, context)
    model.coefficients().to_csv(out / "learned_coefficients.csv", index=False)
    (out / "model_report.json").write_text(json.dumps(model.fit_info, indent=2))
    if fixtures is not None:
        predictions = model.predict(fixtures)
        (out / "predictions.json").write_text(json.dumps(predictions, indent=2))
        explanations = {f["match_id"]: model.explain(f) for f in fixtures.to_dict("records")}
        (out / "explanations.json").write_text(json.dumps(explanations, indent=2))
    print(json.dumps({"output_dir": str(out), "matches": len(history), "rho": model.rho,
                      "status": "research baseline; assess held-out performance before use"}, indent=2))


if __name__ == "__main__":
    main()
