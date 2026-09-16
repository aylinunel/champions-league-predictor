#!/usr/bin/env python3
"""Download current Champions League clubs' European and domestic results.

The script uses ESPN's public JSON endpoints and pandas. It writes match-level
CSV files, an Excel workbook, and a coverage report. Missing statistics remain
null; they are never replaced with zero.

Example:
    python champions_league_results.py \
        --season-year 2026 \
        --as-of 2026-09-15 \
        --output-dir football_results
"""

from __future__ import annotations

import argparse
import json
import logging
import time as time_module
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


API_ROOT = "https://site.api.espn.com/apis/site/v2/sports/soccer"
LOGGER = logging.getLogger("champions_league_results")


@dataclass(frozen=True)
class Team:
    espn_id: str
    name: str
    country: str
    domestic_league: str
    domestic_slug: str | None


# 2026/27 league-phase field. The explicit mapping keeps domestic competition
# selection auditable and avoids accidentally mixing cups with league matches.
TEAMS: tuple[Team, ...] = (
    Team("887", "AEK Athens", "Greece", "Greek Super League", "gre.1"),
    Team("359", "Arsenal", "England", "Premier League", "eng.1"),
    Team("362", "Aston Villa", "England", "Premier League", "eng.1"),
    Team("1068", "Atletico Madrid", "Spain", "LaLiga", "esp.1"),
    Team("83", "Barcelona", "Spain", "LaLiga", "esp.1"),
    Team("132", "Bayern Munich", "Germany", "Bundesliga", "ger.1"),
    Team("2980", "Bodo/Glimt", "Norway", "Eliteserien", "nor.1"),
    Team("124", "Borussia Dortmund", "Germany", "Bundesliga", "ger.1"),
    Team("570", "Club Brugge", "Belgium", "Belgian Pro League", "bel.1"),
    Team("2572", "Como", "Italy", "Serie A", "ita.1"),
    Team("436", "Fenerbahce", "Türkiye", "Turkish Super Lig", "tur.1"),
    Team("142", "Feyenoord", "Netherlands", "Eredivisie", "ned.1"),
    Team("432", "Galatasaray", "Türkiye", "Turkish Super Lig", "tur.1"),
    Team("110", "Inter Milan", "Italy", "Serie A", "ita.1"),
    Team("4411", "LASK", "Austria", "Austrian Bundesliga", "aut.1"),
    Team("11420", "RB Leipzig", "Germany", "Bundesliga", "ger.1"),
    Team("175", "Lens", "France", "Ligue 1", "fra.1"),
    Team("166", "Lille", "France", "Ligue 1", "fra.1"),
    Team("364", "Liverpool", "England", "Premier League", "eng.1"),
    Team("382", "Manchester City", "England", "Premier League", "eng.1"),
    Team("360", "Manchester United", "England", "Premier League", "eng.1"),
    Team("114", "Napoli", "Italy", "Serie A", "ita.1"),
    Team("160", "Paris Saint-Germain", "France", "Ligue 1", "fra.1"),
    Team("437", "Porto", "Portugal", "Portuguese Primeira Liga", "por.1"),
    Team("148", "PSV Eindhoven", "Netherlands", "Eredivisie", "ned.1"),
    Team("244", "Real Betis", "Spain", "LaLiga", "esp.1"),
    Team("86", "Real Madrid", "Spain", "LaLiga", "esp.1"),
    Team("104", "Roma", "Italy", "Serie A", "ita.1"),
    Team("21922", "Sabah", "Azerbaijan", "Azerbaijani Premier League", None),
    Team("493", "Shakhtar Donetsk", "Ukraine", "Ukrainian Premier League", None),
    Team("494", "Slavia Prague", "Czechia", "Czech First League", None),
    Team("521", "Slovan Bratislava", "Slovakia", "Slovak Super Liga", None),
    Team("2250", "Sporting CP", "Portugal", "Portuguese Primeira Liga", "por.1"),
    Team("134", "Stuttgart", "Germany", "Bundesliga", "ger.1"),
    Team("510", "Viking", "Norway", "Eliteserien", "nor.1"),
    Team("102", "Villarreal", "Spain", "LaLiga", "esp.1"),
)


STAT_FIELDS = {
    "possessionPct": "possession_pct",
    "totalShots": "shots",
    "shotsOnTarget": "shots_on_target",
    "wonCorners": "corners",
    "passPct": "pass_accuracy",
    "foulsCommitted": "fouls",
    "offsides": "offsides",
    "saves": "saves",
    "yellowCards": "yellow_cards",
    "redCards": "red_cards",
}


class HttpClient:
    """Small standard-library JSON client with bounded retries."""

    def __init__(self, retries: int = 5, timeout: int = 30) -> None:
        self.retries = retries
        self.timeout = timeout

    def get_json(self, url: str) -> dict[str, Any]:
        request = urllib.request.Request(
            url, headers={"User-Agent": "football-results-research/1.0"}
        )
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time_module.sleep(0.75 * attempt)
        raise RuntimeError(f"Could not download {url}: {last_error}")


def parse_as_of(value: str) -> datetime:
    parsed = date.fromisoformat(value)
    return datetime.combine(parsed, time.max, tzinfo=timezone.utc)


def score(competitor: dict[str, Any]) -> int:
    raw = competitor.get("score")
    if isinstance(raw, dict):
        raw = raw.get("value", raw.get("displayValue"))
    return int(float(raw))


def result_code(goals_for: int, goals_against: int) -> str:
    return "W" if goals_for > goals_against else "L" if goals_for < goals_against else "D"


def completed_events(
    payload: dict[str, Any], season_year: int, as_of: datetime
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for event in payload.get("events", []):
        event_date = pd.Timestamp(event["date"])
        if (
            event.get("season", {}).get("year") == season_year
            and event.get("status", {}).get("type", {}).get("completed")
            and event_date <= pd.Timestamp(as_of)
        ):
            output.append(event)
    return output


def event_rows(
    event: dict[str, Any], teams_by_id: dict[str, Team], competition: str, slug: str
) -> list[dict[str, Any]]:
    competitors = event["competitions"][0]["competitors"]
    rows: list[dict[str, Any]] = []
    for current in competitors:
        team_id = str(current["team"]["id"])
        team = teams_by_id.get(team_id)
        if team is None:
            continue
        opponent = next(item for item in competitors if str(item["team"]["id"]) != team_id)
        goals_for, goals_against = score(current), score(opponent)
        rows.append(
            {
                "event_id": str(event["id"]),
                "date": pd.Timestamp(event["date"]).date(),
                "competition": competition,
                "stage": event.get("season", {}).get("slug"),
                "team_id": team_id,
                "team": team.name,
                "country": team.country,
                "venue": "Home" if current.get("homeAway") == "home" else "Away",
                "opponent": opponent["team"]["displayName"],
                "goals_for": goals_for,
                "goals_against": goals_against,
                "result": result_code(goals_for, goals_against),
                "points": 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0,
                "source_url": f"{API_ROOT}/{slug}/summary?event={event['id']}",
            }
        )
    return rows


def fetch_match_stats(
    client: HttpClient, slug: str, event_id: str
) -> dict[str, dict[str, float | None]]:
    payload = client.get_json(f"{API_ROOT}/{slug}/summary?event={event_id}")
    output: dict[str, dict[str, float | None]] = {}
    for entry in payload.get("boxscore", {}).get("teams", []):
        raw_stats = {
            item["name"]: pd.to_numeric(item.get("displayValue"), errors="coerce")
            for item in entry.get("statistics", [])
        }
        values = {column: raw_stats.get(api_name) for api_name, column in STAT_FIELDS.items()}
        if pd.notna(values.get("possession_pct")):
            values["possession_pct"] = float(values["possession_pct"]) / 100
        if pd.notna(values.get("pass_accuracy")):
            pass_value = float(values["pass_accuracy"])
            normalized = pass_value if pass_value <= 1 else pass_value / 100
            # A published 0% value is an unavailable-field placeholder in this feed.
            values["pass_accuracy"] = normalized if normalized > 0 else None
        primary = ("possession_pct", "shots", "corners", "fouls")
        if not any(pd.notna(values.get(field)) and float(values[field]) > 0 for field in primary):
            # Some newly created event pages contain an all-zero statistics template.
            values = {column: None for column in STAT_FIELDS.values()}
        output[str(entry["team"]["id"])] = values
    return output


def add_statistics(
    client: HttpClient,
    frame: pd.DataFrame,
    slug_by_competition: dict[str, str],
    workers: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    jobs = frame[["competition", "event_id"]].drop_duplicates()
    stats_by_event: dict[tuple[str, str], dict[str, dict[str, float | None]]] = {}

    def task(competition: str, event_id: str) -> tuple[tuple[str, str], dict[str, Any]]:
        slug = slug_by_competition[competition]
        return (competition, event_id), fetch_match_stats(client, slug, event_id)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(task, row.competition, row.event_id)
            for row in jobs.itertuples(index=False)
        ]
        for future in as_completed(futures):
            try:
                key, value = future.result()
                stats_by_event[key] = value
            except (RuntimeError, urllib.error.URLError) as exc:
                LOGGER.warning("Statistics request failed: %s", exc)

    stat_records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        values = stats_by_event.get((row.competition, row.event_id), {}).get(row.team_id, {})
        stat_records.append(values)
    stats_frame = pd.DataFrame(stat_records, index=frame.index)
    for column in STAT_FIELDS.values():
        if column not in stats_frame:
            stats_frame[column] = pd.NA
    stats_frame["stats_available"] = stats_frame["shots"].notna()
    return pd.concat([frame, stats_frame], axis=1)


def collect_results(
    client: HttpClient, season_year: int, as_of: datetime, workers: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    teams_by_id = {team.espn_id: team for team in TEAMS}

    # UEFA Champions League results, including qualifying rounds and league phase.
    champions_payload = client.get_json(
        f"{API_ROOT}/uefa.champions/scoreboard?dates={season_year}&limit=1000"
    )
    champions_rows = [
        row
        for event in completed_events(champions_payload, season_year, as_of)
        for row in event_rows(event, teams_by_id, "Champions League", "uefa.champions")
    ]
    champions = pd.DataFrame(champions_rows)

    # Domestic league results are fetched once per competition, then filtered by team ID.
    domestic_rows: list[dict[str, Any]] = []
    slug_to_teams: dict[str, list[Team]] = {}
    for team in TEAMS:
        if team.domestic_slug:
            slug_to_teams.setdefault(team.domestic_slug, []).append(team)

    for slug, league_teams in slug_to_teams.items():
        competition_name = league_teams[0].domestic_league
        try:
            payload = client.get_json(
                f"{API_ROOT}/{slug}/scoreboard?dates={season_year}&limit=1000"
            )
        except (RuntimeError, urllib.error.URLError) as exc:
            LOGGER.warning("Domestic league unavailable (%s): %s", competition_name, exc)
            continue
        for event in completed_events(payload, season_year, as_of):
            domestic_rows.extend(event_rows(event, teams_by_id, competition_name, slug))

    domestic = pd.DataFrame(domestic_rows)
    slug_by_competition = {"Champions League": "uefa.champions"}
    slug_by_competition.update(
        {team.domestic_league: team.domestic_slug for team in TEAMS if team.domestic_slug}
    )
    champions = add_statistics(client, champions, slug_by_competition, workers)
    domestic = add_statistics(client, domestic, slug_by_competition, workers)

    champions_team = champions["team"] if "team" in champions else pd.Series(dtype="string")
    domestic_team = domestic["team"] if "team" in domestic else pd.Series(dtype="string")
    domestic_shots = domestic["shots"] if "shots" in domestic else pd.Series(dtype=float)
    coverage = pd.DataFrame(
        [
            {
                "team": team.name,
                "country": team.country,
                "domestic_league": team.domestic_league,
                "domestic_source_supported": bool(team.domestic_slug),
                "champions_league_matches": int((champions_team == team.name).sum()),
                "domestic_matches": int((domestic_team == team.name).sum()),
                "domestic_matches_with_shot_data": int(
                    ((domestic_team == team.name) & domestic_shots.notna()).sum()
                ),
            }
            for team in TEAMS
        ]
    ).sort_values("team")

    sort_columns = ["team", "date"]
    return (
        champions.sort_values(sort_columns, ignore_index=True),
        domestic.sort_values(sort_columns, ignore_index=True),
        coverage.reset_index(drop=True),
    )


def write_outputs(
    champions: pd.DataFrame,
    domestic: pd.DataFrame,
    coverage: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    champions.to_csv(output_dir / "champions_league_results.csv", index=False)
    domestic.to_csv(output_dir / "domestic_league_results.csv", index=False)
    coverage.to_csv(output_dir / "coverage_report.csv", index=False)

    with pd.ExcelWriter(output_dir / "football_results.xlsx") as writer:
        champions.to_excel(writer, sheet_name="Champions League", index=False)
        domestic.to_excel(writer, sheet_name="Domestic leagues", index=False)
        coverage.to_excel(writer, sheet_name="Coverage", index=False)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season-year", type=int, default=2026)
    parser.add_argument("--as-of", default=date.today().isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path, default=Path("football_results"))
    parser.add_argument("--workers", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    client = HttpClient()
    champions, domestic, coverage = collect_results(
        client=client,
        season_year=args.season_year,
        as_of=parse_as_of(args.as_of),
        workers=max(1, args.workers),
    )
    write_outputs(champions, domestic, coverage, args.output_dir)
    LOGGER.info(
        "Saved %d Champions League rows and %d domestic rows to %s",
        len(champions),
        len(domestic),
        args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
