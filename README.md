# Champions League Predictor

Python results collector and research prediction model, with 26-factor feature
support, chronological evaluation, Dixon–Coles scoring and conditional simulations.

**Status:** research baseline. On the bundled 18-match Champions League holdout,
the model performs worse than the simple baseline. See the evaluation below.

Repository includes the collector, model, tests, input templates, example data,
and recorded evaluation results. [PUBLISH.md](PUBLISH.md) explains cloning and
pushing updates to this repository.

```bash
git clone https://github.com/aylinunel/champions-league-predictor.git
cd champions-league-predictor
```

An executable research model built on the previous `champions_league_results.py`
collector and `football_stats_data.json`, incorporating the architecture in the
uploaded `files (11).zip` and the 26-factor specification in `Pasted markdown.md`.

The attachment lists candidate features, not numerical coefficients. This model
learns coefficients from historical matches rather than asserting predetermined
advantages for particular clubs, managers, leagues, or tactical narratives.
The existing snapshot is dated 15 September 2026; it is reused here without a new
data refresh or independent re-verification of every result.

## Run it

Python 3.10 or later. Install dependencies, then run from the extracted directory:

```bash
python -m pip install -r requirements.txt
python ucl_model.py --history football_stats_data.json --backtest --output-dir model_output
```

The archive includes the previous dataset for a reproducible run. To use refreshed
results from the existing collector:

```bash
python champions_league_results.py --season-year 2026 --output-dir football_results
python ucl_model.py --history football_results --backtest --output-dir model_output
```

That directory must contain `champions_league_results.csv` and
`domestic_league_results.csv`. To predict fixtures, create `fixtures.csv` using
the header in `fixtures_template.csv`, then run:

```bash
python ucl_model.py --history football_stats_data.json --fixtures fixtures.csv --output-dir predictions
```

For enriched data add `--context context.csv`. Templates contain headers only,
so no invented fixtures, coefficients, injuries or odds are presented as facts.
Use exact match IDs from the canonical match table; the JSON/collector adapter
builds IDs as `competition:event_id`. Use full names or the uploaded framework's
team codes (e.g. GAL, ARS, BAY). Names are normalized and aliases reconciled.

Outputs:

- `learned_coefficients.csv`: fitted log-goal-rate coefficients, including team,
  opponent, competition, tournament-stage and home effects.
- `model_report.json`: sample counts, fitting dates, fitted Dixon–Coles rho,
  active features and inactive features.
- `backtest_predictions.csv` and `backtest_summary.json`: genuine held-out
  evaluations against a historical home/away scoring-average baseline.
- `predictions.json` and `explanations.json` when fixtures are provided: goal
  rates, home/draw/away probabilities, over 2.5, both teams to score, most likely
  scores and each feature's contribution to the log goal rate.

Goal rates here are forecasts of goals, **not observed shot-based xG**. Corners
and shots on target are historical predictive inputs; this model does not yet
produce separate corner-count or shots-on-target-count forecasts.

## Algorithm

1. Convert both data sources into one row per match. Reconcile aliases, check
   duplicated scores and remove duplicate perspectives of the same fixture.
2. Build each team's features using only matches before the forecast day.
   The day cutoff deliberately excludes same-day games because the collector
   discards kickoff times in its CSV exports.
3. Calculate recent goals, conceded goals, points, shots, SOT, corners and other
   reported statistics; competition-specific form; shrunk head-to-head and
   stage form; observed rest and fixture counts over 7, 14, 21 and 30 days.
4. Attach match-specific, timestamped contextual features available before
   kickoff. Use no retrospective injury, ranking or market information.
5. Learn two goal intensities through one regularized Poisson regression with
   two observations per match, one per team. Shared team attack/opponent defence
   categorical effects control for opposition in the conditional goal model.
6. Estimate the Dixon–Coles low-score correction on a later calibration block.
   Keep that block separate from the goal-regression fit. The goal regression
   is intentionally not refitted after rho calibration.
7. Normalize the score grid; sum cells for outcome probabilities. Simulations
   sample the same joint distribution, with explicit knockout tie handling.
8. Evaluate on strictly later, disjoint date blocks. Never shuffle matches.

For team i against j:

```text
log(lambda_i) = intercept + attack_i + defence_j
                + competition + stage + leg + European team/home terms
                + beta · standardized_pre_match_features

objective = weighted_mean(Poisson_negative_log_likelihood)
            + alpha / 2 * sum(beta²)

recency_weight = 2 ** (-age_in_days / half_life_days)

P(H=x,A=y) ∝ Poisson(x; lambda_H) * Poisson(y; lambda_A)
             * tau(x,y; lambda_H, lambda_A,rho)
```

The categorical terms are also regularized. Numeric features are imputed and
standardized using fitting data only; missingness indicators are retained.
Entirely missing or constant features are inactive. Providing an injury field
only for tomorrow's match does not produce a learned injury adjustment: that
field needs historical observations and a new fit.

Default hyperparameters are **design choices**, not calibrated facts: ridge
alpha=1, time half-life=120 days, recent-form window=10 games, form-shrinkage
prior=5 matches, schedule-load half-life=5 days and rho prior precision=25.
Choose them on earlier rolling validation data, then use a separate untouched
test set. The delivered backtest does not perform hyperparameter search.

## Mapping all 26 requested factors

| # | Factor | Implementation and evidence required |
|---|---|---|
| 1 | Tournament stage | Stage categories and shrunk historical stage form. Current opener-only data cannot identify a knockout effect. |
| 2 | Aggregate dynamics | Explicit second-leg aggregate lead, learned effect where training examples exist, extra time and penalties. Away-goals legacy exposure is an optional hypothesis field, with no fixed bonus. |
| 3 | League-phase position | Dated CL rank, point gaps to 8th/24th, matches remaining, matchday and dead-rubber inputs. Never infer mathematical qualification from rank alone. |
| 4 | Draw and scheduling | Observed schedule load plus supplied future opponent and bracket strength. Fixed-bracket simulation supports actual drawn paths. |
| 5 | Squad depth | Dated depth score, quality dropoff, expected rotation, available squad share and rest-depth interaction. Player contribution estimation requires an upstream player model. |
| 6 | Absence impact | Historical `key_absences_impact`; no invented injury data or conversion of player market value into goal impact. |
| 7 | Continuity | Historical continuity-of-minutes share, supplied at each forecast. |
| 8 | Manager | Historical CL match count and precomputed out-of-sample managerial residual. No name-based reputation multiplier. |
| 9 | Congestion | Observed matches in 7/14/21/30 days and decaying load; optional actual distance, high-intensity running and sprint totals. |
| 10 | Travel | Supplied travel kilometres, signed time-zone shift and eastbound flag, allowing actual/neutral venues. No unverified stadium coordinates imported. |
| 11 | Rest | Days since last observed match and interaction with squad depth. Missing competitions mean this can overstate true rest. |
| 12 | Pressing/defence | Historical PPDA, line height, compactness; own pressing × opponent progression interaction. |
| 13 | Progression | Historical progressive passes, carries and cross share. |
| 14 | Set pieces | Historical corners plus optional set-piece xG for/against. Corners are not relabelled as set-piece xG. |
| 15 | Game-state response | Optional pre-match historical xG/xGA per 90 while leading, level or trailing. Requires correctly timed events and actual exposure minutes. No synthetic event timing. |
| 16 | Domestic situation | Observed domestic results/form, plus supplied full-league position percentile and title pressure. The tracked-team subset cannot reconstruct a full league table. |
| 17 | Head-to-head | Prior meetings only; point differential shrunk toward zero using a stated prior sample. |
| 18 | Market | Removes overround from dated decimal odds and compares matched held-out log loss. Used as an external benchmark, not blended into the score distribution. No movement model without timestamped historical odds. |
| 19 | UEFA/finances | Dated club coefficient POINTS, association points, wages and revenue. Ranking is distinct from points. Uploaded ranks/placeholders are not automatically trusted or imported. |
| 20 | Momentum | Recent decayed points/goals and unbeaten run. Coefficients determine whether signal survives regularization. |
| 21 | Home/crowd | Learned home advantage and European home-team effect; optional dated crowd capacity share. Neutral fixtures remove home advantage. |
| 22 | Referee | Supplied historical card/penalty rates computed using only earlier assignments. |
| 23 | Opponent adjustment | Joint fitted attack/defence effects and opponent features; not an independent adjusted-xG estimator. Cross-league comparisons remain weak with few European links. |
| 24 | Score dependence | Dixon–Coles joint score probabilities. A general bivariate Poisson/copula or live event-state hazard model is not implemented. |
| 25 | Dixon–Coles | Learned rho on a later calibration block; positivity constraints and a negligible adaptive score-grid tail. |
| 26 | Simulation | Match, league-phase qualification and confirmed knockout bracket simulations. Full draw generation and dynamic entire-season path evolution are not implemented. |

These are candidate predictive associations, not causal effects or proven
advantages. The file's claims about psychology, manager overperformance,
fatigue and market mispricing still require empirical validation.

## Inputs and missing-data rules

Canonical match CSV columns:
`match_id,date,home,away,home_goals,away_goals,competition,stage,leg,neutral`.
Use ISO dates/timestamps, `Champions League` for the European competition,
`league_phase`, `round_of_16`, `quarter_final`, `semi_final`, `final` for stages,
`single`, `first`, `second` for legs, and 0/1 for neutral. Optional raw statistics
use `home_`/`away_` prefixes followed by fields listed in `STATS`.
If adding cups, internationals or other competitions, set `is_domestic_league`
explicitly to 0/1; absent that field, non-CL rows are assumed to be domestic
league results, matching the original collector's scope.
Second legs also need `aggregate_home` and `aggregate_away`, both referring to
the teams at home/away **in the second leg**, excluding that second-leg score.

All model goal targets must be **90-minute regulation scores**. The earlier
collector may return extra-time-inclusive knockout results. It lacks an
explicit regulation-score field: verify/replace those scores before training
on older knockout matches. The bundled league/opener snapshot has no knockout
games. Season-wide historical harvesting is an extension to the earlier
collector, not performed by this offline algorithm.

Context rows require `match_id,team,available_at,source_url`, plus selected
fields in `CONTEXT_FIELDS` and optionally `odds_home,odds_draw,odds_away` on the
home-team row. Rows are full snapshots. The latest eligible row strictly before
kickoff is used. Every feature must represent knowledge available at that time,
even if its CSV was assembled later. Timestamps gate availability; the code
cannot independently audit whether a provider's historical numbers were revised.
Store xG and PPDA separately; use shares in 0–1, financial values in EUR millions,
distances in km and odds in decimal format.

There are no invented defaults for player tracking, injuries or tactical data.
Zero goals, corners and SOT remain legitimate zeros. Unpublished all-zero
box-score templates and unavailable 0% pass accuracy are treated as missing.
Imputation is a modeling operation, not a claim that the source supplied data.

The source contains current CL clubs' games, not every domestic fixture, cup,
international match or player-minute record. Opponent histories and workload
are consequently incomplete. Training only on this selected set can also bias
league baselines. Expand to all clubs' fixtures across several seasons and
all competitions before production evaluation. The collector's current-team
mapping is hard-coded to 2026/27 and must be maintained for future seasons.

## Incorporated improvements to the uploaded ZIP

The supplied nine modules were reviewed. Their five-part feature decomposition,
decaying fatigue/load concept, Haversine/travel interface idea, aggregate-state
orientation and Dixon–Coles equations informed this extension. We use the same
factor names where practical and provide `from_phase1(matches)` to convert the
uploaded `core.Match` objects and short team codes to the new match table.
That bridge accepts only completed matches; never feed it the demo's synthetic
fixtures, which default to 0–0 results. Original zero-default xG and physical
fields are deliberately not copied as observations.

Changes addressing concrete issues:

- The ZIP is flat, but imports refer to `features.*`. This extension is a
  standalone module, avoiding those broken imports.
- Fixed scoring multipliers and rank-to-goal lookup buckets are replaced by
  fitted coefficients. No calibration evidence accompanied the original weights.
- The uploaded `uefa_coeff` stores ranks, with `999` as a placeholder, not UEFA
  coefficient points. Those values and unverified venue/financial fields are
  excluded from default model inputs.
- Negative rho increases 0–0 and 1–1 but decreases 1–0 and 0–1; it does not
  increase all four. The docstring and probability tests reflect that.
- The simulator samples the corrected distribution, rather than switching to
  independent Poisson draws after calculating corrected predictions.
- A 30%/80% data-availability flag is not model confidence. Outputs expose
  sample and missing-input counts, without a fabricated confidence percentage.
- League simulation preserves completed results and validates the full
  eight-opponent schedule. Points-only alphabetical/insertion-order ties are removed.

## Simulation API

`simulate_match(h, a, rho, n, seed)` returns sampled score pairs.

`second_leg_qualification(grid, aggregate_home, aggregate_away, h, a)` computes
the qualification probability, with a 30-minute extra-time approximation and
50/50 shootout prior (optional alternative supplied). It does not apply an
away-goals tiebreak.

`simulate_league_phase(completed, remaining, model, teams, n, seed)` requires
the complete 144-game schedule. It applies UEFA ranking through opponent
collective goals. Future disciplinary totals are unavailable, so unresolved
ties after that criterion are randomized and counted, including ties at
qualification boundaries. It reports separate top-8, playoff and eliminated
probabilities. It must not be labelled an exact official ranking simulation.

`simulate_bracket(entrants, rounds, predict_rates, n, seed)` simulates a confirmed
fixed knockout bracket. Adjacent entrants meet; adjacent winners meet next.
For each two-leg tie, the second entrant hosts leg 2. Configure the actual draw
and home ordering. Each round has `stage,date,legs`, and `second_date` for two
legs. The callback returns `(home_goal_rate, away_goal_rate, rho)` from a fixture:

```python
import pandas as pd
from ucl_model import UCLModel, load_history, simulate_bracket

model = UCLModel().fit(load_history("football_stats_data.json"))

def predict_rates(fixture):
    h, a = model.rates(pd.DataFrame([fixture]))[0]
    return h, a, model.rho

# Supply confirmed entrants and dates from an actual knockout draw:
# result = simulate_bracket(entrants, rounds, predict_rates, n=10000, seed=42)
```

The callback receives each simulated first-leg aggregate before predicting
leg 2. No away-goals rule is applied. Extra time uses one-third of regulation
rates and penalties default to 50/50. These are stated simplifications.
Simulated fatigue, injuries, lineups, form, positional incentives and draw paths
are not automatically re-estimated inside simulations. There is no invented
current-season tournament-winning table. End-to-end league-to-final forecasts
need the remaining schedule, official draw/bracket adapter and path-dependent
feature updates. Monte Carlo error also does not measure parameter uncertainty.

## Evaluation and evidence

Observed results for the bundled snapshot (207 distinct matches):

| Held-out set | Games | Model log loss | Baseline log loss | Model Brier | Baseline Brier |
|---|---:|---:|---:|---:|---:|
| All competitions pooled | 157 | 0.9468 | 1.0404 | 0.5393 | 0.6312 |
| Champions League only | 18 | 1.2435 | 0.9481 | 0.6970 | 0.5642 |

The richer model **loses to the baseline on the CL subset**. Do not infer
Champions League predictive superiority from pooled performance. This run
supports the pipeline working, not the proposed features being validated.

`example_run/` is generated from the existing dataset, not the supplied synthetic
demo. It reports the actual held-out scores and fitted coefficients. Smaller
log loss/Brier scores are better. Brier is the sum across three outcome classes,
so its range is 0–2. Use `competition == Champions League` to assess European
games separately; pooled improvement does not establish Champions League skill.
The baseline estimates home/away scoring rates from earlier results only.
No betting-market comparison is claimed without supplied odds.

Each outer fold refits scaling, coefficients and rho on preceding dates only.
Within a fold, historical-result features are frozen at the fold origin; dated
context is still filtered to each forecast kickoff. The first fold is skipped
when its training sample is below 30 matches. Final fitting reserves roughly
the last 20% of distinct days for rho calibration, when samples allow it.

The data is too short to validate all 26 factors, distinguish club-stage
effects reliably or establish profitability. Add multiple seasons and evaluate
calibration, log loss, Brier, seasonal stability and feature ablations on
untouched future blocks before describing probabilities as calibrated.

Run the behavioral tests:

```bash
python -m unittest -v test_model.py
```

Method/rule references checked 16 September 2026:

- Dixon & Coles (1997), model origin: https://doi.org/10.1111/1467-9876.00065
- UEFA 2026/27 Article 17, league format:
  https://documents.uefa.com/r/Regulations-of-the-UEFA-Champions-League-2026/27/Article-17-Match-system-league-phase-Online
- UEFA 2026/27 Article 18, ranking criteria:
  https://documents.uefa.com/r/Regulations-of-the-UEFA-Champions-League-2026/27/Article-18-Equality-of-points-league-phase-Online
- UEFA 2026/27 regulations, knockout rules:
  https://documents.uefa.com/r/Regulations-of-the-UEFA-Champions-League-2026/27-Online
