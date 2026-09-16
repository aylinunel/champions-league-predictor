"""Behavioral regression tests: leakage, missingness, aggregation and probability math."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ucl_model import (Context, Encoder, Features, UCLModel, canonical, load_history,
                       market_probabilities, probabilities, score_grid,
                       second_leg_qualification, simulate_bracket,
                       simulate_league_phase, simulate_match, utc, validate_history)


def history(n=80):
    rng = np.random.default_rng(120)
    rows = []
    names = ["arsenal", "bayern munich", "galatasaray", "barcelona"]
    for i in range(n):
        h, a = rng.choice(names, 2, replace=False)
        rows.append(dict(match_id=str(i), date=utc("2025-01-01") + pd.Timedelta(days=i),
                         home=h, away=a, home_goals=int(rng.poisson(1.6)),
                         away_goals=int(rng.poisson(1.1)), competition="Champions League",
                         stage="league_phase", home_shots=10 + i % 4, away_shots=8 + i % 3))
    return validate_history(pd.DataFrame(rows))


class ModelTests(unittest.TestCase):
    def test_source_adapter_deduplicates_and_keeps_missing_stats(self):
        raw = dict(eventId="1", date="2026-09-09T20:00Z", name="Inter Milan",
                   opponent="Bayern Munich", venue="Home", gf=1, ga=0,
                   shots=0, possession=0, corners=0, fouls=0)
        reverse = dict(raw, name="Bayern Munich", opponent="Internazionale",
                       venue="Away", gf=0, ga=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.json"
            path.write_text(json.dumps(dict(championsLeague=[raw, reverse], domesticMatches=[])))
            frame = load_history(path)
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0].home_goals, 1)
            self.assertTrue(pd.isna(frame.iloc[0].home_shots))
            reverse["ga"] = 2
            path.write_text(json.dumps(dict(championsLeague=[raw, reverse], domesticMatches=[])))
            with self.assertRaises(ValueError):
                load_history(path)

    def test_score_distribution_and_correction_sign(self):
        baseline = score_grid(1.4, 1.1, 0)[0]
        corrected = score_grid(1.4, 1.1, -.1)[0]
        self.assertGreater(corrected[0, 0], baseline[0, 0])
        self.assertGreater(corrected[1, 1], baseline[1, 1])
        self.assertLess(corrected[1, 0], baseline[1, 0])
        self.assertLess(corrected[0, 1], baseline[0, 1])
        for h, a, rho in ((.05, 8, -20), (8, 8, 3), (1.2, 1.2, -.1)):
            grid, _, tail = score_grid(h, a, rho)
            self.assertTrue((grid >= 0).all())
            self.assertAlmostEqual(grid.sum(), 1)
            p = probabilities(grid)
            self.assertAlmostEqual(p["p_home"] + p["p_draw"] + p["p_away"], 1)
            self.assertLess(tail, 3e-10)

    def test_sampling_uses_corrected_joint_distribution(self):
        scores = simulate_match(1.4, 1.1, -.1, n=60000, seed=42)
        expected = probabilities(score_grid(1.4, 1.1, -.1)[0])["p_draw"]
        self.assertAlmostEqual(np.mean(scores[:, 0] == scores[:, 1]), expected, delta=.008)
        np.testing.assert_array_equal(scores[:100], simulate_match(1.4, 1.1, -.1, n=100, seed=42))

    def test_features_exclude_current_and_future_matches(self):
        data = history()
        fixture = data.iloc[40].to_dict()
        before = Features(data).build(pd.DataFrame([fixture]))[0]
        changed = data.copy()
        changed.loc[changed.date >= fixture["date"], ["home_goals", "away_goals", "home_shots"]] = 99
        after = Features(changed).build(pd.DataFrame([fixture]))[0]
        pd.testing.assert_frame_equal(before, after)

    def test_context_cannot_use_post_kickoff_snapshot(self):
        fixture = dict(match_id="test", date=utc("2025-04-01T20:00Z"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "context.csv"
            pd.DataFrame([
                dict(match_id="test", team="ARS", available_at="2025-04-01T18:00Z", source_url="test", squad_depth_score=.8),
                dict(match_id="test", team="ARS", available_at="2025-04-01T21:00Z", source_url="test", squad_depth_score=.1),
            ]).to_csv(path, index=False)
            self.assertEqual(Context(path).get(fixture, "arsenal")["squad_depth_score"], .8)

    def test_untrained_features_have_no_effect(self):
        numeric = pd.DataFrame({"form": [1., 2., 3.], "injury": [np.nan] * 3})
        cats = [{"team": "a"}] * 3
        encoder = Encoder().fit(numeric, cats)
        a = encoder.transform(numeric, cats)
        numeric["injury"] = 100
        b = encoder.transform(numeric, cats)
        np.testing.assert_array_equal(a.toarray(), b.toarray())
        self.assertIn("injury", encoder.unused)

    def test_second_leg_aggregate_orientation_and_no_away_goals(self):
        grid = np.zeros((4, 4)); grid[0, 1] = 1
        # Leg 1 was away 2-1; a 0-1 second-leg result ties aggregate 2-2.
        # Away goals do NOT award the tie to either side.
        result = second_leg_qualification(grid, 2, 1, 1, 1)
        self.assertAlmostEqual(result["p_home_qualify"], .5)
        self.assertEqual(second_leg_qualification(grid, 4, 0, 1, 1)["p_home_qualify"], 1)

    def test_fixed_bracket_probability_conservation(self):
        result = simulate_bracket(["a", "b", "c", "d"],
            [dict(stage="semi_final", date="2027-05-01", second_date="2027-05-08", legs=2),
             dict(stage="final", date="2027-05-29", legs=1)],
            lambda fixture: (1.2, 1.2, -.08), n=600, seed=12)
        self.assertAlmostEqual(sum(r["wins_tournament"] for r in result.values()), 1)
        self.assertAlmostEqual(sum(r["wins_round_1"] for r in result.values()), 2)
        self.assertTrue(all(.15 < r["wins_tournament"] < .35 for r in result.values()))

    def test_league_simulation_preserves_completed_results_and_reports_ties(self):
        teams = [f"team {i}" for i in range(36)]
        rows = []
        for i in range(36):
            for step in range(1, 5):
                rows.append(dict(match_id=f"{i}-{step}", date=utc("2026-09-01"),
                                 home=teams[i], away=teams[(i + step) % 36],
                                 home_goals=0, away_goals=0))
        completed = pd.DataFrame(rows)
        result = simulate_league_phase(completed, completed.iloc[:0], None, teams, n=100)
        self.assertTrue(all(r["average_points"] == 8 for r in result["teams"].values()))
        self.assertAlmostEqual(sum(r["p_top8"] for r in result["teams"].values()), 8)
        self.assertEqual(result["unresolved_qualification_boundary_simulations"], 100)
        with self.assertRaises(ValueError):
            simulate_league_phase(completed.iloc[:-1], completed.iloc[:0], None, teams)

    def test_end_to_end_fitting_and_historical_prediction_guard(self):
        data = history()
        model = UCLModel().fit(data)
        fixture = data.iloc[0].copy()
        fixture["date"] = utc("2025-05-01")
        result = model.predict(pd.DataFrame([fixture]))[0]
        self.assertAlmostEqual(result["p_home"] + result["p_draw"] + result["p_away"], 1)
        self.assertGreater(model.fit_info["rho_calibration_matches"], 0)
        with self.assertRaises(ValueError):
            model.predict(data.iloc[:1])

    def test_aliases_and_market_margin_removal(self):
        self.assertEqual(canonical("BAY"), canonical("Bayern München"))
        self.assertEqual(canonical("Fenerbahçe"), canonical("FEN"))
        self.assertEqual(canonical("Bodø/Glimt"), canonical("BOD"))
        self.assertAlmostEqual(market_probabilities([2.0, 3.0, 4.0]).sum(), 1)
        with self.assertRaises(ValueError):
            market_probabilities([1.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
