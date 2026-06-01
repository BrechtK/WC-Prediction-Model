import pandas as pd

from wc_predictor.config import KnockoutScoringConfig
from wc_predictor.results import calculate_standings, score_submitted_predictions


def test_realised_standings_handle_group_and_knockout_scores() -> None:
    predictions = pd.DataFrame(
        [
            {"match_id": "G1", "player": "Alice", "predicted_team_a_goals": 2, "predicted_team_b_goals": 0, "predicted_qualifier": pd.NA},
            {"match_id": "K1", "player": "Alice", "predicted_team_a_goals": 2, "predicted_team_b_goals": 1, "predicted_qualifier": "France"},
            {"match_id": "G1", "player": "Bob", "predicted_team_a_goals": 1, "predicted_team_b_goals": 0, "predicted_qualifier": pd.NA},
        ]
    )
    matches = pd.DataFrame([{"match_id": "G1", "stage": "group stage"}, {"match_id": "K1", "stage": "round of 16"}])
    results = pd.DataFrame(
        [
            {"match_id": "G1", "team_a_goals_90": 2, "team_b_goals_90": 0},
            {"match_id": "K1", "team_a_goals_90": 1, "team_b_goals_90": 1, "went_to_extra_time": True, "team_a_goals_120": 2, "team_b_goals_120": 1, "qualifier": "France"},
        ]
    )
    scored = score_submitted_predictions(predictions, matches, results, KnockoutScoringConfig())
    standings = calculate_standings(scored)
    assert scored["realised_points"].tolist() == [10, 21, 5]
    assert standings.iloc[0]["player"] == "Alice"
    assert standings.iloc[0]["total_realised_points"] == 31

