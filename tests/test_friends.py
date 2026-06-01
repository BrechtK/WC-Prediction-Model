from pathlib import Path

from wc_predictor.market_data import load_odds, load_predictions
from wc_predictor.workflow import run_prediction_workflow


EXAMPLES = Path("data/examples")


def test_friend_predictions_receive_expected_value_analysis() -> None:
    result = run_prediction_workflow(
        load_odds(EXAMPLES / "example_odds.csv"),
        load_predictions(EXAMPLES / "example_predictions.csv"),
    )
    assert len(result.friend_report) == 9
    assert (result.friend_report["model_expected_points"] >= 1).all()
    assert {"candidate_rank", "is_consensus", "is_contrarian"}.issubset(result.friend_report.columns)

