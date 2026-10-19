"""Train on all of train_test.csv, write validation_predictions.csv and fill data/december_chart_inputs.csv."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from features import DATA_DIR, load_data, market_calendar  # noqa: E402
from model import LinearModel  # noqa: E402


def main() -> None:
    train, valid, december = load_data()
    model = LinearModel(market_calendar(train, valid)).fit(train)

    template = DATA_DIR / "validation_predictions_template.csv"
    submission = pd.read_csv(template)[["load_id"]]
    predicted = pd.Series(model.predict(valid), index=valid["load_id"])
    submission["predicted_rate"] = submission["load_id"].map(predicted).round(2)
    submission.to_csv(ROOT / "validation_predictions.csv", index=False)

    chart_path = DATA_DIR / "december_chart_inputs.csv"
    chart = pd.read_csv(chart_path)
    chart["predicted_rate"] = model.predict(december).round(2)
    chart.to_csv(chart_path, index=False)

    print(f"wrote {len(submission)} validation and {len(chart)} december predictions")


if __name__ == "__main__":
    main()
