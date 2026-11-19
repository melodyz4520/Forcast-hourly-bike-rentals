import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from clean import OUTPUT, TARGETS

CUTOFF = pd.Timestamp("2012-09-01")
CATEGORICAL = [
    "hr", "weekday", "workingday", "holiday",
    "season", "mnth", "weathersit", "hr_day",
]
NUMERIC = ["yr", "temp", "atemp", "hum", "windspeed"]
FEATURES = CATEGORICAL + NUMERIC


def build_models():
    ridge_inputs = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("numeric", StandardScaler(), NUMERIC),
    ])
    boosting_inputs = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL),
        ("numeric", "passthrough", NUMERIC),
    ])
    return {
        "ridge": make_pipeline(ridge_inputs, Ridge(alpha=10)),
        "gradient_boosting": make_pipeline(
            boosting_inputs,
            HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.08,
                max_leaf_nodes=31,
                early_stopping=False,
                random_state=42,
            ),
        ),
    }


def main():
    cleaned = OUTPUT / "clean_hourly.csv"
    if not cleaned.is_file():
        raise SystemExit("Run python3 clean.py before training.")

    data = pd.read_csv(cleaned, parse_dates=["timestamp"]).sort_values("timestamp")
    data["hr_day"] = data["hr"] + 24 * data["workingday"]
    train = data[data["timestamp"] < CUTOFF]
    test = data[data["timestamp"] >= CUTOFF]
    if train.empty or test.empty:
        raise ValueError("The dataset must contain records on both sides of the cutoff.")
    print(f"Training rows: {len(train):,} | Test rows: {len(test):,}")

    keys = ["workingday", "hr"]
    historical = train.groupby(keys)[TARGETS].mean()
    test_groups = pd.MultiIndex.from_frame(test[keys])
    predictions = test[["timestamp", *keys, *TARGETS, "cnt"]].copy()
    rows = []

    for target in TARGETS:
        baseline = historical[target].reindex(test_groups)
        estimates = {
            "historical_average": baseline.fillna(train[target].mean()).to_numpy()
        }
        for name, model in build_models().items():
            model.fit(train[FEATURES], train[target])
            estimates[name] = np.maximum(model.predict(test[FEATURES]), 0)

        actual = test[target]
        baseline_rmse = np.sqrt(mean_squared_error(actual, estimates["historical_average"]))
        for name, estimate in estimates.items():
            predictions[f"{target}_{name}"] = estimate
            rmse = np.sqrt(mean_squared_error(actual, estimate))
            rows.append({
                "rider": target,
                "model": name,
                "MAE": mean_absolute_error(actual, estimate),
                "RMSE": rmse,
                "R2": r2_score(actual, estimate),
                "RMSE_reduction_pct": 100 * (1 - rmse / baseline_rmse),
            })

    for name in ["historical_average", "ridge", "gradient_boosting"]:
        predictions[f"total_{name}"] = (
            predictions[f"casual_{name}"] + predictions[f"registered_{name}"]
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "metrics.csv", index=False)
    predictions.to_csv(OUTPUT / "test_predictions.csv", index=False)
    print(metrics.round(3).to_string(index=False))


if __name__ == "__main__":
    main()