import os
import warnings
from pathlib import Path

# Prevent joblib from trying (and failing) to infer the physical CPU count on
# some macOS/Python installations. A value below the logical CPU count makes
# joblib respect this explicit limit instead of calling the unavailable sysctl.
# This must be set before importing sklearn.
if "LOKY_MAX_CPU_COUNT" not in os.environ:
    logical_cpus = os.cpu_count() or 1
    os.environ["LOKY_MAX_CPU_COUNT"] = str(max(1, logical_cpus - 1))
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results"
TARGETS = ["casual", "registered"]
TEST_CUTOFF = pd.Timestamp("2012-09-01")

# These expanding windows are entirely before the final test period. Models are
# selected by their average validation RMSE, not by repeatedly checking test R2.
VALIDATION_WINDOWS = [
    (pd.Timestamp("2011-12-01"), pd.Timestamp("2012-03-01")),
    (pd.Timestamp("2012-03-01"), pd.Timestamp("2012-06-01")),
    (pd.Timestamp("2012-06-01"), pd.Timestamp("2012-09-01")),
]

CATEGORICAL = [
    "hr",
    "weekday",
    "workingday",
    "holiday",
    "season",
    "mnth",
    "weathersit",
    "hr_day",
]

# atemp is intentionally excluded because it is almost perfectly correlated
# with temp and its removal improved expanding-window validation performance.
NUMERIC = ["yr", "temp", "hum", "windspeed"]
FEATURES = CATEGORICAL + NUMERIC
MODEL_NAMES = [
    "squared_error_boosting",
    "poisson_boosting",
    "log_target_boosting",
]


def build_model(loss="squared_error"):
    inputs = ColumnTransformer(
        [
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
            ("numeric", "passthrough", NUMERIC),
        ]
    )
    estimator = HistGradientBoostingRegressor(
        loss=loss,
        max_iter=250,
        learning_rate=0.08,
        max_leaf_nodes=31,
        early_stopping=False,
        random_state=42,
    )
    return make_pipeline(inputs, estimator)


def fit_model(model_name, train, target):
    loss = "poisson" if model_name == "poisson_boosting" else "squared_error"
    model = build_model(loss=loss)

    train_target = train[target]
    if model_name == "log_target_boosting":
        train_target = np.log1p(train_target)

    return model.fit(train[FEATURES], train_target)


def predict_model(model_name, model, evaluation):
    estimate = model.predict(evaluation[FEATURES])

    if model_name == "log_target_boosting":
        estimate = np.expm1(estimate)

    return np.maximum(estimate, 0)


def fit_and_predict(model_name, train, evaluation, target):
    model = fit_model(model_name, train, target)
    return predict_model(model_name, model, evaluation)


def historical_average(train, evaluation, target):
    keys = ["workingday", "hr"]
    averages = train.groupby(keys)[target].mean()
    groups = pd.MultiIndex.from_frame(evaluation[keys])
    return averages.reindex(groups).fillna(train[target].mean()).to_numpy()


def score(actual, estimate):
    return {
        "MAE": mean_absolute_error(actual, estimate),
        "RMSE": np.sqrt(mean_squared_error(actual, estimate)),
        "R2": r2_score(actual, estimate),
        # Positive values indicate overprediction; negative values indicate
        # systematic underprediction.
        "mean_error": np.mean(estimate - actual),
    }


def validation_results(data):
    rows = []
    for start, end in VALIDATION_WINDOWS:
        train = data[data["timestamp"] < start]
        validation = data[
            (data["timestamp"] >= start) & (data["timestamp"] < end)
        ]
        if train.empty or validation.empty:
            raise ValueError(f"Empty validation split: {start} to {end}")

        for target in TARGETS:
            baseline = historical_average(train, validation, target)
            rows.append(
                {
                    "validation_start": start,
                    "validation_end": end,
                    "rider": target,
                    "model": "historical_average",
                    **score(validation[target], baseline),
                }
            )
            for model_name in MODEL_NAMES:
                estimate = fit_and_predict(model_name, train, validation, target)
                rows.append(
                    {
                        "validation_start": start,
                        "validation_end": end,
                        "rider": target,
                        "model": model_name,
                        **score(validation[target], estimate),
                    }
                )

    return pd.DataFrame(rows)


def select_models(validation):
    summary = (
        validation.groupby(["rider", "model"], as_index=False)
        .agg(
            mean_MAE=("MAE", "mean"),
            mean_RMSE=("RMSE", "mean"),
            mean_R2=("R2", "mean"),
            mean_error=("mean_error", "mean"),
        )
        .sort_values(["rider", "mean_RMSE"])
    )
    candidate_summary = summary[summary["model"].isin(MODEL_NAMES)]
    best_rows = candidate_summary.loc[
        candidate_summary.groupby("rider")["mean_RMSE"].idxmin()
    ].copy()
    best_rows = best_rows.rename(columns={"model": "selected_model"})
    return summary, best_rows


def final_results(data, selections):
    train = data[data["timestamp"] < TEST_CUTOFF]
    test = data[data["timestamp"] >= TEST_CUTOFF]
    if train.empty or test.empty:
        raise ValueError("The data must contain rows on both sides of TEST_CUTOFF.")

    predictions = test[["timestamp", "workingday", "hr", *TARGETS, "cnt"]].copy()
    train_predictions = {}
    test_predictions = {}
    rows = []

    for target in TARGETS:
        estimates = {
            "historical_average": (
                historical_average(train, train, target),
                historical_average(train, test, target),
            )
        }
        for model_name in MODEL_NAMES:
            # Fit each final model once, then use the same fitted model for
            # both in-sample diagnostics and the held-out test prediction.
            model = fit_model(model_name, train, target)
            estimates[model_name] = (
                predict_model(model_name, model, train),
                predict_model(model_name, model, test),
            )

        baseline_rmse = {}
        for split, frame, position in [("train", train, 0), ("test", test, 1)]:
            baseline_rmse[split] = score(
                frame[target], estimates["historical_average"][position]
            )["RMSE"]
            for model_name, pair in estimates.items():
                estimate = pair[position]
                result = score(frame[target], estimate)
                rows.append(
                    {
                        "split": split,
                        "rider": target,
                        "model": model_name,
                        **result,
                        "RMSE_reduction_pct": 100
                        * (1 - result["RMSE"] / baseline_rmse[split]),
                    }
                )

        for model_name, (_, estimate) in estimates.items():
            predictions[f"{target}_{model_name}"] = estimate
            test_predictions[(target, model_name)] = estimate
        for model_name, (estimate, _) in estimates.items():
            train_predictions[(target, model_name)] = estimate

    selected = dict(zip(selections["rider"], selections["selected_model"]))
    all_models = ["historical_average", *MODEL_NAMES]
    total_baseline_rmse = {}
    for split, frame, stored_predictions in [
        ("train", train, train_predictions),
        ("test", test, test_predictions),
    ]:
        baseline_total = sum(
            stored_predictions[(target, "historical_average")]
            for target in TARGETS
        )
        total_baseline_rmse[split] = score(frame["cnt"], baseline_total)["RMSE"]
        for model_name in all_models:
            estimate = sum(
                stored_predictions[(target, model_name)] for target in TARGETS
            )
            result = score(frame["cnt"], estimate)
            rows.append(
                {
                    "split": split,
                    "rider": "total",
                    "model": model_name,
                    **result,
                    "RMSE_reduction_pct": 100
                    * (1 - result["RMSE"] / total_baseline_rmse[split]),
                }
            )
            if split == "test":
                predictions[f"total_{model_name}"] = estimate

    selected_test_total = sum(
        test_predictions[(target, selected[target])] for target in TARGETS
    )
    selected_train_total = sum(
        train_predictions[(target, selected[target])] for target in TARGETS
    )
    predictions["total_selected"] = selected_test_total

    for split, frame, estimate in [
        ("train", train, selected_train_total),
        ("test", test, selected_test_total),
    ]:
        result = score(frame["cnt"], estimate)
        rows.append(
            {
                "split": split,
                "rider": "total",
                "model": "selected_combination",
                **result,
                "RMSE_reduction_pct": 100
                * (1 - result["RMSE"] / total_baseline_rmse[split]),
            }
        )

    return pd.DataFrame(rows), predictions, train, test


def main():
    cleaned = OUTPUT / "clean_hourly.csv"
    if not cleaned.is_file():
        raise SystemExit("Run python3 clean.py before model_non_atemp.py.")

    data = pd.read_csv(cleaned, parse_dates=["timestamp"]).sort_values("timestamp")
    data["hr_day"] = data["hr"] + 24 * data["workingday"]

    validation = validation_results(data)
    validation_summary, selections = select_models(validation)
    final_metrics, predictions, train, test = final_results(data, selections)

    OUTPUT.mkdir(exist_ok=True)
    validation.to_csv(OUTPUT / "non_atemp_validation_metrics.csv", index=False)
    validation_summary.to_csv(
        OUTPUT / "non_atemp_validation_summary.csv", index=False
    )
    selections.to_csv(OUTPUT / "non_atemp_model_selection.csv", index=False)
    final_metrics.to_csv(OUTPUT / "non_atemp_train_test_metrics.csv", index=False)
    predictions.to_csv(OUTPUT / "non_atemp_test_predictions.csv", index=False)

    print(f"Training rows: {len(train):,} | Test rows: {len(test):,}")
    print("\nExpanding-window validation summary:")
    print(validation_summary.round(3).to_string(index=False))
    print("\nSelected model for each rider:")
    print(
        selections[["rider", "selected_model", "mean_RMSE", "mean_R2"]]
        .round(3)
        .to_string(index=False)
    )
    print("\nFinal train/test metrics:")
    print(final_metrics.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
