from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results"
TARGETS = ["casual", "registered"]


def repair_humidity(data):
    """Flag implausible zero humidity and replace it by time interpolation."""
    data = data.copy()
    invalid = data["hum"].eq(0)
    data["hum_imputed"] = invalid.astype("int8")
    data.loc[invalid, "hum"] = pd.NA

    # Interpolate only the environmental feature, never the rental targets.
    humidity = data.set_index("timestamp")["hum"].interpolate(
        method="time", limit_direction="both"
    )
    data["hum"] = data["timestamp"].map(humidity)
    print(f"Imputed zero-humidity records: {invalid.sum():,}")
    return data


def make_complete_hourly_grid(data, hours):
    """Create an audit grid while leaving unknown rentals and weather as NaN."""
    complete = data.set_index("timestamp").reindex(hours)
    complete.index.name = "timestamp"
    complete["is_observed"] = complete["cnt"].notna().astype("int8")

    # These fields are known from the timestamp or from another hour on the
    # same date. Weather and rental values are intentionally not fabricated.
    complete["hr"] = complete.index.hour
    date_columns = [
        "season", "yr", "mnth", "holiday", "weekday", "workingday"
    ]
    dates = data["timestamp"].dt.normalize()
    for column in date_columns:
        daily_values = data.groupby(dates)[column].first()
        missing = complete[column].isna()
        complete.loc[missing, column] = complete.index[missing].normalize().map(
            daily_values
        )

    return complete.reset_index()


def main():
    raw = pd.read_csv(ROOT / "hour.csv")
    print(f"Raw rows: {len(raw):,}")
    print(f"Missing cells: {raw.isna().sum().sum():,}")
    print(f"Duplicate rows: {raw.duplicated().sum():,}")

    data = raw.drop_duplicates().drop(columns="instant").copy()
    if data.isna().any().any():
        raise ValueError("The hourly data contains missing values.")
    if not data["hr"].isin(range(24)).all():
        raise ValueError("Hour values must be integers from 0 to 23.")

    data["timestamp"] = (
        pd.to_datetime(data.pop("dteday"), format="%Y-%m-%d")
        + pd.to_timedelta(data["hr"], unit="h")
    )
    data = data.sort_values("timestamp").reset_index(drop=True)

    if not data["timestamp"].is_unique:
        raise ValueError("Multiple records share an hourly timestamp.")
    if (data[TARGETS + ["cnt"]] < 0).any().any():
        raise ValueError("Rental counts must be nonnegative.")
    if not (data[TARGETS].sum(axis=1) == data["cnt"]).all():
        raise ValueError("Casual and registered counts must add up to cnt.")

    data = repair_humidity(data)

    hours = pd.date_range(data["timestamp"].min(), data["timestamp"].max(), freq="h")
    missing_hours = hours.difference(data["timestamp"])
    complete = make_complete_hourly_grid(data, hours)
    print(f"Missing hourly records: {len(missing_hours):,}")
    print(data[TARGETS].describe().round(2))
    print(data.groupby("weathersit")[TARGETS].agg(["size", "mean"]).round(2))

    OUTPUT.mkdir(exist_ok=True)
    data.to_csv(OUTPUT / "clean_hourly.csv", index=False)
    complete.to_csv(OUTPUT / "complete_hourly.csv", index=False)
    pd.Series(missing_hours, name="timestamp").to_csv(
        OUTPUT / "missing_hours.csv", index=False
    )

    profile = data.groupby(["workingday", "hr"])[TARGETS].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for target, ax in zip(TARGETS, axes):
        for day, label in [(0, "Weekend or holiday"), (1, "Working day")]:
            hourly = profile.loc[day, target]
            ax.plot(hourly.index, hourly, label=label)
        ax.set(title=target.capitalize(), xlabel="Hour", ylabel="Mean rentals")
        ax.legend()
    fig.savefig(OUTPUT / "hourly_profiles.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
