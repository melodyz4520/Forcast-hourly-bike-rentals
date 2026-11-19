from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results"
TARGETS = ["casual", "registered"]


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

    hours = pd.date_range(data["timestamp"].min(), data["timestamp"].max(), freq="h")
    missing_hours = hours.difference(data["timestamp"])
    print(f"Missing hourly records: {len(missing_hours):,}")
    print(data[TARGETS].describe().round(2))
    print(data.groupby("weathersit")[TARGETS].agg(["size", "mean"]).round(2))

    OUTPUT.mkdir(exist_ok=True)
    data.to_csv(OUTPUT / "clean_hourly.csv", index=False)
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