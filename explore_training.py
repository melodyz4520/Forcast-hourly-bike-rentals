from pathlib import Path

import matplotlib

# This script saves figures to files and does not need an interactive window.
# Agg also makes rendering reliable in terminals, IDE tasks, and CI systems.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results" / "clean_hourly.csv"
OUTPUT = ROOT / "results" / "training_eda"
TRAIN_CUTOFF = pd.Timestamp("2012-09-01")
RIDERS = ["casual", "registered"]

COLORS = {"casual": "#E07A5F", "registered": "#3D5A80"}
WEATHER_LABELS = {
    1: "Clear / partly cloudy",
    2: "Mist / cloudy",
    3: "Light rain / snow",
    4: "Heavy rain / snow",
}


def load_training_data():
    if not INPUT.is_file():
        raise SystemExit("Run python3 clean.py before explore_training.py.")

    data = pd.read_csv(INPUT, parse_dates=["timestamp"]).sort_values("timestamp")
    required = {
        "timestamp",
        "hr",
        "workingday",
        "weathersit",
        "temp",
        "casual",
        "registered",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    training = data[data["timestamp"] < TRAIN_CUTOFF].copy()
    if training.empty:
        raise ValueError("The training period is empty.")

    return training


def format_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def plot_hourly_average(training):
    hourly = training.groupby("hr")[RIDERS].mean()

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for rider in RIDERS:
        ax.plot(
            hourly.index,
            hourly[rider],
            marker="o",
            linewidth=2.2,
            markersize=4,
            color=COLORS[rider],
            label=rider.capitalize(),
        )

    format_axis(
        ax,
        "Average Hourly Rentals in the Training Period",
        "Hour of day",
        "Mean rentals",
    )
    ax.set_xticks(range(24))
    ax.legend(title="Rider group")
    fig.savefig(OUTPUT / "01_hourly_average.png", dpi=180)
    plt.close(fig)

    hourly.reset_index().to_csv(OUTPUT / "01_hourly_average.csv", index=False)
    return hourly


def plot_workingday_profiles(training):
    profiles = training.groupby(["workingday", "hr"])[RIDERS].mean()
    day_labels = {0: "Weekend or holiday", 1: "Working day"}
    line_styles = {0: "--", 1: "-"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, layout="constrained")
    for rider, ax in zip(RIDERS, axes):
        for workingday in [0, 1]:
            values = profiles.loc[workingday, rider]
            ax.plot(
                values.index,
                values,
                marker="o",
                markersize=3.5,
                linewidth=2.1,
                linestyle=line_styles[workingday],
                label=day_labels[workingday],
            )

        format_axis(
            ax,
            f"{rider.capitalize()} Rentals by Day Type",
            "Hour of day",
            "Mean rentals",
        )
        ax.set_xticks(range(0, 24, 2))
        ax.legend()

    fig.suptitle(
        "Working-Day and Nonworking-Day Hourly Profiles",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(OUTPUT / "02_workingday_hourly_profiles.png", dpi=180)
    plt.close(fig)

    profiles.reset_index().to_csv(
        OUTPUT / "02_workingday_hourly_profiles.csv", index=False
    )
    return profiles


def plot_monthly_rentals(training):
    monthly = (
        training.assign(
            year=training["timestamp"].dt.year,
            month=training["timestamp"].dt.month,
        )
        .groupby(["year", "month"])[RIDERS]
        .sum()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, layout="constrained")
    for rider, ax in zip(RIDERS, axes):
        for year, values in monthly.groupby("year"):
            ax.plot(
                values["month"],
                values[rider],
                marker="o",
                linewidth=2.2,
                label=str(year),
            )

        format_axis(
            ax,
            f"Monthly {rider.capitalize()} Rentals",
            "Month",
            "Total rentals",
        )
        ax.set_xticks(range(1, 13))
        ax.ticklabel_format(axis="y", style="plain")
        ax.legend(title="Year")

    fig.suptitle(
        "Monthly Seasonality and Year-over-Year Growth",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(OUTPUT / "03_monthly_rentals.png", dpi=180)
    plt.close(fig)

    monthly.to_csv(OUTPUT / "03_monthly_rentals.csv", index=False)
    return monthly


def plot_weather_and_temperature(training):
    weather_summary = training.groupby("weathersit").agg(
        observations=("casual", "size"),
        casual_mean=("casual", "mean"),
        registered_mean=("registered", "mean"),
    )
    weather_means = weather_summary.rename(
        columns={"casual_mean": "casual", "registered_mean": "registered"}
    )[RIDERS]

    # Equal-width bins keep the normalized temperature scale interpretable.
    temperature = training.copy()
    temperature["temp_bin"] = pd.cut(
        temperature["temp"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
    )
    temperature_profile = (
        temperature.groupby("temp_bin", observed=False)
        .agg(
            temp_mean=("temp", "mean"),
            observations=("temp", "size"),
            casual=("casual", "mean"),
            registered=("registered", "mean"),
        )
        .dropna(subset=["temp_mean"])
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    weather_positions = np.arange(len(weather_means))
    weather_names = [
        WEATHER_LABELS.get(int(value), str(value)) for value in weather_means.index
    ]

    for column, rider in enumerate(RIDERS):
        weather_ax = axes[0, column]
        weather_ax.bar(
            weather_positions,
            weather_means[rider],
            color=COLORS[rider],
            alpha=0.9,
        )
        format_axis(
            weather_ax,
            f"Weather and {rider.capitalize()} Rentals",
            "Weather condition",
            "Mean rentals",
        )
        weather_ax.set_xticks(weather_positions, weather_names, rotation=20, ha="right")

        temp_ax = axes[1, column]
        temp_ax.plot(
            temperature_profile["temp_mean"],
            temperature_profile[rider],
            marker="o",
            linewidth=2.2,
            color=COLORS[rider],
        )
        format_axis(
            temp_ax,
            f"Temperature and {rider.capitalize()} Rentals",
            "Mean normalized temperature in bin",
            "Mean rentals",
        )

    fig.suptitle(
        "Weather and Temperature Relationships by Rider Group",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(OUTPUT / "04_weather_temperature_relationships.png", dpi=180)
    plt.close(fig)

    weather_summary.reset_index().to_csv(
        OUTPUT / "04_weather_summary.csv", index=False
    )
    temperature_profile.reset_index().to_csv(
        OUTPUT / "04_temperature_summary.csv", index=False
    )
    return weather_means, temperature_profile


def print_findings(training, hourly, profiles, monthly, weather, temperature):
    print(
        f"Training observations: {len(training):,} | "
        f"{training['timestamp'].min()} to {training['timestamp'].max()}"
    )
    print("\nPeak average hour:")
    for rider in RIDERS:
        print(
            f"  {rider}: {int(hourly[rider].idxmax()):02d}:00 "
            f"(mean={hourly[rider].max():.2f})"
        )

    print("\nPeak hour by rider and day type:")
    for rider in RIDERS:
        for workingday, label in [(0, "weekend/holiday"), (1, "working day")]:
            values = profiles.loc[workingday, rider]
            print(
                f"  {rider}, {label}: {int(values.idxmax()):02d}:00 "
                f"(mean={values.max():.2f})"
            )

    comparable = monthly[monthly["month"] <= 8]
    january_august = comparable.groupby("year")[RIDERS].sum()
    if {2011, 2012}.issubset(january_august.index):
        growth = 100 * (january_august.loc[2012] / january_august.loc[2011] - 1)
        print("\nJanuary-August growth from 2011 to 2012:")
        for rider in RIDERS:
            print(f"  {rider}: {growth[rider]:.1f}%")

    print("\nMean rentals by weather category:")
    print(weather.round(2).to_string())
    print("\nCorrelation with normalized temperature:")
    for rider in RIDERS:
        correlation = training["temp"].corr(training[rider])
        print(f"  {rider}: {correlation:.3f}")

    print(f"\nCharts and summaries saved to: {OUTPUT}")


def write_summary(training, hourly, profiles, monthly, weather):
    comparable = monthly[monthly["month"] <= 8]
    january_august = comparable.groupby("year")[RIDERS].sum()
    growth = 100 * (january_august.loc[2012] / january_august.loc[2011] - 1)

    casual_nonwork = profiles.loc[0, "casual"]
    casual_work = profiles.loc[1, "casual"]
    registered_nonwork = profiles.loc[0, "registered"]
    registered_work = profiles.loc[1, "registered"]
    weather_counts = training["weathersit"].value_counts().sort_index()

    summary = f"""# Training-Period Exploratory Analysis

## Scope

All figures use only the {len(training):,} training observations from {training['timestamp'].min():%Y-%m-%d %H:%M} through {training['timestamp'].max():%Y-%m-%d %H:%M}. The final test period is excluded so that exploratory decisions are not influenced by future outcomes.

## Main findings

### 1. Average hourly demand

Average registered demand is much higher than casual demand. Both series reach their overall maximum at {int(hourly['registered'].idxmax()):02d}:00 in the training data, but the combined profile hides important day-type differences. Registered demand has a sharp morning peak and a larger evening peak, while casual demand increases more gradually through the day.

This result supports including `hr` and modeling casual and registered riders separately.

### 2. Working-day and nonworking-day profiles

- On weekends and holidays, casual demand peaks at {int(casual_nonwork.idxmax()):02d}:00 with a mean of {casual_nonwork.max():.2f} rentals.
- On working days, casual demand peaks at {int(casual_work.idxmax()):02d}:00 with a much lower mean of {casual_work.max():.2f} rentals.
- Registered weekend/holiday demand peaks at {int(registered_nonwork.idxmax()):02d}:00 with a mean of {registered_nonwork.max():.2f} rentals.
- Registered working-day demand has strong commuting peaks and reaches its maximum at {int(registered_work.idxmax()):02d}:00 with a mean of {registered_work.max():.2f} rentals.

The different shapes justify the `workingday × hr` interaction represented by `hr_day`. They also show why one model for total rentals would hide meaningful rider-specific behavior.

### 3. Monthly seasonality and growth

Demand is lowest during the winter and rises substantially during warmer months. Comparing January–August in both years avoids treating the incomplete September–December 2012 training period as missing data. From January–August 2011 to the same months in 2012:

- Casual rentals increased by {growth['casual']:.1f}%.
- Registered rentals increased by {growth['registered']:.1f}%.

The strong seasonal pattern supports including `mnth` or `season`, while the year-over-year growth supports including `yr` or another time-trend feature.

### 4. Weather and temperature

Mean demand declines as weather becomes less favorable. Casual mean demand falls from {weather.loc[1, 'casual']:.2f} in clear or partly cloudy conditions to {weather.loc[3, 'casual']:.2f} in light rain or snow. Registered mean demand falls from {weather.loc[1, 'registered']:.2f} to {weather.loc[3, 'registered']:.2f} across the same categories.

The correlation between normalized temperature and demand is {training['temp'].corr(training['casual']):.3f} for casual riders and {training['temp'].corr(training['registered']):.3f} for registered riders. Both groups respond positively to warmer temperatures, but the relationship is stronger for casual users.

Weather category 4 contains only {int(weather_counts.get(4, 0))} training observations, so its mean should not be interpreted as a stable effect. The results support retaining `weathersit` and `temp`, while allowing the model to learn different responses for casual and registered riders.

## Modeling implications

The training-period exploration supports the following decisions:

1. Train casual and registered models separately because their hourly and working-day patterns differ substantially.
2. Include hour and working-day information, preferably with an interaction such as `hr_day`.
3. Include month/season and year to represent seasonality and growth.
4. Include weather condition and temperature because demand falls in poor weather and rises with temperature.
5. Use a nonlinear model because the hourly peaks, seasonal curves, and temperature relationships are not well represented by a single straight-line effect.
"""
    (OUTPUT / "training_eda_summary.md").write_text(summary, encoding="utf-8")


def main():
    training = load_training_data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    hourly = plot_hourly_average(training)
    profiles = plot_workingday_profiles(training)
    monthly = plot_monthly_rentals(training)
    weather, temperature = plot_weather_and_temperature(training)
    write_summary(training, hourly, profiles, monthly, weather)
    print_findings(training, hourly, profiles, monthly, weather, temperature)


if __name__ == "__main__":
    main()
