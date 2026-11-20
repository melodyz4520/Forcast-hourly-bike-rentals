# Training-Period Exploratory Analysis

## Scope

All figures use only the 14,491 training observations from 2011-01-01 00:00 through 2012-08-31 23:00. The final test period is excluded so that exploratory decisions are not influenced by future outcomes.

## Main findings

### 1. Average hourly demand

Average registered demand is much higher than casual demand. Both series reach their overall maximum at 17:00 in the training data, but the combined profile hides important day-type differences. Registered demand has a sharp morning peak and a larger evening peak, while casual demand increases more gradually through the day.

This result supports including `hr` and modeling casual and registered riders separately.

### 2. Working-day and nonworking-day profiles

- On weekends and holidays, casual demand peaks at 14:00 with a mean of 133.10 rentals.
- On working days, casual demand peaks at 17:00 with a much lower mean of 55.88 rentals.
- Registered weekend/holiday demand peaks at 12:00 with a mean of 220.78 rentals.
- Registered working-day demand has strong commuting peaks and reaches its maximum at 17:00 with a mean of 440.83 rentals.

The different shapes justify the `workingday × hr` interaction represented by `hr_day`. They also show why one model for total rentals would hide meaningful rider-specific behavior.

### 3. Monthly seasonality and growth

Demand is lowest during the winter and rises substantially during warmer months. Comparing January–August in both years avoids treating the incomplete September–December 2012 training period as missing data. From January–August 2011 to the same months in 2012:

- Casual rentals increased by 51.8%.
- Registered rentals increased by 73.6%.

The strong seasonal pattern supports including `mnth` or `season`, while the year-over-year growth supports including `yr` or another time-trend feature.

### 4. Weather and temperature

Mean demand declines as weather becomes less favorable. Casual mean demand falls from 39.71 in clear or partly cloudy conditions to 15.32 in light rain or snow. Registered mean demand falls from 154.69 to 89.94 across the same categories.

The correlation between normalized temperature and demand is 0.473 for casual riders and 0.371 for registered riders. Both groups respond positively to warmer temperatures, but the relationship is stronger for casual users.

Weather category 4 contains only 3 training observations, so its mean should not be interpreted as a stable effect. The results support retaining `weathersit` and `temp`, while allowing the model to learn different responses for casual and registered riders.

## Modeling implications

The training-period exploration supports the following decisions:

1. Train casual and registered models separately because their hourly and working-day patterns differ substantially.
2. Include hour and working-day information, preferably with an interaction such as `hr_day`.
3. Include month/season and year to represent seasonality and growth.
4. Include weather condition and temperature because demand falls in poor weather and rises with temperature.
5. Use a nonlinear model because the hourly peaks, seasonal curves, and temperature relationships are not well represented by a single straight-line effect.
