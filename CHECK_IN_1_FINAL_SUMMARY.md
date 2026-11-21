# Check-in 1 Final Summary

## Project objective

The goal is to forecast hourly bike rentals for **casual** and **registered** riders separately and determine whether the models improve on a historical-average baseline.

## 1. Data cleaning

The original Capital Bikeshare hourly dataset covers January 2011 through December 2012.

| Cleaning result | Final status |
|---|---:|
| Observed hourly records | 17,379 |
| Duplicate rows or timestamps | 0 |
| Missing cells in observed data | 0 |
| Invalid or negative rental counts | 0 |
| Missing timestamps in the complete hourly sequence | 165 (0.94%) |
| Suspicious zero-humidity records repaired | 22 |

The cleaning process removed the index-only `instant` field, created a chronological `timestamp`, verified that `casual + registered = cnt`, repaired the humidity anomaly, and created a complete hourly audit grid without inventing rental values for missing hours.

**Purpose:** produce a consistent dataset suitable for chronological exploration and forecasting while preserving genuine demand peaks and unusual events.

## 2. Training-period exploration

All exploratory figures use only the 14,491 observations before September 1, 2012, so the final test period does not influence feature decisions.

Key findings:

- Registered riders show strong working-day commuting peaks around 8:00 a.m. and 5:00–6:00 p.m.
- Casual demand is highest during weekend and holiday afternoons.
- From January–August 2011 to the same period in 2012, casual rentals increased by **51.8%** and registered rentals by **73.6%**.
- Demand decreases as weather becomes worse.
- Temperature is more strongly associated with casual demand (correlation **0.473**) than registered demand (**0.371**).

**Purpose:** explain why casual and registered riders should be modeled separately and why the models include hour, working-day status, month/season, year, weather, and temperature.

## 3. Model development

The data were divided chronologically:

- **Training:** 14,491 rows, January 2011–August 2012.
- **Testing:** 2,888 rows, September–December 2012.

The initial experiment compared the historical-average baseline, Ridge regression, and squared-error Histogram Gradient Boosting. In the September–December 2012 test period, Boosting achieved the lowest RMSE for both casual and registered riders.

The second experiment removed `atemp` because it was almost perfectly correlated with `temp` (correlation **0.9877**). It then compared squared-error, Poisson, and log-target Gradient Boosting using expanding-window validation.

## 4. Final model results

Following the expanding-window validation rule implemented in the code, the final approach uses **log-target Histogram Gradient Boosting for casual riders** and **squared-error Histogram Gradient Boosting for registered riders**, both without `atemp`.

| Test target | Historical R² | Final R² | Final MAE | Final RMSE | RMSE improvement |
|---|---:|---:|---:|---:|---:|
| Casual: log-target Boosting | 0.543 | **0.859** | 11.089 | 20.987 | 44.50% |
| Registered: squared-error Boosting | 0.606 | **0.876** | 45.260 | 65.532 | 43.87% |
| Validation-selected total | 0.595 | **0.880** | 51.976 | 75.256 | 45.67% |

We selected one model per rider group based on the lowest mean RMSE across the three expanding validation windows.

## Final conclusion

For Check-in 1, we prepared the hourly rental data and identified distinct demand patterns for casual and registered riders. Combining the two selected models reduced total hourly rental RMSE by 45.67% compared with the historical-average baseline. In the next phase, we will use these forecasts to compare simulated bike deployment plans based on the historical baseline and the selected models.