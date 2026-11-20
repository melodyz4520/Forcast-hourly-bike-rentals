# Hourly Bike Rental Dataset Cleaning Result

## 1. Cleaning objective

This project cleans the hourly Capital Bikeshare dataset so that it can be used for exploratory analysis and forecasting. The original file, `hour.csv`, contains hourly rental demand and calendar, weather, and environmental variables for 2011–2012.

## 2. How `clean.py` cleans and validates the data

The cleaning program performs the following steps:

1. Loads `hour.csv` and reports the original number of rows, missing cells, and exact duplicate rows.
2. Removes exact duplicate rows and drops `instant`, because it is only a record identifier and does not provide useful predictive information.
3. Checks that the remaining data contain no missing values.
4. Checks that `hr` is within the valid range from 0 to 23.
5. Combines `dteday` and `hr` into one hourly `timestamp`, then sorts all observations chronologically.
6. Confirms that every observed timestamp is unique.
7. Confirms that `casual`, `registered`, and `cnt` are nonnegative.
8. Verifies the accounting identity `casual + registered = cnt` for every row.
9. Creates a complete hourly date range and identifies hours that are absent from the source data. These gaps are saved separately rather than being silently filled or deleted.
10. Flags the 22 implausible zero-humidity records in `hum_imputed`, replaces only their humidity values using time interpolation, and leaves all rental targets unchanged.
11. Creates `results/complete_hourly.csv`, a complete 17,544-row audit grid. Missing-hour targets and weather values remain `NaN`, and `is_observed` distinguishes real observations from inserted grid rows.
12. Saves the observed cleaned data as `results/clean_hourly.csv`, the list of gaps as `results/missing_hours.csv`, and an hourly demand profile as `results/hourly_profiles.png`.

This is a conservative cleaning strategy: it corrects the structure and validates important rules without automatically changing potentially meaningful demand values.

## 3. Final cleaned-data status

| Check | Result |
|---|---:|
| Original rows | 17,379 |
| Cleaned rows | 17,379 |
| Original columns | 17 |
| Cleaned columns | 17, including the `hum_imputed` audit flag |
| Missing cells | 0 |
| Exact duplicate rows | 0 |
| Duplicate timestamps | 0 |
| Negative rental counts | 0 |
| Rows where `casual + registered != cnt` | 0 |
| Observed date range | 2011-01-01 00:00 to 2012-12-31 23:00 |
| Calendar days represented | 731 |
| Missing hourly timestamps | 165 of 17,544 possible hours (0.94%) |
| Complete 24-hour days | 655 |
| Partial days | 76 |

No rows were removed because the raw data had no exact duplicates or missing cells. The main structural changes were removing `instant`, replacing the separate date representation with a sortable hourly `timestamp`, and adding the humidity-imputation flag. The observed cleaned dataset therefore has 17,379 rows and 17 variables. A separate complete-grid file has 17,544 rows and does not fabricate unknown rental targets.

Additional consistency checks found that `yr`, `mnth`, `hr`, `weekday`, and `workingday` all agree with the constructed timestamp and holiday/weekend logic. All categorical fields also use their documented ranges: four seasons, two years, 12 months, 24 hours, seven weekdays, and four weather categories.

## 4. Initial exploration of the cleaned dataset

### Rental-demand distributions

| Rider group | Mean | Standard deviation | Median | Maximum |
|---|---:|---:|---:|---:|
| Casual | 35.68 | 49.31 | 17 | 367 |
| Registered | 153.79 | 151.36 | 115 | 886 |

Registered rentals are much more common than casual rentals. Both target variables are right-skewed, especially casual demand. There are 1,581 observations with zero casual rentals and 24 with zero registered rentals, but total demand is never zero in the observed rows.

An IQR rule labels 6.86% of casual observations, 3.91% of registered observations, and 2.91% of total-demand observations as high values. These should not automatically be deleted: demand peaks are plausible during commuting periods, favorable weather, and high-demand seasons, and they are important for forecasting.

### Time-of-day patterns

The hourly profile shows clear and credible behavioral differences:

- On working days, registered demand has strong morning and evening commuting peaks around 08:00 and 17:00–18:00.
- On weekends and holidays, registered demand is smoother and concentrated around the middle of the day.
- Casual demand is highest during the afternoon on weekends and holidays.

These patterns support keeping `hr` and `workingday`, and they suggest that their interaction is useful for modeling.

### Weather patterns

Average demand declines as the weather becomes worse. Clear or partly cloudy conditions have the highest mean demand, while light rain or snow has lower demand. This is consistent with expected bike-rental behavior.

The most severe weather category (`weathersit = 4`) appears only three times. It is valid according to the data dictionary, but its extreme rarity means that a model cannot estimate its effect reliably. It may be appropriate to combine categories 3 and 4 for some models or to report category 4 results cautiously.

## 5. Issues that still require attention

### 5.1 Missing hourly records

Although the table has no blank cells, 165 timestamps are absent from the complete two-year hourly sequence. Most gaps occur during low-demand early-morning hours: hour 03:00 and hour 04:00 each account for 34 missing timestamps. There are 115 gaps in 2011 and 50 in 2012.

Some gaps may reflect unusual real events rather than ordinary data errors. For example, 2012-10-29 contains only one observed hour and 2012-10-30 contains 11; this period coincides with the Hurricane Sandy event discussed in the dataset documentation. Such periods should be preserved as events or flagged, not automatically treated as normal missing-at-random observations.

The cleaning pipeline now also writes a complete hourly grid and uses `is_observed = 0` for these 165 gaps. Calendar fields that can be recovered from the timestamp or another observation on the same date are populated, while unknown rental targets and weather fields remain `NaN`. The existing forecasting model continues to train on `clean_hourly.csv`, so inserted grid rows cannot become fabricated training labels. Missing rental counts should not be set to zero unless the data source confirms that an absent row means no rentals.

### 5.2 Suspicious zero humidity values

There are 22 records with `hum = 0`, all on 2011-03-10. Most of these rows are classified as rainy or snowy weather (`weathersit = 3`), so zero relative humidity is not physically consistent with the weather description. This block is likely a sensor, extraction, or encoding problem.

The revised cleaning pipeline flags these observations with `hum_imputed = 1`, treats their humidity as missing, and fills only humidity by time interpolation between the surrounding valid observations. This produces plausible values from 0.9328 to 0.9972 while keeping rental counts and every other field unchanged. Source verification would still be preferable if the original weather feed is available.

### 5.3 Skewness and high-demand observations

The target variables are right-skewed (`casual` skewness 2.499, `registered` 1.558, and `cnt` 1.277). This is a modeling issue rather than evidence of dirty data. Recommended responses include a log1p target transformation, count-aware models, robust loss functions, and reporting MAE together with RMSE. High-demand rows should not be removed solely because an IQR rule calls them outliers.

### 5.4 Strongly related predictors

`temp` and `atemp` have a correlation of 0.9877. Keeping both is acceptable for tree-based models, but they may create multicollinearity in an unregularized linear model. A linear analysis should use regularization or retain only one of these variables. This is feature selection, not data cleaning.

## 6. Evaluation and recommendation

The current `clean.py` is a good first-stage cleaning and validation pipeline. It produces a consistent, reproducible dataset with no cell-level missingness, duplicates, invalid counts, or broken target totals. The cleaned file is ready for basic exploratory analysis and for models that treat observations as irregularly spaced records.

The revised pipeline now handles the time grid and zero-humidity anomaly transparently. More exploration is still recommended before treating every inserted hour as a usable supervised-learning observation:

1. Investigate the source meaning of the 165 absent timestamps before assigning rental targets to them.
2. Verify the 22 repaired humidity values against the original weather source if it is available.
3. Add validation rules for every documented field range, not only `hr` and rental counts.
4. Preserve genuine event periods and demand peaks; add event or anomaly flags instead of deleting them.
5. Explore seasonality by hour, weekday, month, year, weather, and rider type, using a chronological train/test split to avoid future-data leakage.
6. Examine target transformations and rare-category handling during model development.

Overall assessment: the dataset is structurally clean and suitable for continued analysis. The pipeline now exposes a complete hourly audit grid and repairs the known humidity anomaly without inventing demand. Further work should focus on understanding the missing-hour mechanism, validating the imputation, and improving feature/model choices rather than broad row deletion or generic outlier removal.
