# Bike Rental Forecasting: Final Model Summary

## 1. Final conclusion

No additional model training is required for the current project. The two modeling scripts have already completed the essential workflow:

1. Trained separate forecasting models for casual and registered riders.
2. Established a historical-average baseline.
3. Compared linear and nonlinear boosting models.
4. Used chronological training, validation, and testing periods to avoid future-data leakage.
5. Identified and removed the highly redundant `atemp` variable.
6. Compared squared-error, Poisson, and log-target boosting approaches.
7. Evaluated casual, registered, and total rentals using MAE, RMSE, and R².

Following the selection rule implemented in `model_non_atemp.py`, the final validation-selected approach uses **log-target Histogram Gradient Boosting for casual riders** and **squared-error Histogram Gradient Boosting for registered riders**. Their predictions are added to obtain total predicted demand.

The validation-selected models produced the following test results:

| Rider group | MAE | RMSE | R² | RMSE reduction from historical baseline |
|---|---:|---:|---:|---:|
| Casual: log-target Boosting | 11.089 | 20.987 | 0.859 | 44.50% |
| Registered: squared-error Boosting | 45.260 | 65.532 | 0.876 | 43.87% |
| Validation-selected total | 51.976 | 75.256 | 0.880 | 45.67% |

These models are the formal selections because they achieved the lowest mean validation RMSE before the final test period. Poisson Boosting produced the strongest post-selection test scores, but it was not selected by the validation rule and therefore remains a candidate for confirmation on new future data.

## 2. Data split and prediction targets

Both modeling scripts use the same final chronological train/test split:

| Dataset | Period | Observations | Percentage |
|---|---|---:|---:|
| Training set | January 1, 2011–August 31, 2012 | 14,491 | 83.38% |
| Testing set | September 1, 2012–December 31, 2012 | 2,888 | 16.62% |

The models predict two targets separately:

- `casual`: hourly rentals by nonregistered riders.
- `registered`: hourly rentals by registered riders.

Total predicted demand is calculated as:

```text
predicted total = predicted casual + predicted registered
```

Separate modeling is useful because the two rider groups have different behavior. Registered demand is strongly influenced by weekday commuting peaks, while casual demand is more concentrated during weekends, holidays, and daytime hours.

## 3. First modeling script: `model.py`

### 3.1 Variables and feature engineering

The first experiment uses `yr`, `temp`, `atemp`, `hum`, and `windspeed` as numeric variables. Its categorical variables are `hr`, `weekday`, `workingday`, `holiday`, `season`, `mnth`, `weathersit`, and `hr_day`.

The interaction feature is defined as:

```python
hr_day = hr + 24 * workingday
```

This separates hours 0–23 on weekends or holidays from hours 24–47 on working days, allowing the models to learn different hourly profiles for the two day types.

### 3.2 Historical-average baseline

The baseline predicts demand using the mean rental count for each `workingday × hr` group in the training data. For example, a future working day at 8:00 a.m. is predicted using the historical average for working days at 8:00 a.m.

This baseline is simple and interpretable, but it does not use weather, season, month, or year information. It provides a reference for deciding whether more complex models offer a meaningful improvement.

### 3.3 Ridge regression

The Ridge pipeline performs the following steps:

1. Applies one-hot encoding to categorical variables.
2. Standardizes numeric variables.
3. Fits a linear regression with L2 regularization.

Ridge provides a useful linear benchmark and can tolerate correlated predictors. However, it cannot fully represent the nonlinear relationships among hour, weather, season, and rider behavior.

### 3.4 Squared-error Histogram Gradient Boosting

The Gradient Boosting model builds a sequence of decision trees, with each tree correcting errors made by the earlier trees. It can learn nonlinear effects and interactions among time, working-day status, weather, and seasonal variables. Any negative predictions are clipped to zero so that rental predictions remain nonnegative.

### 3.5 Test results from `model.py`

#### Casual riders

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Historical average | 21.916 | 37.814 | 0.543 |
| Ridge | 18.586 | 31.238 | 0.688 |
| Gradient Boosting | **11.495** | **20.787** | **0.862** |

#### Registered riders

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Historical average | 81.402 | 116.752 | 0.606 |
| Ridge | 65.729 | 88.951 | 0.771 |
| Gradient Boosting | **46.288** | **66.156** | **0.873** |

#### Total rentals

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Historical average | 94.780 | 138.527 | 0.595 |
| Ridge | 77.598 | 104.193 | 0.771 |
| Gradient Boosting | **52.545** | **75.539** | **0.880** |

Ridge improves on the historical baseline, but Gradient Boosting performs substantially better. This result demonstrates that important nonlinear relationships are present in the data and that a purely linear model is insufficient.

## 4. Why a second modeling script was created

The cleaning and exploratory analysis found a correlation of 0.9877 between `temp` and `atemp`. These variables therefore contain almost the same information. Including both increases redundancy without necessarily improving predictive performance.

Removing `atemp` slightly improved the ordinary squared-error Gradient Boosting model:

| Rider group | Test R² with `atemp` | Test R² without `atemp` |
|---|---:|---:|
| Casual | 0.8618 | 0.8666 |
| Registered | 0.8734 | 0.8758 |

The second script, `model_non_atemp.py`, therefore excludes `atemp`, retains `temp`, and compares different loss and target-transformation approaches.

## 5. Second modeling script: `model_non_atemp.py`

### 5.1 Expanding-window validation

The second script uses three validation windows before the final test period:

1. December 1, 2011–February 29, 2012.
2. March 1, 2012–May 31, 2012.
3. June 1, 2012–August 31, 2012.

For each window, only observations occurring before the validation window are used for training. The fitted model then predicts the following three months. This expanding-window design more closely represents a real forecasting task than random cross-validation.

### 5.2 Squared-error Boosting

This is similar to the Gradient Boosting model in `model.py`, but it excludes `atemp`. It minimizes squared prediction error and therefore places greater emphasis on large errors during high-demand periods.

### 5.3 Poisson Boosting

The Poisson model uses:

```python
HistGradientBoostingRegressor(loss="poisson")
```

Poisson loss is appropriate because casual and registered rentals are nonnegative counts. Demand variability also tends to increase as expected demand increases. The Poisson formulation is designed for this type of mean–variance relationship and naturally produces positive predictions.

### 5.4 Log-target Boosting

The log-target model first transforms the target using:

```python
y_log = np.log1p(y)
```

Predictions are converted back to the original scale using:

```python
y_prediction = np.expm1(log_prediction)
```

This transformation reduces right skew and the influence of extreme demand values. However, converting predictions back to the original scale can introduce downward bias, so its performance must be compared on validation data.

## 6. Expanding-window validation results

| Rider group | Model | Mean validation MAE | Mean validation RMSE | Mean validation R² |
|---|---|---:|---:|---:|
| Casual | Squared-error | 16.174 | 26.126 | 0.670 |
| Casual | Poisson | 13.614 | 24.258 | **0.701** |
| Casual | Log-target | **13.388** | **23.813** | 0.699 |
| Registered | Squared-error | **46.813** | **68.542** | **0.749** |
| Registered | Poisson | 47.148 | 71.116 | 0.726 |
| Registered | Log-target | 50.040 | 75.491 | 0.709 |

Average validation RMSE is the selection criterion implemented in the code. It selects log-target Boosting for casual riders and squared-error Boosting for registered riders. The rankings are not perfectly stable across time: squared-error wins the first casual window, Poisson wins the second, and log-target wins the most recent casual window; for registered riders, squared-error wins the first two windows and Poisson wins the most recent window.

## 7. Final train/test results from `model_non_atemp.py`

### 7.1 Casual riders

| Model | Train RMSE | Train R² | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|
| Squared-error | 11.696 | 0.940 | 11.320 | 20.426 | 0.867 |
| Poisson | 11.134 | 0.946 | 10.876 | 19.781 | 0.875 |
| **Log-target (validation selected)** | **13.914** | **0.915** | **11.089** | **20.987** | **0.859** |

### 7.2 Registered riders

| Model | Train RMSE | Train R² | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|
| **Squared-error (validation selected)** | **26.674** | **0.965** | **45.260** | **65.532** | **0.876** |
| Poisson | 25.156 | 0.968 | 42.177 | 63.114 | 0.885 |
| Log-target | 29.258 | 0.957 | 46.186 | 68.746 | 0.863 |

### 7.3 Total rentals

| Model | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|
| Historical average | 94.780 | 138.527 | 0.595 |
| Squared-error | 51.326 | 74.211 | 0.884 |
| Poisson | **48.112** | **71.676** | **0.892** |
| Log-target | 52.735 | 78.082 | 0.871 |
| **Validation-selected combination** | **51.976** | **75.256** | **0.880** |

Poisson Boosting has the best observed scores on the final testing period, but the code-selected combination is casual log-target plus registered squared-error because selection is based on pre-test validation RMSE. The selected combination retains an R² of 0.880 on later observations without using the final test period to choose the model.

## 8. Direct comparison of the two scripts

| Category | `model.py` | `model_non_atemp.py` |
|---|---|---|
| Main purpose | Establish a baseline and compare linear with nonlinear models | Remove a redundant variable and compare count-target modeling methods |
| Includes `atemp` | Yes | No |
| Validation approach | One fixed train/test split | Three expanding validation windows plus final test |
| Candidate models | Baseline, Ridge, squared-error Boosting | Baseline, squared-error, Poisson, and log-target Boosting |
| Best casual test R² | 0.862 | **0.875** |
| Best registered test R² | 0.873 | **0.885** |
| Best total test R² | 0.880 | **0.892** |
| Final role | Documents the improvement from baseline to nonlinear modeling | Selects the final per-rider models using expanding-window validation |

The scripts are two stages of the same modeling process rather than conflicting alternatives. `model.py` establishes that Gradient Boosting is necessary and outperforms both the baseline and Ridge. `model_non_atemp.py` then performs variable selection and loss-function comparison, producing a simpler and more accurate final candidate.

## 9. Final model selection

The final approach selected by the code is:

```text
Casual → log_target_boosting
Registered → squared_error_boosting
```

The two predictions are then combined:

```text
Log-target model for casual riders
Squared-error model for registered riders
total prediction = casual prediction + registered prediction
```

The reasons for selecting this approach are:

1. **Pre-test selection.** Both models have the lowest mean validation RMSE for their rider group.
2. **No test-driven model choice.** The final testing period is used for evaluation rather than selection.
3. **Rider-specific behavior.** Different loss treatments are allowed because casual and registered demand have different distributions and patterns.
4. **Simpler feature set.** Removing `atemp` reduces redundancy without harming performance.
5. **Strong final performance.** The combined model reaches a test R² of 0.880 and reduces total RMSE by 45.67% relative to the historical baseline.

Poisson remains the strongest post-selection candidate: it achieved a total test R² of 0.892. However, because it did not win the predefined mean validation RMSE rule, it should be confirmed on a new future holdout period before being declared the formal winner.

## 10. Is additional model training necessary?

### For the current course or project report: no

Repeatedly calling `.fit()` with the same data, variables, and hyperparameters will not provide new information. The models use `random_state=42`, so repeated runs produce the same results. The project already includes:

- A clear historical baseline.
- A linear benchmark.
- Multiple boosting loss functions.
- Chronological validation.
- A held-out testing period.
- Complete metrics for casual, registered, and total demand.

The model search can therefore stop for the current report, and the validation-selected casual log-target and registered squared-error models can be reported as the formal final combination.

### For deployment on future data: one final fit is appropriate

After the model structure has been fixed and no further tuning will be performed, the selected casual log-target and registered squared-error models can be refitted using all 17,379 observed rows. This allows deployment models to learn from the most recent test-period observations. However, the previously reported test metrics cannot be treated as a new independent evaluation of the refitted models.

Additional model research is justified only if the project requires a higher R² threshold, new future observations become available, real-time lag features can be used, or a saved production model and automated update process are required. Otherwise, testing more models on the same test period would increase the risk of overfitting the evaluation data.

## 11. Interpretation of the evaluation metrics

- **MAE:** the average absolute hourly prediction error in rental counts; lower is better.
- **RMSE:** places greater weight on large errors and missed demand peaks; lower is better.
- **R²:** the proportion of demand variation explained by the model; values closer to 1 are better.
- **Mean error:** prediction minus actual demand; a negative value indicates systematic underprediction.

This is a regression and forecasting problem, so classification accuracy is not an appropriate metric. Model quality should be reported using MAE, RMSE, and R² together.

## 12. Report-ready summary

The first experiment used a historical average, Ridge regression, and Histogram Gradient Boosting to forecast casual and registered rentals separately. Gradient Boosting substantially outperformed the baseline and Ridge, producing test R² values of 0.862 for casual riders and 0.873 for registered riders. These results indicate that bike demand has important nonlinear relationships with time, weather, and calendar variables.

The second experiment removed `atemp`, which was highly correlated with temperature, and used expanding-window validation to compare squared-error, Poisson, and log-target Boosting. Mean validation RMSE selected log-target Boosting for casual riders and squared-error Boosting for registered riders. On the final test period, these models produced an MAE of 11.089, RMSE of 20.987, and R² of 0.859 for casual riders, and an MAE of 45.260, RMSE of 65.532, and R² of 0.876 for registered riders. Their combined total-demand MAE was 51.976, RMSE was 75.256, and R² was 0.880, reducing RMSE by 45.67% relative to the historical-average baseline. Poisson achieved a higher post-selection test R² of 0.892 and remains a candidate for confirmation on new data.

The final code-selected approach therefore uses log-target Histogram Gradient Boosting for casual riders and squared-error Histogram Gradient Boosting for registered riders, both without `atemp`. The current results satisfy the model-comparison objective, so no further tuning should be performed on the same test set. Poisson should be evaluated on new future data before replacing the validation-selected models.
