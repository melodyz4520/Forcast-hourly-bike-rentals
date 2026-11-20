# No-`atemp` Boosting Model Evaluation

## 1. 实验设计

新模型文件 `model_non_atemp.py` 完全删除了 `atemp`，保留 `temp`，并分别为 casual 和 registered riders 比较以下模型：

1. `squared_error_boosting`：普通 squared-error Histogram Gradient Boosting。
2. `poisson_boosting`：使用 Poisson loss 的 Histogram Gradient Boosting，适合非负计数目标。
3. `log_target_boosting`：先对目标应用 `log1p`，预测后使用 `expm1` 转回原始单位。
4. `historical_average`：按照 working day 和 hour 计算的历史平均 baseline。

所有候选模型使用相同的特征、模型复杂度和时间窗口，确保比较公平。模型选择使用最终测试期之前的三个 expanding-window validation periods：

- 2011-12-01 至 2012-02-29。
- 2012-03-01 至 2012-05-31。
- 2012-06-01 至 2012-08-31。

三个窗口的平均 validation RMSE 是主要选择标准。最终测试集仍为 2012-09-01 至 2012-12-31。

## 2. Expanding-window validation 结果

| Rider | Model | Mean MAE | Mean RMSE | Mean R² | Mean error |
|---|---|---:|---:|---:|---:|
| Casual | Historical average | 26.509 | 41.743 | -0.184 | -11.892 |
| Casual | Squared-error boosting | 16.174 | 26.126 | 0.670 | -1.398 |
| Casual | Poisson boosting | 13.614 | 24.258 | **0.701** | -5.270 |
| Casual | Log-target boosting | **13.388** | **23.813** | 0.699 | -7.117 |
| Registered | Historical average | 70.274 | 99.462 | 0.611 | -57.480 |
| Registered | Squared-error boosting | **46.813** | **68.542** | **0.749** | -32.452 |
| Registered | Poisson boosting | 47.148 | 71.116 | 0.726 | -39.196 |
| Registered | Log-target boosting | 50.040 | 75.491 | 0.709 | -43.700 |

按照预先确定的 mean validation RMSE：

- Casual 选择 `log_target_boosting`，mean RMSE = 23.813。
- Registered 选择 `squared_error_boosting`，mean RMSE = 68.542。

Casual 的 log-target 与 Poisson 非常接近：log-target 的平均 RMSE 更低，而 Poisson 的平均 R²略高。Registered 的 squared-error 在三个窗口平均结果中更好。

## 3. Casual train/test 结果

| Model | Train MAE | Train RMSE | Train R² | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| Historical average | 21.099 | 33.887 | 0.498 | 21.916 | 37.814 | 0.543 |
| Squared-error boosting | 7.399 | 11.696 | 0.940 | 11.320 | 20.426 | 0.867 |
| Poisson boosting | **6.742** | **11.134** | **0.946** | **10.876** | **19.781** | **0.875** |
| Log-target boosting | 7.601 | 13.914 | 0.915 | 11.089 | 20.987 | 0.859 |

最终测试中 Poisson 的 casual 指标最好：

- RMSE 比 historical baseline 降低 47.69%。
- Test R² 达到 0.875。
- 相比原始包含 `atemp` 的 Gradient Boosting，R² 从 0.8618 提升至 0.875。

但是，正式的模型选择是在查看最终测试结果之前完成的，所以当前可复现的选择仍是 validation 选出的 log-target。Poisson 应作为下一轮锁定验证中的最强候选，而不能仅因为本次 test 更高就反向修改选择规则。

## 4. Registered train/test 结果

| Model | Train MAE | Train RMSE | Train R² | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| Historical average | 51.120 | 79.824 | 0.682 | 81.402 | 116.752 | 0.606 |
| Squared-error boosting | 17.095 | 26.674 | 0.965 | 45.260 | 65.532 | 0.876 |
| Poisson boosting | **15.653** | **25.156** | **0.968** | **42.177** | **63.114** | **0.885** |
| Log-target boosting | 17.329 | 29.258 | 0.957 | 46.186 | 68.746 | 0.863 |

最终测试中 Poisson 同样最好：

- RMSE 比 historical baseline 降低 45.94%。
- Test R² 达到 0.885。
- 平均低估从 baseline 的 58.56 次减少到 23.68 次。
- 相比原始包含 `atemp` 的 Gradient Boosting，R² 从 0.8734 提升至 0.885。

Registered 的 validation winner 是 squared-error boosting。Poisson 在前两个 validation windows 并未稳定胜出，但在最近的 2012-06 至 2012-08 window 和最终测试期表现最好。这可能说明随着需求水平增长，Poisson loss 更适合后期的计数分布，也可能只是时间段差异，需要更多滚动窗口确认。

## 5. Total rentals 结果

| Test model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Historical average | 94.780 | 138.527 | 0.595 |
| Squared-error for both riders | 51.326 | 74.211 | 0.884 |
| Poisson for both riders | **48.112** | **71.676** | **0.892** |
| Log-target for both riders | 52.735 | 78.082 | 0.871 |
| Validation-selected combination | 51.976 | 75.256 | 0.880 |

如果在最终测试结果上进行事后比较，Poisson 对两类用户都最好，并且合计需求的 R²达到 0.892。但是 validation-selected combination 才是未使用测试集选模的正式结果，其总需求 R²为 0.880。

## 6. 最终模型选择

本实验应区分两个结论：

### 正式、无测试泄漏的选择

- Casual：`log_target_boosting`。
- Registered：`squared_error_boosting`。

这是代码保存到 `non_atemp_model_selection.csv` 的选择，也是依据 expanding-window validation RMSE 得出的可复现结论。

### 下一轮最值得验证的模型

- Casual：`poisson_boosting`。
- Registered：`poisson_boosting`。

Poisson 在最终测试集上对 casual、registered 和 total 都取得最高 R²和最低 MAE/RMSE，因此它是最有潜力的统一模型。但是若现在直接因为 test score 而宣布 Poisson 为正式 winner，会把测试集变成另一个 validation set，使最终分数偏乐观。

推荐做法是：把 Poisson 预先锁定为下一轮 candidate，然后使用新的时间留出区间、更多 rolling-origin folds，或者未来新数据进行确认。如果 Poisson 继续稳定胜出，再将其作为最终 production model。

## 7. 是否改善了模型

删除 `atemp` 是有效的：普通 squared-error boosting 已经比原模型略有提升。

| Rider | 原模型 Test R² | 删除 `atemp` 后 squared-error R² |
|---|---:|---:|
| Casual | 0.8618 | 0.8666 |
| Registered | 0.8734 | 0.8758 |

Poisson loss 在本次最终测试上进一步提升到 casual 0.875、registered 0.885。因此，本轮实验确认了两点：

1. `atemp` 可以删除，模型更简单且表现略好。
2. Poisson loss 很有潜力，但仍需要独立时间验证确认其稳定性。

## 8. 输出文件

运行 `python3 model_non_atemp.py` 后会生成：

- `results/non_atemp_validation_metrics.csv`：每一个 validation window 的详细指标。
- `results/non_atemp_validation_summary.csv`：三个窗口的平均指标。
- `results/non_atemp_model_selection.csv`：按照 validation RMSE 选择的模型。
- `results/non_atemp_train_test_metrics.csv`：所有模型的 train/test 指标。
- `results/non_atemp_test_predictions.csv`：最终测试期的真实值和各模型预测值。

总体结论：按照严格的 validation 规则，casual 应选 log-target boosting，registered 应选 squared-error boosting；从最终测试表现看，Poisson 是下一轮最值得锁定并独立验证的统一候选模型。
