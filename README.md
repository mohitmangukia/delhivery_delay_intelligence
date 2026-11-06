## Data findings



The cleaned modelling dataset contains **1,200 unique shipments**, with 278 delayed shipments and an overall delay rate of **23.17%**.



1. **Delay by service type:** Express shipments were delayed most often, with a delay rate of **48.97%** (119 out of 243 shipments), compared with 20.56% for Standard and 8.57% for Economy.



2. **Delay by hub tier:** The delay rate increases substantially with hub tier: **4.00%** at Tier 1, **21.50%** at Tier 2, and **44.00%** at Tier 3.



3. **Delay by distance:** Delay rates increase across distance bands: **2.19%** for 0–200 km, **13.68%** for 201–400 km, **43.64%** for 401–600 km, and **85.23%** for shipments above 600 km.



4. **Correlation with delivery hours:** The three numeric columns most strongly correlated with `delivery_hours`, ranked by absolute Pearson correlation and excluding `delivery_hours` itself, are:

   - `promised_hours`: **+0.8372**

   - `distance_km`: **+0.4013**

   - `n_stops`: **+0.3247**



5. **Log transformation of declared value:** The filled `declared_value_inr` column has a mean of **1533.0542** and a median of **1121.5000**, indicating a right-skewed distribution; applying the natural logarithm compresses larger values, bringing the transformed mean (**7.0277**) and median (**7.0224**) much closer together.



## Part B — Delivery Time Regression



### Objective



Predict shipment delivery time in hours using the engineered shipment and hub features created in Part A.



### Data splitting and scaling



The 1,200 historical shipments were divided into:



- Training: 840 rows (70%)

- Validation: 180 rows (15%)

- Test: 180 rows (15%)



Feature scaling was fitted only on the training data to prevent data leakage. The maximum absolute training-feature mean after scaling was approximately 0, while the corresponding test value was 0.2187.



### Baseline model



A mean baseline predicted the average training delivery time (57.7493 hours) for every validation shipment.



- Validation MAE: 15.7952 hours

- Validation RMSE: 18.8552 hours

- Validation R²: -0.0212



The negative R² indicates that predicting a constant training mean performs worse than using the validation-set mean as a predictor. This establishes the need for a model that learns relationships between shipment features and delivery time.



### Decision Tree depth experiment



Eight tree depths were evaluated. Depth 6 achieved the lowest validation RMSE of 6.2796 hours.



A shallow tree (depth 1) had high training and validation errors, indicating underfitting (high bias). The unlimited-depth tree achieved zero training RMSE but a validation RMSE of 7.8368 hours, indicating overfitting (high variance). Depth 6 provided a better balance between model complexity and generalization.



### Candidate model comparison



| Model | Validation RMSE | Validation R² |

|---|---:|---:|

| Linear Regression | 4.8912 | 0.9313 |

| Gradient Boosting | 4.9165 | 0.9306 |

| Random Forest | 5.5095 | 0.9128 |

| Decision Tree (depth 6) | 6.2796 | 0.8867 |



Linear Regression achieved the lowest validation RMSE and was selected as the final model. Gradient Boosting performed almost identically, but Linear Regression was marginally better on the specified validation split.



### Gradient Boosting learning-rate experiment



The experiment compared learning rates of 0.01, 0.1, and 1.0 with 100 estimators.



| Learning rate | Validation RMSE | Validation R² |

|---:|---:|---:|

| 0.01 | 10.0018 | 0.7127 |

| 0.10 | 4.9165 | 0.9306 |

| 1.00 | 7.3701 | 0.8440 |



A learning rate of 0.01 learned too slowly with only 100 estimators. A rate of 1.0 was too aggressive and produced weaker validation performance. A rate of 0.1 provided the best balance.



### Cross-validation



Five-fold cross-validation was performed on the combined training and validation data (1,020 rows), using the same shuffled KFold configuration for both leading candidates.



| Model | Mean R² | Standard deviation |

|---|---:|---:|

| Linear Regression | 0.9261 | 0.0048 |

| Gradient Boosting | 0.9353 | 0.0045 |



Both models passed the cross-validation requirements, with high mean R² and low variation across folds.



Gradient Boosting achieved a higher average cross-validation R², even though Linear Regression had the lower validation RMSE. The final model selection followed the capstone's validation-RMSE criterion, so Linear Regression was retained. Its simpler structure also makes it easier to interpret.



### Final test-set evaluation



The selected Linear Regression model was evaluated on the untouched test set.



| Metric | Test result |

|---|---:|

| MAE | 3.5671 hours |

| MSE | 20.4624 |

| RMSE | 4.5235 hours |

| R² | 0.9493 |



The final model achieved a test R² of 0.9493, exceeding the required benchmark of 0.88. Its predictions explain approximately 94.93% of the variation in delivery hours in the held-out test data.



**Final selected regression model: Linear Regression.**



## Part C — Delay Classification



### 1. Baseline performance



The blind baseline predicts every shipment as on-time. It achieves an accuracy of **76.67%**, because most shipments in the dataset are not delayed.



However, it identifies none of the delayed shipments:



- Precision: 0.00

- Recall: 0.00

- F1-score: 0.00



This demonstrates why accuracy alone is insufficient for an imbalanced classification problem. A model that never identifies a delayed shipment is not operationally useful.



### 2. Classifier comparison



The three trained classifiers performed better than the blind baseline:



| Model | Validation accuracy | Precision | Recall | F1-score |

|---|---:|---:|---:|---:|

| Logistic Regression | 0.9056 | 0.7778 | 0.8333 | 0.8046 |

| Decision Tree (depth 6) | 0.8833 | 0.7561 | 0.7381 | 0.7470 |

| Random Forest | 0.9056 | 0.8378 | 0.7381 | 0.7848 |



The Decision Tree depth experiment selected depth 6 based on the highest validation F1-score. The trained models all exceeded the baseline accuracy and met their respective F1 acceptance thresholds.



### 3. Feature signal and noise



The top seven features ranked by ANOVA F-score were:



1. `distance_km`

2. `hub_tier`

3. `hub_daily_capacity`

4. `n_stops`

5. `promised_hours`

6. `rain_flag`

7. `sale_event_flag`



All five designated noise columns ranked outside the top seven and had F-scores below 10.



The combined Random Forest feature importance of these five noise columns was **22.44%**, within the expected 15–30% range. This shows that Random Forest importance can distribute some importance to weak or noisy variables, even when their individual univariate F-scores are low.



### 4. Feature selection experiment



The validation F1-scores for different feature counts were:



| Number of features (k) | Validation F1 |

|---:|---:|

| 5 | 0.6988 |

| 7 | 0.8333 |

| 10 | 0.7857 |

| 12 | 0.8049 |

| 17 | 0.7848 |



Using seven features produced the strongest recorded result. Adding more features did not improve validation F1 in this experiment. Therefore, `k=7` was selected for the final classification pipeline.



### 5. Hyperparameter tuning



Grid Search evaluated 27 parameter combinations using five-fold cross-validation, resulting in 135 fits. Randomized Search evaluated nine combinations, resulting in 45 fits.



| Search configuration | Best CV accuracy | Validation F1 |

|---|---:|---:|

| Grid Search | 0.8952 | 0.7750 |

| Randomized Search | 0.8905 | 0.7848 |

| Default Random Forest on 7 features | — | 0.8333 |



Neither tuned configuration improved validation F1 over the default Random Forest using seven selected features.



The validation set contains only 180 shipments. Therefore, small differences in validation performance should not be treated as conclusive evidence that one configuration will generalize better. The simpler default configuration was retained.



### 6. Operational decision threshold



The operational rule was to select the highest tested probability threshold that maintained validation recall of at least 0.80.



The selected threshold was **0.5**, with:



- Validation precision: 79.55%

- Validation recall: 83.33%

- Validation F1-score: 81.40%



Increasing the threshold to 0.6 improved precision to 88.89%, but reduced recall to 76.19%, below the operational requirement. Therefore, 0.5 was retained.



This threshold balances identifying delayed shipments against generating false delay alerts.



### 7. Final test performance



The final model was evaluated once on the held-out test set using the selected threshold of 0.5.



| Metric | Test result |

|---|---:|

| Accuracy | 0.8667 |

| Precision | 0.6889 |

| Recall | 0.7561 |

| F1-score | 0.7209 |



Confusion matrix:



| | Predicted on-time | Predicted delayed |

|---|---:|---:|

| Actual on-time | 125 | 14 |

| Actual delayed | 10 | 31 |



All three required test acceptance thresholds were achieved:



- Precision ≥ 0.45: **Passed**

- Recall ≥ 0.55: **Passed**

- F1-score ≥ 0.60: **Passed**



Test recall was 75.61%, compared with validation recall of 83.33%, a decrease of 7.72 percentage points. The test set contains only 41 delayed shipments, so each additional missed delayed shipment has a noticeable effect on recall. This difference should be reported transparently rather than interpreted as definitive evidence of overfitting.



### 8. Final model configuration



| Parameter | Selected value |

|---|---|

| Feature selector | SelectKBest (ANOVA F-score) |

| Number of selected features | 7 |

| Classifier | RandomForestClassifier |

| Number of estimators | 100 |

| Maximum depth | Unlimited |

| Minimum samples per leaf | 1 |

| Decision threshold | 0.5 |

| Configuration selected from | `default_forest_same_7_features` |



The final pipeline was fitted on the training set. The validation set was used for model and threshold selection, and the held-out test set was reserved for the final evaluation.

## How to run

The following commands run the complete pipeline from the repository root. Use the pinned dependencies in `requirements.txt` so that the saved scikit-learn pipeline is recreated in a compatible environment.

```powershell
git clone <repository-url>
cd delhivery_delay_intelligence

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

On macOS/Linux, activate the environment with:

```bash
source .venv/bin/activate
```

Run the scripts in the order shown. `part_a_clean_eda.py` creates the modelling table and historical imputation medians; Parts B and C create their evaluation artefacts and the selected classification configuration; `part_d_segments.py` creates the hub segments; `package_model.py` packages and reload-checks the final classifier; and `predict_delay.py` scores the 48 incoming shipments.

`package_model.py` regenerates `models/delay_model.pkl`. Keep the pinned scikit-learn version from `requirements.txt` when loading or recreating this artifact.

## Operations recommendation

### Hub-level operating segments

The three-cluster solution produces operationally distinct groups:

| Segment | Hubs | Mean delivery time | Delay rate | Mean distance | Mean stops |
|---|---:|---:|---:|---:|---:|
| Standard | 7 | 47.24 hours | 2.57% | 188.93 km | 55.37 |
| Watch | 8 | 56.81 hours | 15.50% | 314.18 km | 70.72 |
| Escalate | 9 | 64.65 hours | 46.00% | 470.48 km | 84.34 |

The Escalate segment contains HUB13 (Nagpur, 46%), HUB15 (Kochi, 36%), HUB17 (Patna, 46%), HUB18 (Ranchi, 38%), HUB19 (Guwahati, 62%), HUB20 (Raipur, 48%), HUB21 (Bhubaneswar, 40%), HUB23 (Siliguri, 44%), and HUB24 (Agartala, 54%). These hubs should receive the strongest operational attention: review dispatch planning, route length and stop density, and investigate recurring capacity or process bottlenecks.

Use the segments as practical operating playbooks:

- **Standard:** continue normal monitoring and routine exception handling.
- **Watch:** review daily delay indicators and investigate deterioration before it becomes persistent.
- **Escalate:** review the hub's operating constraints with the hub manager and prioritize corrective action.

### Tier-versus-segment mismatches

Four hubs do not align with the segment expected from their tier:

| Hub | Location | Tier | Observed segment | Operational interpretation |
|---|---|---:|---|---|
| HUB07 | Kolkata | 1 | Watch | Do not assume low operational risk from Tier 1 status; apply the Watch playbook. |
| HUB13 | Nagpur | 2 | Escalate | The observed delay pattern warrants escalation despite its Tier 2 classification. |
| HUB15 | Kochi | 2 | Escalate | Apply the Escalate playbook rather than relying on the tier label. |
| HUB22 | Dehradun | 3 | Watch | The observed segment is less severe than the tier-based expectation; retain Watch monitoring rather than automatically escalating. |

This comparison shows why hub tier should not be used as the sole operational risk indicator. The segment assignment reflects the combined hub profile used in clustering.

### Shipment-level alert policy

The selected classifier threshold is **0.50**. On the held-out test set, precision was **68.89%** and recall was **75.61%**. In practical terms, approximately 6.9 of every 10 flagged shipments were delayed, while the model identified approximately 7.6 of every 10 delayed shipments in that test sample.

For the 48 incoming shipments, the scoring pipeline produced:

- **8 flagged shipments**
- **6 hub-manager escalations** (flagged shipment in an Escalate hub)
- **2 proactive customer calls** (flagged shipment in Standard or Watch)
- **40 shipments with no action**

The action mapping is deliberately simple: a flagged shipment in an Escalate hub is sent to the hub manager; another flagged shipment triggers a proactive customer call; an unflagged shipment receives no action.

### Why three hub segments?

The separation ratio improves as K increases: 2.894 for K=2, 5.393 for K=3, 6.305 for K=4, and 6.887 for K=5. However, the smallest cluster contains four hubs for K=4 and two hubs for K=5. Those small groups would be difficult to support with their own stable operating playbooks. K=3 gives a useful separation ratio while keeping cluster sizes at 9, 8, and 7 hubs, which is more practical for the three defined actions: Standard, Watch, and Escalate.

## Production plan

The capstone pipeline demonstrates several production-oriented characteristics. The following controls would be appropriate when moving from this synthetic exercise to a live operational system.

| Characteristic | Evidence in this project | Production application |
|---|---|---|
| Accuracy | Regression and classification are evaluated on held-out test data. | Re-evaluate against matured delivery outcomes and business costs before release. |
| Reliability | `package_model.py` reloads the saved pipeline and compares test probabilities; the maximum absolute difference was 0.0. | Run a model-load and prediction smoke test during deployment and after every model package change. |
| Reproducibility | Fixed random seeds, pinned dependencies, persisted threshold and feature-order files. | Version the data snapshot, code commit, dependency lock and model artefacts together. |
| Performance | The scoring workflow processes 48 shipments with 17 input features. | Benchmark end-to-end scoring at expected peak batch size before choosing batch or online serving. |
| Maintainability | Cleaning and feature engineering are shared in `src/shipment_prep.py`; modelling, packaging and inference are separate scripts. | Keep transformations in the shared module and add automated tests for schema, types and action rules. |
| Consistency | Inference reuses the historical imputation medians and saved feature order. | Treat preprocessing values, feature order, model and threshold as one versioned release unit. |
| Monitorability | Each scored shipment has a probability, flag, hub segment and action. | Store predictions and join them to eventual delivery outcomes for ongoing performance monitoring. |

### Proposed monitoring controls

These are proposed operational alert thresholds, not thresholds learned from the capstone dataset. Calibrate them with the operations team after observing real production volumes.

1. **Delayed-delivery rate:** investigate if the matured weekly delay rate exceeds 20%, or rises by more than 5 percentage points against the comparable baseline.
2. **Classifier recall:** review the model and operating threshold if recall on matured, labelled shipments falls below 70% over a rolling evaluation window.
3. **Flag rate:** investigate if more than 25% or fewer than 5% of shipments are flagged in a week; first check for genuine operational changes and input-data problems before changing the threshold.
4. **Input quality:** alert if more than 1% of records have missing required model features after preprocessing, or if any shipment fails its hub lookup.
5. **Action completion:** track the share of flagged shipments for which the assigned call or hub-manager review was completed within the agreed service window.

### Retraining and event-season handling

A reasonable initial policy is to evaluate the model monthly once enough newly delivered shipments have matured and their actual delivery outcomes are available. Retrain only when the new labelled data is sufficient and evaluation shows a sustained performance or drift issue; also trigger an out-of-cycle review after material changes to routes, hub operations, service definitions, scanning systems or feature generation.

The **15–28 September sale window** should be evaluated against comparable sale-period data, not against ordinary non-sale weeks. Compare delay rate, flag rate, precision and recall for the sale window with previous comparable sale windows, and monitor the `sale_event_flag` feature and the mix of services, distances and hubs during the event. The current synthetic dataset is not sufficient to establish how the model will behave during a real high-volume sale.

Examples of data drift to monitor include changes in distance and stop-count distributions, hub and hub-tier mix, service-type mix, rain and sale-event frequency, parcel-weight distribution, and scanner or labelling process identifiers. Also monitor target drift: the relationship between those inputs and actual delay outcomes may change even when input distributions look stable.

## Limitations

1. **Synthetic data:** the dataset is generated for this capstone and is not a substitute for real Delhivery shipment, route, hub or delivery-event data. Results do not establish real-world operational performance.
2. **Small evaluation sample:** the classification test set contains 180 shipments, including 41 delayed shipments. Precision and recall can move noticeably when only a few predictions change.
3. **Single split and threshold selection:** model and threshold choices were made using the specified train/validation split. Independent temporal and geographic validation is needed before operational use.
4. **Limited hub population:** clustering is based on 24 hubs. Segment membership and the resulting playbooks may change with a larger and more representative hub network.
5. **No live operational signals:** the current features do not include live traffic, road closures, real-time weather severity, vehicle availability or continuously updated hub capacity.
6. **Probability calibration is not established:** the 0.50 threshold was selected using the validation-recall rule, not a calibrated probability model or a measured financial cost of false positives and false negatives.
7. **Temporal generalization is untested:** the random train/validation/test split does not fully simulate forecasting future shipments. A time-based holdout should be used with real historical data to test performance on later periods.
8. **Operational actions are simplified:** the current action mapping assigns a call or manager escalation from the predicted flag and hub segment. It does not account for staffing, contact history, action capacity, customer preferences or whether an intervention actually prevents a delay.
