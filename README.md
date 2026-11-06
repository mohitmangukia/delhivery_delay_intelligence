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