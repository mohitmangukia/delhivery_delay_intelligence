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