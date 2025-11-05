# R² Score and RMSE Synergy Investigation

## Purpose

This tool investigates the relationship between R² (R-squared) score and RMSE (Root Mean Square Error) metrics for regression tasks. Specifically, it answers the question:

**"Does low RMSE always imply high R² score, or can we have low RMSE with low or even negative R² score?"**

## Key Findings

### Answer: YES, low RMSE can coexist with low R²!

The critical factor is **ground truth variance**:

- **High variance data**: Low RMSE typically implies high R²
- **Low variance data**: Low RMSE can coexist with low R² (or even negative R²)

### Why This Happens

- **RMSE** is scale-dependent: measures absolute prediction error
  - RMSE = 2.0 on data ranging 0-100 → excellent (2% of range)
  - RMSE = 2.0 on data ranging 50-52 → poor (as large as the entire variance)

- **R²** is scale-independent: measures proportion of variance explained
  - R² = 1 - (SS_residual / SS_total)
  - SS_total = variance of ground truth
  - Small SS_total → R² can be low even with small SS_residual

## Usage

```bash
python investigate_r_score_rmse_synergy.py
```

## Requirements

```bash
pip install numpy matplotlib scikit-learn seaborn pandas
```

## Outputs

The script generates organized outputs in three separate directories:

### 📊 `metric_synergy_visualizations/`

1. **scenario_comparison_r2_vs_rmse.png**: Overview scatter plots showing 8 different scenarios
   - Perfect prediction
   - Constant offset
   - High/low variance data
   - Inverse relationships
   - Random predictions
   - **Key scenario**: Near-mean prediction on low variance data

2. **variance_noise_impact_heatmap.png**: Heatmaps showing:
   - R² score as function of ground truth variance and prediction noise
   - RMSE as function of the same parameters

3. **detailed_diagnostic_*.png**: Four detailed diagnostic plots with:
   - **Scatter plot** showing:
     - Mean GT line (ȳ) - green dotted lines showing where SS_total is computed from
     - Perfect prediction line (y=x) - red dashed line
     - Residuals - orange vertical lines from each point to the y=x line
   - **Residual plot**: Shows residuals vs true values
   - **Residual histogram**: Distribution of prediction errors
   - **Statistical breakdown**: Complete metrics including:
     - R² and RMSE calculations
     - Variance decomposition (SS_total, SS_residual, SS_explained)
     - Step-by-step R² calculation showing how SS_total (based on mean GT) is used

### 📋 `metric_synergy_analysis_tables/`
3. **scenarios_summary_metrics.csv**: Summary table with all scenarios and their metrics

### 📄 `metric_synergy_reports/`
4. **detailed_analysis_report.txt**: Comprehensive text report with all insights and analysis

## Scenarios Explored

1. **Perfect Prediction**: RMSE=0, R²=1
2. **Constant Offset**: Low RMSE, High R² (if high variance)
3. **High Variance Data**: Good model, low RMSE, high R²
4. **Low Variance Data**: Small errors but low R²
5. **Near-Mean Prediction (Low Variance)**: **KEY CASE** - Low RMSE, Low R²
6. **Constant Prediction**: R²=0 (baseline)
7. **Inverse Relationship**: Negative R²
8. **Random Predictions**: Very poor performance

## Practical Implications

1. **Always report both metrics** - neither is sufficient alone
2. **Check ground truth variance** when interpreting RMSE
3. **Low RMSE ≠ good model** if R² is also low
4. **Negative R²** means model is worse than predicting the mean
5. For **low variance data**, R² becomes a more critical metric

## Visual Elements Explained

### Understanding the Detailed Diagnostic Plots

#### 1. Mean GT Lines (Green Dotted)
- **Horizontal line at ȳ**: Shows the mean of ground truth values
- **Purpose**: This is the reference point for computing SS_total = Σ(y - ȳ)²
- **Interpretation**: All variance in the data is measured relative to this line

#### 2. Perfect Prediction Line (Red Dashed, y=x)
- **Equation**: y_pred = y_true
- **Purpose**: Represents perfect predictions where every point falls exactly on this line
- **Interpretation**: Distance from this line represents prediction error

#### 3. Residuals (Orange Vertical Lines)
- **Definition**: Vertical distance from a blue point to the red dashed line (y=x)
- **Formula**: Residual = ŷ - y (predicted minus true)
- **Direction**: 
  - Line goes **up** from red line → **over-prediction** (positive residual)
  - Line goes **down** to red line → **under-prediction** (negative residual)
- **Relation to RMSE**: RMSE = √(mean of squared residuals)

#### 4. Blue Points (Predictions)
- **Coordinates**: (y_true, y_pred)
- **Vertical distance to y=x line**: The residual for that prediction
- **Vertical distance to green line**: Contribution to SS_total

#### Key Insight
- **SS_total** is computed from green lines (variance around mean ȳ)
- **SS_residual** is computed from distances to red line (prediction errors)
- **R² = 1 - (SS_residual / SS_total)** compares these two variance measures

## Mathematical Background

### R² Score
$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} = 1 - \frac{SS_{res}}{SS_{tot}}$$

- Range: (-∞, 1], where 1 is perfect
- R² = 0 means predicting the mean
- R² < 0 means worse than predicting the mean

### RMSE
$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

- Range: [0, ∞), where 0 is perfect
- Same units as the target variable
- Sensitive to outliers

## Example Use Cases

### Case 1: Building Height Estimation (High Variance)
- Heights range: 0-100m (variance: ~833 m²)
- RMSE: 5m
- R²: 0.97
- **Interpretation**: Excellent model ✓

### Case 2: Rooftop Height Estimation (Low Variance)
- Heights range: 48-52m (variance: ~1.3 m²)
- RMSE: 0.8m
- R²: 0.50
- **Interpretation**: Low RMSE but poor variance explanation ⚠

## Recommendations

For building height estimation tasks:

1. Report both RMSE and R² in your results
2. Include ground truth statistics (mean, std, range)
3. If dataset has mixed variance (e.g., low-rise and high-rise buildings), consider stratified evaluation
4. Use R² to understand if model captures the variability in data
5. Use RMSE to understand absolute error magnitude

## Author

Research Team - Building Height Estimation Project
Date: November 2025
