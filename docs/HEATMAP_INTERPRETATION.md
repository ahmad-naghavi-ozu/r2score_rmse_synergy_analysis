# Variance-Noise Impact Heatmap Interpretation

## Overview

The `variance_noise_impact_heatmap.png` visualizes how two independent factors affect R² and RMSE metrics:
- **Ground Truth Variance (σ²)**: How spread out the true values are (vertical axis)
- **Prediction Noise (σ)**: Standard deviation of prediction errors (horizontal axis)

## Terminology Clarification

- **Variance**: Property of the **ground truth data** only
  - Measures: `Var(y_true) = σ²`
  - Example: Building heights ranging 40-60m have higher variance than 49-51m
  
- **Noise**: Property of the **prediction errors** only  
  - Measures: Standard deviation of errors `σ_noise`
  - Generated as: `y_pred = y_true + random_error(mean=0, std=σ_noise)`

These are independent: you can have high variance data with low noise (good predictions on diverse data) or low variance data with high noise (poor predictions on uniform data).

## Heatmap Insights

### Left Plot: R² Score Heatmap

**Pattern**: Diagonal gradient from green (top-left) to red (bottom-right)

**Key Observations**:
- **Top-left (Green)**: High variance + Low noise → **High R²** (0.9+)
  - Model captures most of the data variance
  - Example: Predicting building heights (40-60m range) with ±1m error → R² ≈ 0.95
  
- **Bottom-right (Red)**: Low variance + High noise → **Negative R²**
  - Model errors exceed the natural data variance
  - Example: Predicting building heights (49-51m range) with ±2m error → R² < 0
  
- **Diagonal boundary**: R² ≈ 0 line
  - When noise equals the standard deviation of ground truth
  - Model performs no better than predicting the mean

**Formula**: R² = 1 - (σ²_noise / σ²_data)
- High variance (σ²_data large) → Even moderate noise gives good R²
- Low variance (σ²_data small) → Same noise gives poor/negative R²

### Right Plot: RMSE Heatmap

**Pattern**: Horizontal bands (independent of vertical position)

**Key Observations**:
- **Horizontal bands only**: RMSE values form horizontal stripes
  - RMSE is **completely independent** of ground truth variance
  - Only depends on prediction noise (horizontal axis)
  
- **RMSE ≈ Prediction Noise (σ)**
  - RMSE directly reflects the magnitude of prediction errors
  - Example: σ_noise = 5 → RMSE ≈ 5, regardless of data variance

**Formula**: RMSE = √(σ²_noise) = σ_noise
- Does NOT account for how difficult the prediction task is
- Same absolute error on narrow vs. wide data ranges

## Critical Insight: Why Low RMSE ≠ High R²

The heatmaps visually demonstrate the **decoupling** of RMSE and R²:

**Scenario**: Prediction noise σ = 1.0

- **High variance data** (σ² = 100):
  - RMSE = 1.0 (low)
  - R² = 1 - (1/100) = 0.99 (excellent)
  - ✓ Both metrics agree: good model

- **Low variance data** (σ² = 1):
  - RMSE = 1.0 (same low value!)
  - R² = 1 - (1/1) = 0.0 (poor)
  - ⚠ Metrics disagree: RMSE says "good", R² says "no better than mean"

- **Very low variance data** (σ² = 0.25):
  - RMSE = 1.0 (same low value!)
  - R² = 1 - (1/0.25) = -3.0 (terrible)
  - ✗ Metrics strongly disagree: RMSE says "good", R² says "worse than mean"

## Practical Implications

### For Model Evaluation:

1. **Always report both metrics** with ground truth statistics (mean, std, range)
2. **Low RMSE alone is misleading** when data variance is low
3. **R² contextualizes RMSE** by comparing against baseline (predicting mean)
4. **Use normalized metrics** when comparing across datasets with different scales

### For Dataset Analysis:

1. **Before training**: Check ground truth variance
   - High variance → RMSE and R² will likely agree
   - Low variance → Expect potential RMSE/R² discrepancy
   
2. **When R² is low despite low RMSE**:
   - Check: Is ground truth variance very small?
   - Consider: Coefficient of Variation (CV = σ/μ)
   - Solution: Use normalized RMSE (NRMSE = RMSE/range) or MAPE

### For Building Height Estimation:

- **Diverse buildings** (heights 10-100m, σ ≈ 25m):
  - RMSE = 3m → R² ≈ 0.985 ✓ Excellent
  
- **Uniform neighborhood** (heights 48-52m, σ ≈ 1m):
  - RMSE = 3m → R² < 0 ✗ Model fails
  - Even though absolute error is same, model captures none of the variation

## Mathematical Relationship

The heatmaps encode the relationship:

```
R² = 1 - (SS_residual / SS_total)
   = 1 - (n·σ²_noise / n·σ²_data)
   = 1 - (σ²_noise / σ²_data)

RMSE = √(SS_residual / n)
     = √(σ²_noise)
     = σ_noise
```

**R²** depends on the **ratio** (noise/variance) → Normalized, scale-independent
**RMSE** depends on **noise only** → Absolute, scale-dependent

## Conclusion

The variance-noise heatmap reveals that:
- R² is a **relative** metric (compares prediction error to data variance)
- RMSE is an **absolute** metric (independent of data characteristics)
- Neither is "better" - they measure different aspects of model performance
- Understanding both together provides complete picture of model quality

For robust model evaluation, always consider the context: what is the baseline difficulty of the prediction task (data variance)?
