# Near-Minimum Prediction Scenario: When R² → -∞

## Overview

This document explains the critical scenario where predictions cluster near the minimum value (`y_min`) with very low variance, causing `SS_total` to approach zero and R² score to decline towards negative infinity, even with relatively low RMSE.

## The Problem

### Mathematical Foundation

The R² score is defined as:

```
R² = 1 - (SS_residual / SS_total)
```

Where:
- `SS_total = Σ(y_i - ȳ)²` - Total sum of squares (variance of ground truth)
- `SS_residual = Σ(ŷ_i - y_i)²` - Residual sum of squares (prediction errors)

### Critical Condition

When ground truth data has **extremely low variance** concentrated near the **minimum value**, `SS_total` approaches zero:

```
Low variance → Small Σ(y_i - ȳ)² → SS_total ≈ 0
```

Even small prediction errors result in:

```
R² = 1 - (SS_residual / SS_total)
R² = 1 - (33.92 / 7.35)  ← SS_residual > SS_total
R² = 1 - 4.62
R² = -3.62  ← Highly negative!
```

## Example from the Diagnostic Plot

### Scenario Parameters

- **Ground Truth**: 
  - Mean: ~45.5
  - Standard deviation: 0.3 (extremely low!)
  - Variance: 0.073 (nearly zero)
  - Range: ~1.5 (data clustered near y_min = 45.0)

- **Predictions**:
  - Centered near y_min = 45.0
  - Small random noise (std = 0.2)

### Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | -3.616 | Highly negative! Model worse than baseline |
| **RMSE** | 0.582 | Relatively low in absolute terms |
| **SS_total** | 7.35 | Nearly zero (low variance) |
| **SS_residual** | 33.92 | Larger than SS_total |
| **Ratio** | 4.62 | SS_residual / SS_total >> 1 |

### Key Insight

**Low RMSE does NOT guarantee good R² when ground truth variance is extremely low!**

The RMSE of 0.582 might seem small, but relative to the tiny variance in the ground truth (σ² = 0.073), it's enormous:

```
RMSE² / Variance = 0.582² / 0.073 = 4.64
```

This means the model's errors are **4.64 times larger** than the natural variance in the data!

## Why This Matters

### In Building Height Estimation

This scenario can occur when:

1. **Data is near-uniform**: Buildings in a region have very similar heights (e.g., all ~45m)
2. **Limited range**: Height measurements cluster around minimum values
3. **High precision required**: Small deviations matter more when the range is tiny

### Practical Implications

1. **Always report variance**: Include ground truth statistics (mean, std, range)
2. **Context matters**: RMSE must be interpreted relative to data variance
3. **Use multiple metrics**: Neither R² nor RMSE alone tells the full story
4. **Baseline comparison**: Compare predictions to simply predicting the mean

## Visual Interpretation

The diagnostic plot shows:

1. **Left panel**: Scatter plot with all points clustered in a tiny region near (45, 45)
   - Green lines (mean) are close to all points
   - Red line (perfect prediction) passes through the cluster
   - Orange lines (residuals) are small but significant relative to the cluster size

2. **Top-right panel**: Residual plot shows small but non-zero errors
   
3. **Bottom-right panel**: Residual distribution is narrow but exists

4. **Right panel**: Statistical breakdown emphasizes:
   - SS_total ≈ 0 (critical warning)
   - Even tiny errors → large ratio → highly negative R²

## Comparison with Mean-Based Scenario

| Aspect | Near-Mean (Previous) | Near-Min (New) |
|--------|----------------------|----------------|
| Data location | Around mean (~50) | Around minimum (~45) |
| Variance | Low (σ² ~ 4) | Extremely low (σ² ~ 0.073) |
| SS_total | Small (~400) | Nearly zero (~7.35) |
| R² sensitivity | Moderate | Extreme |
| R² value | ~-0.09 | ~-3.62 |
| RMSE | ~2.0 | ~0.58 |

The near-min scenario is **more extreme** because:
- SS_total is smaller (closer to zero)
- R² becomes more sensitive to any error
- Even excellent RMSE can't prevent highly negative R²

## Recommendations

When working with low-variance data:

1. ✓ Always compute and report **coefficient of variation** (CV = σ/μ)
2. ✓ Use **normalized RMSE** (NRMSE = RMSE/range or RMSE/mean)
3. ✓ Report **percentage errors** rather than absolute errors
4. ✓ Consider using **alternative metrics** like MAE, MAPE when appropriate
5. ✓ Include **confidence intervals** and uncertainty quantification
6. ⚠ Be cautious interpreting R² when variance is very low

## Mathematical Limit

As variance approaches zero:

```
lim (variance → 0) SS_total → 0
lim (SS_total → 0) R² = 1 - (SS_res / SS_total) → -∞
```

Unless predictions are **perfect** (SS_residual = 0), R² will become arbitrarily negative!

## Conclusion

This diagnostic plot demonstrates a critical edge case in model evaluation: **when ground truth has extremely low variance near a boundary value (like y_min), even models with low RMSE can have highly negative R² scores**. This is not a flaw in the metrics but a fundamental property of how R² accounts for the challenge of the prediction task. A low-variance dataset is "easy" in absolute terms but requires near-perfect predictions to achieve good R² scores.

---

**Related Files:**
- `create_near_min_diagnostic.py` - Script to generate this diagnostic plot
- `investigate_r_score_rmse_synergy.py` - Full investigation including multiple scenarios
- `metric_synergy_visualizations/detailed_diagnostic_near_min_low_variance_neg_inf_r2.png` - The diagnostic visualization
