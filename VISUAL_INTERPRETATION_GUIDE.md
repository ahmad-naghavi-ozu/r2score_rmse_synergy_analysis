# Quick Reference Guide: Interpreting the Diagnostic Plots

## Visual Elements in Detailed Diagnostic Plots

### 📍 Blue Points (Scatter)
- **Coordinates**: (ground truth, prediction) = (y, ŷ)
- Each point represents one data sample

### 📏 Red Dashed Line (y = x)
- **Meaning**: Perfect prediction line
- **If point is ON this line**: Perfect prediction for that sample
- **Distance to this line**: Represents the prediction error (residual)

### 🟢 Green Dotted Lines (Horizontal & Vertical)
- **Position**: At the mean of ground truth (ȳ)
- **Purpose**: Shows the reference point for computing SS_total
- **Formula**: SS_total = Σ(y - ȳ)²
- **Interpretation**: Measures total variance in ground truth data

### 🟠 Orange Vertical Lines
- **What they show**: Visual representation of residuals
- **Starting point**: The blue point (actual prediction)
- **Ending point**: The red dashed line (where perfect prediction would be)
- **Length**: |ŷ - y| = absolute error for that sample

---

## How to Answer Your Questions Using the Plots

### Q1: "How to infer the avg_gt (ȳ) by which SS_total is computed?"

**Answer**: Look at the **green dotted lines**
- The horizontal green line shows ȳ (mean of ground truth)
- The vertical green line is at the same position (x = ȳ)
- The value is printed in the legend: "Mean GT (ȳ=X.XX)"
- In the right panel, see "Ground Truth Statistics → Mean (ȳ)"

**SS_total calculation**:
```
SS_total = Σ(y_i - ȳ)²
```
This measures how far each true value is from the mean.

---

### Q2: "Can I consider the vertical difference between a blue point and the red dashed line as the residual?"

**Answer**: **YES, exactly!** 🎯

The **orange vertical lines** show exactly this:
- They connect each blue point to the red dashed line
- This vertical distance = ŷ - y = residual

**Important notes**:
1. **Direction matters**:
   - Line goes **UP** (point above red line) → **over-prediction** (ŷ > y, positive residual)
   - Line goes **DOWN** (point below red line) → **under-prediction** (ŷ < y, negative residual)

2. **For RMSE calculation**:
   ```
   RMSE = √(mean of all squared residuals)
   RMSE = √(Σ(ŷ_i - y_i)² / n)
   ```

3. **For R² calculation**:
   ```
   SS_residual = Σ(ŷ_i - y_i)²  ← sum of squared orange line lengths
   R² = 1 - (SS_residual / SS_total)
   ```

---

## Reading the Statistical Panel (Right Side)

### Variance Decomposition Section
```
SS_total:     850.17    ← Variance around mean (green lines)
  (Σ(y - ȳ)²)

SS_residual:  26.05     ← Prediction errors (orange lines)
  (Σ(ŷ - y)²)

SS_explained: 824.12    ← Variance captured by model
  (SS_total - SS_residual)
```

### R² Calculation Step-by-Step
Shows exactly how R² is computed:
```
R² = 1 - (SS_res / SS_tot)
R² = 1 - (26.05 / 850.17)
R² = 1 - 0.030637
R² = 0.969363
```

---

## Key Scenarios to Examine

### 1. `detailed_diagnostic_perfect_prediction.png`
- All blue points exactly on red line
- No orange lines (zero residuals)
- R² = 1.000, RMSE = 0.000

### 2. `detailed_diagnostic_high_variance.png`
- Large SS_total (data spread far from green line)
- Small orange lines relative to spread
- **Result**: High R², moderate RMSE

### 3. `detailed_diagnostic_low_variance_low_r2.png` ⚠️ **CRITICAL CASE**
- Small SS_total (data clustered near green line)
- Orange lines may look small in absolute terms
- **But** SS_residual is large relative to SS_total
- **Result**: LOW RMSE but LOW (or negative) R²!
- **This answers your main question!**

### 4. `detailed_diagnostic_inverse_relationship.png`
- Points form opposite trend (negative correlation)
- Very large orange lines
- R² is negative (worse than just predicting mean)

---

## Practical Interpretation Examples

### Example 1: Building Heights (High Variance)
```
Ground truth: 5m to 100m (variance = 900 m²)
RMSE: 5m
Orange lines: Typically 5m long
Green line: At 52.5m (mean)

SS_total = 900 × 100 = 90,000
SS_residual = 5² × 100 = 2,500
R² = 1 - (2,500 / 90,000) = 0.972 ✓ Excellent!
```

### Example 2: Rooftop Fine-tuning (Low Variance)
```
Ground truth: 48m to 52m (variance = 1.3 m²)
RMSE: 0.8m
Orange lines: Typically 0.8m long
Green line: At 50m (mean)

SS_total = 1.3 × 100 = 130
SS_residual = 0.8² × 100 = 64
R² = 1 - (64 / 130) = 0.508 ⚠️ Moderate only!
```

**Same RMSE quality, but very different R² scores due to variance difference!**

---

## Common Misconceptions

❌ **WRONG**: "Small orange lines = good model"
✅ **CORRECT**: "Small orange lines *relative to distance from green line* = good model"

❌ **WRONG**: "Low RMSE always means high R²"
✅ **CORRECT**: "Low RMSE compared to ground truth variance means high R²"

❌ **WRONG**: "R² is just another error metric"
✅ **CORRECT**: "R² measures proportion of variance explained, normalized by data spread"

---

## Summary

| Visual Element | What It Shows | Used For |
|---------------|---------------|----------|
| Blue points | (y, ŷ) coordinates | Actual predictions |
| Red line (y=x) | Perfect prediction | Reference for residuals |
| Green lines (at ȳ) | Mean of ground truth | Computing SS_total |
| Orange lines | Residuals (ŷ - y) | RMSE & R² calculation |
| Distance: point ↔ red line | Prediction error | Individual residual |
| Distance: true value ↔ green line | Deviation from mean | Contribution to SS_total |

**Key Formula**:
```
R² = 1 - (Σ orange_line_length² / Σ distance_to_green_line²)
```

---

Generated by: investigate_r_score_rmse_synergy.py
Date: November 2025
