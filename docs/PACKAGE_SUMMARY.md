# R² Score and RMSE Synergy Analysis - Complete Package

## 📦 What's Included

This directory contains a comprehensive investigation tool for understanding the relationship between R² score and RMSE metrics.

## 📂 Directory Structure

```
tools/r_score_rmse_synergy/
│
├── 📄 Main Scripts
│   ├── investigate_r_score_rmse_synergy.py  # Main analysis script
│   └── create_visual_guide.py                # Creates annotated guide
│
├── 📖 Documentation
│   ├── README.md                             # Overview and usage
│   ├── VISUAL_INTERPRETATION_GUIDE.md        # How to read the plots
│   └── PACKAGE_SUMMARY.md                    # This file
│
├── 📊 metric_synergy_visualizations/
│   ├── scenario_comparison_r2_vs_rmse.png    # 8 scenarios overview
│   ├── variance_noise_impact_heatmap.png     # R² and RMSE heatmaps
│   ├── visual_guide_annotated.png            # Annotated tutorial
│   ├── detailed_diagnostic_perfect_prediction.png
│   ├── detailed_diagnostic_high_variance.png
│   ├── detailed_diagnostic_low_variance_low_r2.png ⭐ KEY PLOT
│   └── detailed_diagnostic_inverse_relationship.png
│
├── 📋 metric_synergy_analysis_tables/
│   └── scenarios_summary_metrics.csv         # All metrics in table form
│
└── 📄 metric_synergy_reports/
    └── detailed_analysis_report.txt          # Full text analysis

```

## 🎯 Main Question Answered

**"Can you have low RMSE with low or negative R² score?"**

### Answer: YES! ✅

This happens when the ground truth data has **low variance**. The detailed diagnostic plots show exactly how and why.

## 🚀 Quick Start

### 1. Run the Analysis
```bash
cd tools/r_score_rmse_synergy
python investigate_r_score_rmse_synergy.py
```

### 2. View Key Plots

Start with these in order:

1. **visual_guide_annotated.png** - Learn how to read the plots
2. **detailed_diagnostic_low_variance_low_r2.png** ⭐ - The KEY case showing low RMSE + low R²
3. **detailed_diagnostic_high_variance.png** - Compare: same RMSE but high R²
4. **scenario_comparison_r2_vs_rmse.png** - Overview of all 8 scenarios

### 3. Read the Documentation

1. **VISUAL_INTERPRETATION_GUIDE.md** - Explains all visual elements
2. **README.md** - Full documentation
3. **detailed_analysis_report.txt** - Statistical analysis

## 📊 Understanding the Plots

### Color-Coded Elements

| Color | Element | Meaning |
|-------|---------|---------|
| 🟢 Green dotted lines | Mean GT (ȳ) | Shows where SS_total is computed from |
| 🔴 Red dashed line | y = x | Perfect prediction line |
| 🟠 Orange vertical lines | Residuals | Visual representation of (ŷ - y) |
| 🔵 Blue points | Predictions | Each point is (y_true, y_pred) |

### Key Insights from Visual Elements

1. **Mean GT (green lines)**: The horizontal green line shows ȳ, which is used to calculate:
   ```
   SS_total = Σ(y_i - ȳ)²
   ```

2. **Residuals (orange lines)**: The vertical distance from each blue point to the red y=x line represents:
   ```
   residual_i = ŷ_i - y_i
   ```

3. **R² calculation**: Compares orange line lengths to data spread:
   ```
   R² = 1 - (Σ orange_line²) / (Σ distance_to_green_line²)
   ```

## 🔍 Key Scenarios

### Scenario 1: Perfect Prediction
- **Plot**: detailed_diagnostic_perfect_prediction.png
- **R²**: 1.000, **RMSE**: 0.000
- All points on red line, no orange lines

### Scenario 2: High Variance Data ✅
- **Plot**: detailed_diagnostic_high_variance.png
- **R²**: 0.969, **RMSE**: 4.746
- Data spread: 0-100 (high variance)
- **Result**: Low RMSE + High R² ✓

### Scenario 3: Low Variance Data ⚠️ **CRITICAL**
- **Plot**: detailed_diagnostic_low_variance_low_r2.png
- **R²**: -0.086, **RMSE**: 2.206
- Data spread: 48-52 (low variance)
- **Result**: LOW RMSE but NEGATIVE R² ⚠️
- **This is the key case!**

### Scenario 4: Inverse Relationship ❌
- **Plot**: detailed_diagnostic_inverse_relationship.png
- **R²**: -3.099, **RMSE**: 59.029
- Predictions inversely correlated with truth
- Very large orange lines

## 💡 Practical Applications

### For Building Height Estimation

#### Case A: City-wide estimation (High variance)
```
Heights: 5m - 100m
Variance: ~900 m²
RMSE: 5m → R² ≈ 0.97 ✓ Excellent!
```

#### Case B: Single building fine-tuning (Low variance)
```
Heights: 48m - 52m
Variance: ~1.3 m²
RMSE: 0.8m → R² ≈ 0.51 ⚠️ Moderate only!
```

**Same absolute RMSE quality, but very different R² scores!**

## 📝 Answering Your Specific Questions

### Q1: "How to infer the avg_gt by which SS_total is computed?"

**Answer**: Look at the **green dotted lines** in the detailed diagnostic plots:
- The horizontal green line is at y = ȳ (mean of ground truth)
- Value is shown in the legend: "Mean GT (ȳ=XX.XX)"
- Also printed in the right panel under "Ground Truth Statistics"

### Q2: "Can I consider the vertical difference between a blue point and the red dashed line as the residual?"

**Answer**: **YES, exactly!** That's precisely what the **orange vertical lines** show:
- They connect each blue point to the red y=x line
- Their length = |ŷ - y| = absolute residual
- Direction indicates over/under-prediction
- RMSE = √(mean of squared orange line lengths)

## 📈 Outputs Generated

### Visualizations (7 files)
- 1 overview comparison plot
- 1 heatmap analysis
- 4 detailed diagnostic plots
- 1 annotated guide

### Tables (1 file)
- CSV with all scenarios and metrics

### Reports (1 file)
- Complete text analysis with insights

## 🎓 Educational Value

This tool is perfect for:
- Understanding metric limitations
- Teaching R² vs RMSE trade-offs
- Debugging model evaluation issues
- Preparing research presentations
- Writing methodology sections

## ⚠️ Key Takeaways

1. ✅ **Always report BOTH R² and RMSE**
2. ✅ **Include ground truth variance statistics**
3. ✅ **Low RMSE ≠ Good model** (depends on variance)
4. ✅ **R² < 0 means model is worse than predicting mean**
5. ✅ **Check data variance when interpreting metrics**

## 🔗 Related Files

- Main results: `../../results.md`
- Evaluation tools: `../gt_pre_eval/`
- Visualization notebook: `../../visualizations/visualizations.ipynb`

## 📧 Questions?

Refer to:
1. `VISUAL_INTERPRETATION_GUIDE.md` - Visual elements explained
2. `README.md` - Detailed documentation
3. `detailed_analysis_report.txt` - Statistical insights

---

**Generated**: November 2025  
**Purpose**: Building Height Estimation Research  
**Topic**: Metric Synergy Analysis for Regression Tasks
