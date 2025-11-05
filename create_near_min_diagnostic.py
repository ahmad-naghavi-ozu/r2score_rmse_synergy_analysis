"""
Create a diagnostic plot for the scenario where predictions are near minimum (y_min),
causing SS_total to approach zero and R² to decline towards negative infinity.

This demonstrates the extreme case where even low RMSE can produce highly negative R²
when the ground truth has extremely low variance concentrated near the minimum value.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Output directory
OUTPUT_DIR = Path(__file__).parent / "metric_synergy_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# Generate scenario: predictions near minimum with very low variance
n_samples = 100
y_min = 45.0

# Ground truth: very low variance concentrated near minimum
y_true = np.random.normal(y_min + 0.5, 0.3, n_samples)  # Mean ≈ 45.5, std = 0.3

# Predictions: also near minimum but with slight offset/noise
y_pred = np.full(n_samples, y_min) + np.random.normal(0, 0.2, n_samples)

# Calculate metrics
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
y_mean = np.mean(y_true)
residuals = y_pred - y_true

# Calculate variance components
ss_total = np.sum((y_true - y_mean)**2)
ss_residual = np.sum((y_pred - y_true)**2)
ss_explained = ss_total - ss_residual

# Create detailed diagnostic plot
fig = plt.figure(figsize=(20, 6))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# ============================================================================
# Plot 1: Scatter with detailed annotations
# ============================================================================
ax1 = fig.add_subplot(gs[:, 0])
ax1.scatter(y_true, y_pred, alpha=0.6, s=50, c='steelblue', 
            edgecolors='black', linewidth=0.5, label='Predictions', zorder=5)

# Perfect prediction line (y=x)
plot_min = min(y_true.min(), y_pred.min()) - 0.5
plot_max = max(y_true.max(), y_pred.max()) + 0.5
ax1.plot([plot_min, plot_max], [plot_min, plot_max],
        'r--', linewidth=2.5, label='Perfect prediction (y=x)', zorder=2)

# Mean of ground truth (horizontal and vertical)
ax1.axhline(y=y_mean, color='green', linestyle=':', linewidth=2.5, 
           label=f'Mean GT (ȳ={y_mean:.2f})', alpha=0.8, zorder=1)
ax1.axvline(x=y_mean, color='green', linestyle=':', linewidth=2.5, alpha=0.8, zorder=1)

# Show residuals for sample points
sample_indices = np.linspace(0, len(y_true)-1, min(8, len(y_true)), dtype=int)
for i, idx in enumerate(sample_indices):
    color = 'orange' if i == 0 else 'coral'
    alpha = 0.8 if i == 0 else 0.4
    linewidth = 2.5 if i == 0 else 1.5
    label = 'Residuals (ŷ-y)' if i == 0 else None
    # Vertical line from point to y=x line
    ax1.plot([y_true[idx], y_true[idx]], [y_pred[idx], y_true[idx]], 
           color=color, linewidth=linewidth, alpha=alpha, zorder=4, label=label)

ax1.set_xlabel('True Values (y)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Predicted Values (ŷ)', fontsize=13, fontweight='bold')
ax1.set_title('Prediction vs Ground Truth\nNear-Min Pred. (Low Var)', 
             fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='best', framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal', adjustable='box')

# ============================================================================
# Plot 2: Residual plot
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_true, residuals, alpha=0.6, s=50, c='coral', 
            edgecolors='black', linewidth=0.5)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('True Values (y)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Residuals (ŷ - y)', fontsize=11, fontweight='bold')
ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# ============================================================================
# Plot 3: Residual histogram
# ============================================================================
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(residuals, bins=20, alpha=0.7, color='coral', edgecolor='black')
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax3.axvline(x=np.mean(residuals), color='blue', linestyle=':', linewidth=2, 
            label=f'Mean: {np.mean(residuals):.3f}')
ax3.set_xlabel('Residuals (ŷ - y)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Residual Distribution', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

# ============================================================================
# Plot 4: Statistical breakdown
# ============================================================================
ax4 = fig.add_subplot(gs[:, 2])
ax4.axis('off')

# Create text summary
stats_text = f"""
STATISTICAL BREAKDOWN
{'='*40}

METRICS:
  R² Score:     {r2:.6f}
  RMSE:         {rmse:.6f}

GROUND TRUTH STATISTICS:
  Mean (ȳ):     {y_mean:.6f}
  Std Dev:      {np.std(y_true):.6f}
  Variance:     {np.var(y_true):.6f}
  Min:          {y_true.min():.6f}
  Max:          {y_true.max():.6f}
  Range:        {y_true.max() - y_true.min():.6f}

PREDICTION STATISTICS:
  Mean:         {np.mean(y_pred):.6f}
  Std Dev:      {np.std(y_pred):.6f}
  Min:          {y_pred.min():.6f}
  Max:          {y_pred.max():.6f}

VARIANCE DECOMPOSITION:
  SS_total:     {ss_total:.2f}
    (Σ(y - ȳ)²)
  
  SS_residual:  {ss_residual:.2f}
    (Σ(ŷ - y)²)
  
  SS_explained: {ss_explained:.2f}
    (SS_total - SS_residual)

R² CALCULATION:
  R² = 1 - (SS_res / SS_tot)
  R² = 1 - ({ss_residual:.2f} / {ss_total:.2f})
  R² = 1 - {ss_residual/ss_total if ss_total > 0 else float('inf'):.6f}
  R² = {r2:.6f}

RESIDUAL STATISTICS:
  Mean Error:   {np.mean(residuals):.6f}
  Std Dev:      {np.std(residuals):.6f}
  MAE:          {np.mean(np.abs(residuals)):.6f}

INTERPRETATION:
  ✗ Model worse than mean baseline
  RMSE/Range: {rmse/(y_true.max()-y_true.min())*100 if y_true.max()-y_true.min() > 0 else 0:.1f}%

  ⚠ CRITICAL: SS_total ≈ 0!
  When data variance is extremely low,
  SS_total approaches zero, making R²
  extremely sensitive. Even tiny errors
  can cause R² → -∞!
  
  In this case:
  • Ground truth clustered near y_min
  • Very small variance (σ² = {np.var(y_true):.3f})
  • SS_total = {ss_total:.2f} ≈ 0
  • Even small SS_residual = {ss_residual:.2f}
  • Results in R² = {r2:.2f} (highly negative!)
  
  Despite low absolute RMSE ({rmse:.3f}),
  the model explains none of the tiny
  variance in the data.
"""

ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

fig.suptitle('Detailed Diagnostic Analysis: Near-Min Pred. (Low Var)', 
            fontsize=16, fontweight='bold', y=0.98)

# Save plot
output_path = OUTPUT_DIR / 'detailed_diagnostic_near_min_low_variance_neg_inf_r2.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
print(f"\nKey finding:")
print(f"  - R² Score: {r2:.6f} (highly negative!)")
print(f"  - RMSE: {rmse:.6f} (relatively low)")
print(f"  - Ground truth variance: {np.var(y_true):.6f} (extremely low)")
print(f"  - SS_total: {ss_total:.2f} ≈ 0")
print(f"  - SS_residual: {ss_residual:.2f}")
print(f"  - Ratio: SS_res/SS_tot = {ss_residual/ss_total if ss_total > 0 else float('inf'):.2f}")
print(f"\nThis demonstrates: When predictions cluster near y_min with low variance,")
print(f"SS_total → 0, making R² → -∞ even with low absolute RMSE!")

plt.show()
