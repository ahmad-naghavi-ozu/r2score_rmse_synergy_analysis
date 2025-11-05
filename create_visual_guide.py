"""
Create a simple educational diagram showing how to read the diagnostic plots.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# ============================================================================
# LEFT PLOT: Annotated example
# ============================================================================
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_aspect('equal')

# Sample data
y_mean = 5.0
points = [
    (3, 4),   # Under-prediction
    (5, 7),   # Over-prediction
    (7, 7),   # Perfect prediction (on line)
    (8, 6),   # Under-prediction
]

# Draw green lines (mean)
ax1.axhline(y=y_mean, color='green', linestyle=':', linewidth=3, alpha=0.8, zorder=1)
ax1.axvline(x=y_mean, color='green', linestyle=':', linewidth=3, alpha=0.8, zorder=1)

# Draw red line (y=x)
ax1.plot([0, 10], [0, 10], 'r--', linewidth=3, alpha=0.8, zorder=2)

# Draw points and residuals
for i, (x, y) in enumerate(points):
    # Blue point
    ax1.scatter(x, y, s=200, c='steelblue', edgecolors='black', linewidth=2, zorder=5)
    
    # Orange residual line (to y=x)
    ax1.plot([x, x], [y, x], color='orange', linewidth=3, alpha=0.8, zorder=4)
    
    # Annotate specific points
    if i == 0:  # First point - detailed annotation
        # Arrow pointing to the point
        ax1.annotate('Blue Point\n(y_true, y_pred)\n= (3, 4)', 
                    xy=(x, y), xytext=(0.5, 2),
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        
        # Arrow for residual
        ax1.annotate('Residual\n= ŷ - y\n= 4 - 3 = +1\n(over-pred)', 
                    xy=(x, (y+x)/2), xytext=(1.2, 3.8),
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', lw=2, color='orange'))
    
    elif i == 1:  # Second point
        ax1.annotate('Over-prediction\n(above y=x)', 
                    xy=(x, y), xytext=(3.5, 8.5),
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    elif i == 2:  # Third point
        ax1.annotate('Perfect!\n(on y=x line)', 
                    xy=(x, y), xytext=(8, 8.5),
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', lw=2, color='green'))

# Annotate mean line
ax1.text(0.3, y_mean + 0.3, 'Mean GT (ȳ = 5.0)\n[for SS_total]', 
        fontsize=11, fontweight='bold', color='green',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Annotate y=x line
ax1.text(8.5, 9, 'y = x\n(Perfect)', 
        fontsize=11, fontweight='bold', color='red',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# Labels
ax1.set_xlabel('Ground Truth (y)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Predicted (ŷ)', fontsize=14, fontweight='bold')
ax1.set_title('How to Read the Diagnostic Plot', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3)

# ============================================================================
# RIGHT PLOT: Formula explanation
# ============================================================================
ax2.axis('off')

explanation = """
KEY FORMULAS AND VISUAL ELEMENTS
═══════════════════════════════════════════════════════════

🟢 GREEN LINES (Mean of Ground Truth)
   Position: y = ȳ and x = ȳ
   Purpose: Reference for computing SS_total
   
   SS_total = Σ(y_i - ȳ)²
   ↓
   Measures total variance in ground truth data


🔴 RED LINE (Perfect Prediction)
   Equation: ŷ = y  (or y_pred = y_true)
   Purpose: Shows where perfect predictions lie
   
   If all points on this line → R² = 1, RMSE = 0


🟠 ORANGE LINES (Residuals)
   Definition: Vertical distance from point to red line
   Formula: residual_i = ŷ_i - y_i
   
   Length represents prediction error for each sample


🔵 BLUE POINTS (Predictions)
   Coordinates: (y_true, y_pred)
   Position relative to red line indicates quality:
   • Above red line → over-prediction
   • Below red line → under-prediction
   • On red line → perfect prediction


═══════════════════════════════════════════════════════════
CALCULATING METRICS FROM THE PLOT
═══════════════════════════════════════════════════════════

Step 1: SS_total (from green lines)
   SS_total = Σ(y_i - ȳ)²
   = Sum of squared vertical distances 
     from each TRUE value to green line

Step 2: SS_residual (from orange lines)
   SS_residual = Σ(ŷ_i - y_i)²
   = Sum of squared LENGTHS of orange lines

Step 3: R² Score
   R² = 1 - (SS_residual / SS_total)
   
   Interpretation:
   • R² = 1.0  → All orange lines are zero
   • R² = 0.0  → Orange lines as large as variance
   • R² < 0.0  → Model worse than predicting mean

Step 4: RMSE
   RMSE = √(SS_residual / n)
   RMSE = √(mean of squared orange line lengths)


═══════════════════════════════════════════════════════════
⚠️  CRITICAL INSIGHT
═══════════════════════════════════════════════════════════

Why can LOW RMSE have LOW R²?

If SS_total is SMALL (data close to green line):
  → Even small orange lines (low RMSE)
  → Can be comparable to SS_total
  → Leading to low R²!

Example:
  Data range: 49-51 (SS_total = 50)
  RMSE: 0.8 (SS_residual = 64)
  R² = 1 - (64/50) = -0.28 ⚠️ NEGATIVE!

The RMSE is small in absolute terms (0.8),
but large relative to the data variance!


═══════════════════════════════════════════════════════════
SUMMARY
═══════════════════════════════════════════════════════════

✓ Orange line length = Residual = |ŷ - y|
✓ Short orange lines relative to data spread = High R²
✓ Low RMSE + Low variance = Can still have low R²
✓ Always check BOTH metrics + ground truth variance!
"""

ax2.text(0.05, 0.95, explanation, transform=ax2.transAxes,
        fontsize=10.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
output_path = 'metric_synergy_visualizations/visual_guide_annotated.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved: {output_path}")
plt.show()
