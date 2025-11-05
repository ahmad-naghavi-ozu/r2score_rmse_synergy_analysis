"""
Script to investigate the synergy between R² score and RMSE metrics.

This script explores whether:
1. Low RMSE always implies high R² score
2. Whether it's possible to have low RMSE with low or negative R² score
3. The conditions under which these metrics diverge

Author: Research Team
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from typing import Tuple, List, Dict
import pandas as pd
import os
from pathlib import Path

# Define output directories
OUTPUT_BASE_DIR = Path(__file__).parent
PLOTS_DIR = OUTPUT_BASE_DIR / "metric_synergy_visualizations"
TABLES_DIR = OUTPUT_BASE_DIR / "metric_synergy_analysis_tables"
REPORTS_DIR = OUTPUT_BASE_DIR / "metric_synergy_reports"

# Create output directories
for directory in [PLOTS_DIR, TABLES_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Calculate R² score and RMSE for given true and predicted values.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Tuple of (r2_score, rmse)
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, rmse


def scenario_perfect_prediction(n_samples: int = 100) -> Dict:
    """Scenario 1: Perfect prediction"""
    y_true = np.linspace(0, 100, n_samples)
    y_pred = y_true.copy()
    r2, rmse = calculate_metrics(y_true, y_pred)
    
    return {
        'name': 'Perfect Prediction',
        'y_true': y_true,
        'y_pred': y_pred,
        'r2': r2,
        'rmse': rmse,
        'description': 'Perfect predictions: y_pred = y_true'
    }


def scenario_constant_offset(n_samples: int = 100, offset: float = 5.0) -> Dict:
    """Scenario 2: Constant offset (preserves correlation)"""
    y_true = np.linspace(0, 100, n_samples)
    y_pred = y_true + offset  # Constant offset
    r2, rmse = calculate_metrics(y_true, y_pred)
    
    return {
        'name': f'Constant Offset ({offset})',
        'y_true': y_true,
        'y_pred': y_pred,
        'r2': r2,
        'rmse': rmse,
        'description': f'Predictions shifted by constant {offset}'
    }


def scenario_constant_prediction(n_samples: int = 100, constant: float = 50.0) -> Dict:
    """Scenario 3: Constant prediction (worst R² possible)"""
    y_true = np.linspace(0, 100, n_samples)
    y_pred = np.full(n_samples, constant)  # Always predict the same value
    r2, rmse = calculate_metrics(y_true, y_pred)
    
    return {
        'name': f'Constant Prediction ({constant})',
        'y_true': y_true,
        'y_pred': y_pred,
        'r2': r2,
        'rmse': rmse,
        'description': f'Always predict {constant} (mean of y_true)'
    }


def scenario_small_variance_data(n_samples: int = 100, noise_std: float = 0.5) -> Dict:
    """Scenario 4: Low variance in ground truth with small prediction errors"""
    # Data with very small variance around mean
    y_true = np.random.normal(50, 1.0, n_samples)  # Small std=1.0
    y_pred = y_true + np.random.normal(0, noise_std, n_samples)
    r2, rmse = calculate_metrics(y_true, y_pred)
    
    return {
        'name': 'Low Variance Data (std=1.0)',
        'y_true': y_true,
        'y_pred': y_pred,
        'r2': r2,
        'rmse': rmse,
        'description': 'Ground truth has low variance, small prediction errors'
    }


def scenario_high_variance_data(n_samples: int = 100, noise_std: float = 5.0) -> Dict:
    """Scenario 5: High variance in ground truth with same RMSE"""
    # Data with high variance
    y_true = np.random.normal(50, 30.0, n_samples)  # Large std=30.0
    y_pred = y_true + np.random.normal(0, noise_std, n_samples)
    r2, rmse = calculate_metrics(y_true, y_pred)
    
    return {
        'name': 'High Variance Data (std=30.0)',
        'y_true': y_true,
        'y_pred': y_pred,
        'r2': r2,
        'rmse': rmse,
        'description': 'Ground truth has high variance, same absolute errors'
    }


def scenario_inverse_relationship(n_samples: int = 100) -> Dict:
    """Scenario 6: Inverse relationship (negative R²)"""
    y_true = np.linspace(0, 100, n_samples)
    y_pred = 100 - y_true + np.random.normal(0, 5, n_samples)  # Inverse + noise
    r2, rmse = calculate_metrics(y_true, y_pred)
    
    return {
        'name': 'Inverse Relationship',
        'y_true': y_true,
        'y_pred': y_pred,
        'r2': r2,
        'rmse': rmse,
        'description': 'Predictions inversely correlated with truth'
    }


def scenario_near_mean_prediction_low_variance(n_samples: int = 100) -> Dict:
    """Scenario 7: Predict near mean for low variance data (low RMSE, low R²)"""
    # This is the KEY scenario: low variance data, predict near mean
    y_true = np.random.normal(50, 2.0, n_samples)  # Very small variance (std=2)
    y_pred = np.full(n_samples, 50.0) + np.random.normal(0, 0.5, n_samples)
    r2, rmse = calculate_metrics(y_true, y_pred)
    
    return {
        'name': 'Near-Mean Pred. (Low Var)',
        'y_true': y_true,
        'y_pred': y_pred,
        'r2': r2,
        'rmse': rmse,
        'description': 'Low variance data, predict near mean → LOW RMSE, LOW R²'
    }


def scenario_near_min_prediction_low_variance(n_samples: int = 100) -> Dict:
    """Scenario 8: Predict near minimum for low variance data (SS_tot→0, R²→-∞)"""
    # Critical scenario: low variance data concentrated near minimum
    # When data has low variance near min, predicting near min makes SS_total ≈ 0
    # Even small errors make SS_residual > SS_total, causing R² → -∞
    y_min = 45.0
    y_true = np.random.normal(y_min + 0.5, 0.3, n_samples)  # Very small variance near min (std=0.3)
    y_pred = np.full(n_samples, y_min) + np.random.normal(0, 0.2, n_samples)  # Predict near min
    r2, rmse = calculate_metrics(y_true, y_pred)
    
    return {
        'name': 'Near-Min Pred. (Low Var)',
        'y_true': y_true,
        'y_pred': y_pred,
        'r2': r2,
        'rmse': rmse,
        'description': 'Very low variance data near min, predict near min → SS_tot≈0, R²→-∞'
    }


def scenario_random_predictions(n_samples: int = 100) -> Dict:
    """Scenario 9: Completely random predictions"""
    y_true = np.linspace(0, 100, n_samples)
    y_pred = np.random.uniform(0, 100, n_samples)  # Random predictions
    r2, rmse = calculate_metrics(y_true, y_pred)
    
    return {
        'name': 'Random Predictions',
        'y_true': y_true,
        'y_pred': y_pred,
        'r2': r2,
        'rmse': rmse,
        'description': 'Completely random predictions'
    }


def visualize_scenario(scenario: Dict, ax: plt.Axes, show_details: bool = False) -> None:
    """Visualize a single scenario."""
    y_true = scenario['y_true']
    y_pred = scenario['y_pred']
    
    # Calculate statistics
    y_mean = np.mean(y_true)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, zorder=3, label='Predictions')
    
    # Perfect prediction line (y=x)
    plot_min = min(y_true.min(), y_pred.min())
    plot_max = max(y_true.max(), y_pred.max())
    ax.plot([plot_min, plot_max], [plot_min, plot_max],
            'r--', linewidth=2, label='Perfect prediction (y=x)', zorder=2)
    
    # Always show mean lines for visual reference
    ax.axhline(y=y_mean, color='green', linestyle=':', linewidth=1.5, 
               label=f'Mean GT (ȳ={y_mean:.1f})', alpha=0.6, zorder=1)
    ax.axvline(x=y_mean, color='green', linestyle=':', linewidth=1.5, 
               alpha=0.6, zorder=1)
    
    if show_details:
        # Show residuals for a few points
        sample_indices = np.linspace(0, len(y_true)-1, min(5, len(y_true)), dtype=int)
        for idx in sample_indices:
            # Vertical line from point to y=x line (residual visualization)
            ax.plot([y_true[idx], y_true[idx]], [y_pred[idx], y_true[idx]], 
                   'orange', linewidth=1.5, alpha=0.6, zorder=4)
    
    ax.set_xlabel('True Values (y_true)', fontsize=10)
    ax.set_ylabel('Predicted Values (y_pred)', fontsize=10)
    ax.set_title(f"{scenario['name']}\nR²={scenario['r2']:.3f}, RMSE={scenario['rmse']:.3f}",
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')


def create_comprehensive_visualization(scenarios: List[Dict]) -> None:
    """Create comprehensive visualization of all scenarios."""
    n_scenarios = len(scenarios)
    n_cols = 3
    n_rows = (n_scenarios + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    axes = axes.flatten() if n_scenarios > 1 else [axes]
    
    for idx, scenario in enumerate(scenarios):
        visualize_scenario(scenario, axes[idx], show_details=False)
    
    # Hide unused subplots
    for idx in range(n_scenarios, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = PLOTS_DIR / 'scenario_comparison_r2_vs_rmse.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)  # Close instead of show for non-interactive mode


def create_detailed_diagnostic_plot(scenario: Dict) -> plt.Figure:
    """
    Create a detailed diagnostic plot for a single scenario showing:
    1. Scatter plot with mean GT line and residuals
    2. Residual plot
    3. Statistical breakdown
    """
    fig = plt.figure(figsize=(20, 6))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    y_true = scenario['y_true']
    y_pred = scenario['y_pred']
    y_mean = np.mean(y_true)
    residuals = y_pred - y_true
    
    # Calculate variance components
    ss_total = np.sum((y_true - y_mean)**2)
    ss_residual = np.sum((y_pred - y_true)**2)
    ss_explained = ss_total - ss_residual
    
    # Plot 1: Scatter with detailed annotations
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.scatter(y_true, y_pred, alpha=0.6, s=50, c='steelblue', 
                edgecolors='black', linewidth=0.5, label='Predictions', zorder=5)
    
    # Perfect prediction line (y=x)
    plot_min = min(y_true.min(), y_pred.min())
    plot_max = max(y_true.max(), y_pred.max())
    ax1.plot([plot_min, plot_max], [plot_min, plot_max],
            'r--', linewidth=2.5, label='Perfect prediction (y=x)', zorder=2)
    
    # Mean of ground truth (horizontal and vertical)
    ax1.axhline(y=y_mean, color='green', linestyle=':', linewidth=2.5, 
               label=f'Mean GT (ȳ={y_mean:.2f})', alpha=0.8, zorder=1)
    ax1.axvline(x=y_mean, color='green', linestyle=':', linewidth=2.5, alpha=0.8, zorder=1)
    
    # Show residuals for sample points (vertical lines showing prediction errors)
    # Use more samples for better visibility
    sample_indices = np.linspace(0, len(y_true)-1, min(15, len(y_true)), dtype=int)
    for i, idx in enumerate(sample_indices):
        color = 'orange' if i == 0 else 'coral'
        alpha = 0.9 if i == 0 else 0.5
        linewidth = 2.5 if i == 0 else 1.8
        label = 'Residuals (ŷ-y)' if i == 0 else None
        # Vertical line from point to y=x line (showing residual)
        ax1.plot([y_true[idx], y_true[idx]], [y_pred[idx], y_true[idx]], 
               color=color, linewidth=linewidth, alpha=alpha, zorder=4, label=label)
    
    # Show distance from true values to mean (horizontal lines showing SS_total components)
    for i, idx in enumerate(sample_indices):
        color = 'purple' if i == 0 else 'mediumpurple'
        alpha = 0.9 if i == 0 else 0.55  # Increased alpha for better visibility
        linewidth = 2.5 if i == 0 else 1.8
        label = 'y - ȳ (for SS_total)' if i == 0 else None
        # Horizontal line from true value to mean line (showing deviation from mean)
        ax1.plot([y_true[idx], y_mean], [y_pred[idx], y_pred[idx]], 
               color=color, linewidth=linewidth, alpha=alpha, zorder=3, label=label)
    
    ax1.set_xlabel('True Values (y)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Predicted Values (ŷ)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Prediction vs Ground Truth\n{scenario["name"]}', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Plot 2: Residual plot
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y_true, residuals, alpha=0.6, s=50, c='coral', 
                edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('True Values (y)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Residuals (ŷ - y)', fontsize=11, fontweight='bold')
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residual histogram
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
    
    # Plot 4: Statistical breakdown
    ax4 = fig.add_subplot(gs[:, 2])
    ax4.axis('off')
    
    # Create text summary
    stats_text = f"""
STATISTICAL BREAKDOWN
{'='*40}

METRICS:
  R² Score:     {scenario['r2']:.6f}
  RMSE:         {scenario['rmse']:.6f}

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
  R² = 1 - {ss_residual/ss_total if ss_total > 0 else 0:.6f}
  R² = {scenario['r2']:.6f}

RESIDUAL STATISTICS:
  Mean Error:   {np.mean(residuals):.6f}
  Std Dev:      {np.std(residuals):.6f}
  MAE:          {np.mean(np.abs(residuals)):.6f}

INTERPRETATION:
"""
    
    # Add interpretation
    if scenario['r2'] > 0.9:
        stats_text += "  ✓ Excellent model performance"
    elif scenario['r2'] > 0.7:
        stats_text += "  ✓ Good model performance"
    elif scenario['r2'] > 0.5:
        stats_text += "  ⚠ Moderate model performance"
    elif scenario['r2'] > 0:
        stats_text += "  ⚠ Poor model performance"
    else:
        stats_text += "  ✗ Model worse than mean baseline"
    
    stats_text += f"\n  RMSE/Range: {scenario['rmse']/(y_true.max()-y_true.min())*100 if y_true.max()-y_true.min() > 0 else 0:.1f}%"
    
    if np.var(y_true) < 10 and scenario['r2'] < 0.5:
        stats_text += "\n\n  ⚠ LOW VARIANCE DATA!\n  Low R² despite potentially low RMSE\n  due to small ground truth variance."
    
    if np.var(y_true) < 1 and ss_total < 10:
        stats_text += "\n\n  ⚠ CRITICAL: SS_total ≈ 0!\n  When data variance is extremely low,\n  SS_total approaches zero, making R²\n  extremely sensitive. Even tiny errors\n  can cause R² → -∞!"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle(f'Detailed Diagnostic Analysis: {scenario["name"]}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    return fig


def create_all_detailed_diagnostics(scenarios: List[Dict]) -> None:
    """Create detailed diagnostic plots for key scenarios."""
    print("\nGenerating detailed diagnostic plots...")
    
    # Select key scenarios to visualize in detail
    key_scenarios = [
        ('Perfect Prediction', 'perfect_prediction'),
        ('High Variance Data (std=30.0)', 'high_variance'),
        ('Near-Mean Pred. (Low Var)', 'low_variance_low_r2'),
        ('Near-Min Pred. (Low Var)', 'near_min_low_variance_neg_inf_r2'),
        ('Inverse Relationship', 'inverse_relationship'),
    ]
    
    for scenario in scenarios:
        for name, file_suffix in key_scenarios:
            if scenario['name'] == name:
                fig = create_detailed_diagnostic_plot(scenario)
                output_path = PLOTS_DIR / f'detailed_diagnostic_{file_suffix}.png'
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {output_path}")
                plt.close(fig)


def create_rmse_r2_heatmap(n_points: int = 50) -> None:
    """
    Create a heatmap showing R² vs RMSE relationship for different data characteristics.
    """
    variance_range = np.logspace(-1, 2, n_points)  # 0.1 to 100
    noise_range = np.logspace(-1, 1.5, n_points)   # 0.1 to ~31
    
    r2_grid = np.zeros((n_points, n_points))
    rmse_grid = np.zeros((n_points, n_points))
    
    print("Generating R² vs RMSE heatmap...")
    for i, var in enumerate(variance_range):
        for j, noise in enumerate(noise_range):
            # Generate data with specific variance and noise
            y_true = np.random.normal(50, np.sqrt(var), 100)
            y_pred = y_true + np.random.normal(0, noise, 100)
            r2_grid[i, j] = r2_score(y_true, y_pred)
            rmse_grid[i, j] = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: R² heatmap
    im1 = ax1.contourf(noise_range, variance_range, r2_grid, levels=20, cmap='RdYlGn')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Prediction Noise (σ)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Ground Truth Variance (σ²)', fontsize=12, fontweight='bold')
    ax1.set_title('R² Score Heatmap\nHow R² varies with data variance and prediction noise', 
                  fontsize=13, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1, label='R² Score')
    ax1.grid(True, alpha=0.3)
    
    # Add interpretation text
    ax1.text(0.02, 0.98, 'High variance + Low noise → High R²\nLow variance + High noise → Low/Negative R²', 
             transform=ax1.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: RMSE heatmap
    im2 = ax2.contourf(noise_range, variance_range, rmse_grid, levels=20, cmap='YlOrRd')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Prediction Noise (σ)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Ground Truth Variance (σ²)', fontsize=12, fontweight='bold')
    ax2.set_title('RMSE Heatmap\nHow RMSE varies with data variance and prediction noise', 
                  fontsize=13, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2, label='RMSE')
    ax2.grid(True, alpha=0.3)
    
    # Add interpretation text
    ax2.text(0.02, 0.98, 'RMSE ≈ Prediction Noise (σ)\nIndependent of ground truth variance!', 
             transform=ax2.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Add overall figure title
    fig.suptitle('Impact of Ground Truth Variance and Prediction Noise on Metrics\n' + 
                 'Key Insight: RMSE depends only on noise, but R² depends on BOTH variance and noise',
                 fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    output_path = PLOTS_DIR / 'variance_noise_impact_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    print(f"\nHeatmap Interpretation:")
    print(f"  LEFT (R² Heatmap): Shows R² is sensitive to BOTH variance and noise")
    print(f"    - Green (high R²): High variance OR low noise")
    print(f"    - Red (low R²): Low variance AND high noise")
    print(f"  RIGHT (RMSE Heatmap): Shows RMSE depends mainly on prediction noise")
    print(f"    - RMSE ≈ prediction noise (horizontal bands)")
    print(f"    - Ground truth variance has minimal effect on RMSE")
    plt.close(fig)  # Close instead of show for non-interactive mode


def create_summary_table(scenarios: List[Dict]) -> pd.DataFrame:
    """Create a summary table of all scenarios."""
    data = []
    for scenario in scenarios:
        data.append({
            'Scenario': scenario['name'],
            'R² Score': scenario['r2'],
            'RMSE': scenario['rmse'],
            'Description': scenario['description']
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('RMSE')
    
    print("\n" + "="*100)
    print("SUMMARY TABLE: R² Score vs RMSE Analysis")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")
    
    # Save to CSV
    csv_path = TABLES_DIR / 'scenarios_summary_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}\n")
    
    return df


def analyze_key_insights(scenarios: List[Dict]) -> None:
    """Analyze and print key insights about R² and RMSE relationship."""
    # Prepare output for both console and file
    report_lines = []
    
    def log(text):
        """Print to console and save to report."""
        print(text)
        report_lines.append(text)
    
    log("\n" + "="*100)
    log("KEY INSIGHTS: Can Low RMSE Coexist with Low/Negative R²?")
    log("="*100 + "\n")
    
    log("1. MATHEMATICAL RELATIONSHIP:")
    log("   - RMSE measures absolute prediction error (scale-dependent)")
    log("   - R² measures explained variance (scale-independent)")
    log("   - R² = 1 - (SS_res / SS_tot) where SS_tot = variance of y_true")
    log("")
    
    log("2. CRITICAL FINDING: YES, Low RMSE can have Low R²!")
    log("   Condition: When ground truth has LOW VARIANCE")
    log("")
    
    # Find scenarios with low RMSE
    df = pd.DataFrame([{
        'name': s['name'],
        'rmse': s['rmse'],
        'r2': s['r2'],
        'var_true': np.var(s['y_true'])
    } for s in scenarios])
    
    log("3. EXAMPLES FROM SCENARIOS:")
    log("-" * 100)
    
    # Case 1: Low RMSE, High R²
    high_var_scenarios = df[df['var_true'] > 100].sort_values('rmse')
    if len(high_var_scenarios) > 0:
        example = high_var_scenarios.iloc[0]
        log(f"   ✓ High Variance Data ({example['name']}):")
        log(f"     - Ground truth variance: {example['var_true']:.2f}")
        log(f"     - RMSE: {example['rmse']:.3f} (low)")
        log(f"     - R²: {example['r2']:.3f} (high)")
        log(f"     → Low RMSE + High R² ✓")
        log("")
    
    # Case 2: Low RMSE, Low R²
    low_var_scenarios = df[df['var_true'] < 10].sort_values('rmse')
    if len(low_var_scenarios) > 0:
        example = low_var_scenarios.iloc[0]
        log(f"   ⚠ Low Variance Data ({example['name']}):")
        log(f"     - Ground truth variance: {example['var_true']:.2f}")
        log(f"     - RMSE: {example['rmse']:.3f} (low)")
        log(f"     - R²: {example['r2']:.3f} (low!)")
        log(f"     → Low RMSE + Low R² ⚠ POSSIBLE!")
        log("")
    
    log("4. WHY THIS HAPPENS:")
    log("   - RMSE of 2.0 on data ranging 0-100 → excellent (2% error)")
    log("   - RMSE of 2.0 on data ranging 50-52 → poor (100% of variance)")
    log("   - R² accounts for this by normalizing against variance")
    log("")
    
    log("5. PRACTICAL IMPLICATIONS:")
    log("   - Always report BOTH metrics")
    log("   - Low RMSE alone doesn't guarantee good model")
    log("   - If R² is low despite low RMSE → model may not capture variance")
    log("   - Check ground truth variance when interpreting metrics")
    log("")
    
    log("6. EXTREME CASE - Negative R²:")
    neg_r2_scenarios = [s for s in scenarios if s['r2'] < 0]
    if neg_r2_scenarios:
        for scenario in neg_r2_scenarios:
            log(f"   ⚠ {scenario['name']}: R²={scenario['r2']:.3f}, RMSE={scenario['rmse']:.3f}")
            log(f"     → Model performs WORSE than simply predicting the mean!")
    log("")
    
    log("="*100 + "\n")
    
    # Save report to file
    report_path = REPORTS_DIR / 'detailed_analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"Saved detailed analysis report: {report_path}\n")


def main():
    """Main execution function."""
    print("="*100)
    print("INVESTIGATING R² SCORE AND RMSE SYNERGY")
    print("="*100)
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate all scenarios
    print("Generating scenarios...")
    scenarios = [
        scenario_perfect_prediction(),
        scenario_constant_offset(offset=5.0),
        scenario_high_variance_data(noise_std=5.0),
        scenario_small_variance_data(noise_std=0.5),
        scenario_near_mean_prediction_low_variance(),
        scenario_near_min_prediction_low_variance(),
        scenario_constant_prediction(constant=50.0),
        scenario_inverse_relationship(),
        scenario_random_predictions(),
    ]
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_comprehensive_visualization(scenarios)
    
    # Create detailed diagnostic plots for key scenarios
    create_all_detailed_diagnostics(scenarios)
    
    # Create summary table
    df_summary = create_summary_table(scenarios)
    
    # Create heatmap
    create_rmse_r2_heatmap(n_points=50)
    
    # Analyze insights
    analyze_key_insights(scenarios)
    
    # Print final summary
    print("="*100)
    print("ANALYSIS COMPLETE!")
    print("="*100)
    print(f"\n📊 Visualizations saved to:")
    print(f"   → {PLOTS_DIR.relative_to(OUTPUT_BASE_DIR)}/")
    print(f"      - scenario_comparison_r2_vs_rmse.png (overview)")
    print(f"      - variance_noise_impact_heatmap.png (heatmaps)")
    print(f"      - detailed_diagnostic_*.png (4 detailed plots)")
    print(f"\n📋 Tables saved to:")
    print(f"   → {TABLES_DIR.relative_to(OUTPUT_BASE_DIR)}/")
    print(f"      - scenarios_summary_metrics.csv")
    print(f"\n📄 Reports saved to:")
    print(f"   → {REPORTS_DIR.relative_to(OUTPUT_BASE_DIR)}/")
    print(f"      - detailed_analysis_report.txt")
    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    main()
