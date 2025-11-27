import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent
sns.set_style("whitegrid")


class GPVisualizer:
    """Visualization tools for GP baseline results"""
    
    def __init__(self, results_dir=None):
        if results_dir is None:
            results_dir = PROJECT_ROOT / 'results' / 'baseline_gp'
        self.results_dir = Path(results_dir)
    
    def plot_predictions_vs_true(self, dataset_name, X_test, y_test, y_pred, y_std, 
                                  save_path=None):
        """Plot predicted vs true values with uncertainty bands"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 1. Predictions vs True
        ax = axes[0]
        ax.scatter(y_test, y_pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val, max_val = y_test.min(), y_test.max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{dataset_name}: Predictions vs True')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Residuals vs Predicted
        ax = axes[1]
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5, s=20)
        ax.axhline(0, color='r', linestyle='--', lw=2)
        
        # Add ±2σ bands
        ax.fill_between(sorted(y_pred), -2*y_std[np.argsort(y_pred)], 
                        2*y_std[np.argsort(y_pred)], alpha=0.2, color='blue',
                        label='±2σ predicted')
        
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals with Uncertainty')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Uncertainty vs Absolute Error
        ax = axes[2]
        abs_errors = np.abs(residuals)
        ax.scatter(y_std, abs_errors, alpha=0.5, s=20)
        
        # Ideal calibration line (error should match std)
        max_std = y_std.max()
        ax.plot([0, max_std], [0, max_std], 'r--', lw=2, label='Perfect calibration')
        
        ax.set_xlabel('Predicted Std Dev')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Calibration: Uncertainty vs Error')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Saved to {save_path}")
            plt.close(fig)  # Close figure to free memory
        
        return fig
    
    def plot_calibration_curve(self, y_test, y_pred, y_std, dataset_name, 
                               n_bins=10, save_path=None):
        """Plot calibration curve showing predicted vs observed confidence"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Compute calibration for different confidence levels
        confidence_levels = np.linspace(0.1, 0.9, n_bins)
        observed_frequencies = []
        
        errors = np.abs(y_test - y_pred)
        
        for conf in confidence_levels:
            # Z-score for this confidence level
            from scipy.stats import norm
            z_score = norm.ppf((1 + conf) / 2)
            predicted_interval = y_std * z_score
            
            # Fraction within interval
            within = (errors <= predicted_interval).astype(float).mean()
            observed_frequencies.append(within)
        
        # 1. Calibration curve
        ax = axes[0]
        ax.plot(confidence_levels, observed_frequencies, 'o-', lw=2, markersize=8,
               label='GP Calibration')
        ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Calibration')
        
        ax.set_xlabel('Predicted Confidence')
        ax.set_ylabel('Observed Frequency')
        ax.set_title(f'{dataset_name}: Calibration Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # 2. Calibration error
        ax = axes[1]
        calibration_errors = np.abs(np.array(observed_frequencies) - confidence_levels)
        ax.bar(confidence_levels, calibration_errors, width=0.08, alpha=0.7)
        ax.axhline(0, color='r', linestyle='--', lw=2)
        
        ax.set_xlabel('Confidence Level')
        ax.set_ylabel('|Observed - Expected|')
        ax.set_title('Calibration Error by Confidence Level')
        ax.grid(True, alpha=0.3)
        
        # Show ECE
        ece = calibration_errors.mean()
        ax.text(0.5, max(calibration_errors)*0.9, f'ECE = {ece:.4f}',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               ha='center', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Saved to {save_path}")
            plt.close(fig)  # Close figure to free memory
        
        return fig
    
    def plot_uncertainty_map_2d(self, gp_model, X_train, X_test, y_test, 
                                dataset_name, save_path=None):
        """Plot 2D uncertainty map (only works for 2D datasets)"""
        if X_train.shape[1] != 2:
            print(f"[WARN] Uncertainty map only available for 2D data, got {X_train.shape[1]}D")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Create grid
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Get predictions on grid
        grid_mean, grid_std = gp_model.predict(grid_points, return_std=True)
        grid_mean = grid_mean.reshape(xx.shape)
        grid_std = grid_std.reshape(xx.shape)
        
        # 1. Mean predictions
        ax = axes[0]
        c = ax.contourf(xx, yy, grid_mean, levels=20, cmap='viridis')
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=20, 
                  edgecolors='black', linewidths=0.5, cmap='viridis')
        plt.colorbar(c, ax=ax, label='Predicted Mean')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'{dataset_name}: Mean Predictions')
        
        # 2. Uncertainty (std dev)
        ax = axes[1]
        c = ax.contourf(xx, yy, grid_std, levels=20, cmap='Reds')
        ax.scatter(X_train[:, 0], X_train[:, 1], c='black', s=10, alpha=0.3,
                  label='Training data')
        plt.colorbar(c, ax=ax, label='Std Dev')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Uncertainty Map')
        ax.legend()
        
        # 3. Combined: uncertainty with data density
        ax = axes[2]
        c = ax.contourf(xx, yy, grid_std, levels=20, cmap='Reds', alpha=0.6)
        # Plot training points with size based on local density
        ax.scatter(X_train[:, 0], X_train[:, 1], c='blue', s=20, alpha=0.4,
                  label='Dense regions')
        ax.scatter(X_test[:, 0], X_test[:, 1], c='green', s=10, alpha=0.3,
                  label='Test points')
        plt.colorbar(c, ax=ax, label='Uncertainty')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Regions for AURORA')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Saved to {save_path}")
            plt.close(fig)  # Close figure to free memory
        
        return fig
    
    def plot_results_comparison(self, save_path=None):
        """Compare results across all datasets"""
        # Load all results
        summary_path = self.results_dir / 'summary.json'
        if not summary_path.exists():
            print(f"[WARN] No summary file found at {summary_path}")
            return None
        
        with open(summary_path, 'r') as f:
            results = json.load(f)
        
        # Extract data
        datasets = list(results.keys())
        metrics = {
            'RMSE': [results[d]['rmse'] for d in datasets],
            'NLL': [results[d]['nll'] for d in datasets],
            'ECE': [results[d]['ece'] for d in datasets],
            'Training Time (s)': [results[d]['training_time'] for d in datasets]
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx]
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(datasets)))
            bars = ax.bar(range(len(datasets)), values, color=colors, alpha=0.7)
            
            ax.set_xticks(range(len(datasets)))
            ax.set_xticklabels(datasets, rotation=45, ha='right')
            ax.set_ylabel(metric_name)
            ax.set_title(f'Exact GP: {metric_name}')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] Saved to {save_path}")
            plt.close(fig)  # Close figure to free memory
        
        return fig
    
    def generate_report(self, dataset_name):
        """Generate comprehensive visualization report for a dataset"""
        print(f"\n{'='*60}")
        print(f"GENERATING VISUALIZATION REPORT: {dataset_name}")
        print(f"{'='*60}")
        
        # Load model and data
        model_dir = self.results_dir / dataset_name
        
        # Check if files exist
        if not (model_dir / 'metrics.json').exists():
            print(f"[WARN] No results found for {dataset_name}")
            return
        
        # Load data - resolve path relative to project root
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        data_path = project_root / 'data' / f'{dataset_name}.npz'
        data = np.load(data_path, allow_pickle=True)
        X, y = data['X'], data['y']
        
        # Split (same as training)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Load GP model
        from gp_baseline import GaussianProcessBaseline
        gp = GaussianProcessBaseline()
        gp.load(model_dir)
        
        # Get predictions
        y_pred, y_std = gp.predict(X_test, return_std=True)
        
        # Create visualizations directory
        viz_dir = model_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Generate plots
        print("Generating plots...")
        
        # 1. Predictions
        self.plot_predictions_vs_true(
            dataset_name, X_test, y_test, y_pred, y_std,
            save_path=viz_dir / 'predictions.png'
        )
        
        # 2. Calibration
        self.plot_calibration_curve(
            y_test, y_pred, y_std, dataset_name,
            save_path=viz_dir / 'calibration.png'
        )
        
        # 3. Uncertainty map (if 2D)
        if X.shape[1] == 2:
            self.plot_uncertainty_map_2d(
                gp, X_train, X_test, y_test, dataset_name,
                save_path=viz_dir / 'uncertainty_map.png'
            )
        
        print(f"[OK] Report generated in {viz_dir}")


def visualize_all_datasets(results_dir=None):
    """Generate visualizations for all datasets"""
    if results_dir is None:
        results_dir = PROJECT_ROOT / 'results' / 'baseline_gp'
    results_dir = Path(results_dir)
    
    visualizer = GPVisualizer(results_dir)
    
    # Get all dataset directories
    datasets = [d.name for d in results_dir.iterdir() 
               if d.is_dir() and (d / 'metrics.json').exists()]
    
    print(f"\nFound {len(datasets)} datasets to visualize")
    
    # Generate individual reports
    for dataset in datasets:
        visualizer.generate_report(dataset)
    
    # Generate comparison plot
    print("\nGenerating comparison plot...")
    visualizer.plot_results_comparison(
        save_path=results_dir / 'results_comparison.png'
    )
    
    print("\n[OK] All visualizations complete!")


if __name__ == "__main__":
    visualize_all_datasets()