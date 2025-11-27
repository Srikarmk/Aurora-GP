"""
AURORA - Step 3: Visualization Tools for Approximation Methods
Compare Exact GP, RFF, and Nyström visually
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).parent.parent
sns.set_style("whitegrid")


class ApproximationVisualizer:
    """Visualization tools for comparing approximation methods"""
    
    def __init__(self, results_dir=None):
        if results_dir is None:
            results_dir = PROJECT_ROOT / 'results' / 'approximations'
        self.results_dir = Path(results_dir)
    
    def plot_accuracy_comparison(self, save_path=None):
        """Compare accuracy metrics across all datasets"""
        
        # Load summary
        summary_file = self.results_dir / 'summary.json'
        if not summary_file.exists():
            print(f"No summary file found at {summary_file}")
            return None
        
        with open(summary_file, 'r') as f:
            results = json.load(f)
        
        # Extract data
        datasets = []
        methods = ['exact_gp', 'rff', 'nystrom']
        method_names = ['Exact GP', 'RFF', 'Nyström']
        metrics_data = {m: {'rmse': [], 'nll': [], 'ece': []} for m in methods}
        
        for dataset_name, dataset_results in results.items():
            if 'error' in dataset_results:
                continue
            
            datasets.append(dataset_name)
            
            for method in methods:
                if method in dataset_results and 'error' not in dataset_results[method]:
                    metrics_data[method]['rmse'].append(dataset_results[method]['rmse'])
                    metrics_data[method]['nll'].append(dataset_results[method]['nll'])
                    metrics_data[method]['ece'].append(dataset_results[method]['ece'])
                else:
                    metrics_data[method]['rmse'].append(np.nan)
                    metrics_data[method]['nll'].append(np.nan)
                    metrics_data[method]['ece'].append(np.nan)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        x = np.arange(len(datasets))
        width = 0.25
        
        # Plot each metric
        for idx, (metric_name, ax) in enumerate(zip(['rmse', 'nll', 'ece'], axes)):
            for i, (method, display_name) in enumerate(zip(methods, method_names)):
                values = metrics_data[method][metric_name]
                offset = (i - 1) * width
                ax.bar(x + offset, values, width, label=display_name, alpha=0.8)
            
            ax.set_xlabel('Dataset')
            ax.set_ylabel(metric_name.upper())
            ax.set_title(f'{metric_name.upper()} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(datasets, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_speedup_comparison(self, save_path=None):
        """Compare training times and speedups"""
        
        summary_file = self.results_dir / 'summary.json'
        with open(summary_file, 'r') as f:
            results = json.load(f)
        
        datasets = []
        times = {'exact_gp': [], 'rff': [], 'nystrom': []}
        
        for dataset_name, dataset_results in results.items():
            if 'error' in dataset_results:
                continue
            
            datasets.append(dataset_name)
            
            for method in ['exact_gp', 'rff', 'nystrom']:
                if method in dataset_results and 'error' not in dataset_results[method]:
                    times[method].append(dataset_results[method]['training_time'])
                else:
                    times[method].append(np.nan)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training time comparison
        x = np.arange(len(datasets))
        width = 0.25
        
        for i, (method, name) in enumerate([('exact_gp', 'Exact GP'), ('rff', 'RFF'), ('nystrom', 'Nyström')]):
            offset = (i - 1) * width
            ax1.bar(x + offset, times[method], width, label=name, alpha=0.8)
        
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Training Time (seconds)')
        ax1.set_title('Training Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_yscale('log')
        
        # Speedup comparison
        speedup_rff = [gp / rff if rff > 0 else 0 for gp, rff in zip(times['exact_gp'], times['rff'])]
        speedup_nystrom = [gp / nys if nys > 0 else 0 for gp, nys in zip(times['exact_gp'], times['nystrom'])]
        
        ax2.bar(x - width/2, speedup_rff, width, label='RFF', alpha=0.8, color='C1')
        ax2.bar(x + width/2, speedup_nystrom, width, label='Nyström', alpha=0.8, color='C2')
        ax2.axhline(1, color='r', linestyle='--', label='Exact GP baseline')
        
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('Speedup vs Exact GP')
        ax2.set_xticks(x)
        ax2.set_xticklabels(datasets, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_calibration_comparison(self, dataset_name, y_test, predictions, save_path=None):
        """Compare calibration curves for all three methods
        
        Args:
            dataset_name: Name of dataset
            y_test: True test values
            predictions: Dict with keys 'exact_gp', 'rff', 'nystrom', 
                        each containing (y_pred, y_std) tuples
        """
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        n_bins = 10
        confidence_levels = np.linspace(0.1, 0.9, n_bins)
        
        colors = {'exact_gp': 'C0', 'rff': 'C1', 'nystrom': 'C2'}
        names = {'exact_gp': 'Exact GP', 'rff': 'RFF', 'nystrom': 'Nyström'}
        
        # Plot calibration curves
        ax = axes[0]
        
        for method, (y_pred, y_std) in predictions.items():
            errors = np.abs(y_test - y_pred)
            observed_freqs = []
            
            for conf in confidence_levels:
                z_score = norm.ppf((1 + conf) / 2)
                predicted_interval = y_std * z_score
                within = (errors <= predicted_interval).astype(float).mean()
                observed_freqs.append(within)
            
            ax.plot(confidence_levels, observed_freqs, 'o-', 
                   label=names[method], color=colors[method], linewidth=2, markersize=6)
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
        
        ax.set_xlabel('Predicted Confidence')
        ax.set_ylabel('Observed Frequency')
        ax.set_title(f'{dataset_name}: Calibration Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Plot ECE comparison
        ax = axes[1]
        
        eces = {}
        for method, (y_pred, y_std) in predictions.items():
            errors = np.abs(y_test - y_pred)
            ece = 0.0
            
            for conf in confidence_levels:
                z_score = norm.ppf((1 + conf) / 2)
                predicted_interval = y_std * z_score
                within = (errors <= predicted_interval).astype(float).mean()
                ece += np.abs(within - conf) / n_bins
            
            eces[method] = ece
        
        methods_list = list(eces.keys())
        ece_values = [eces[m] for m in methods_list]
        display_names = [names[m] for m in methods_list]
        
        bars = ax.bar(display_names, ece_values, alpha=0.7,
                     color=[colors[m] for m in methods_list])
        
        ax.set_ylabel('ECE')
        ax.set_title('Expected Calibration Error')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, ece_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_accuracy_vs_speed_tradeoff(self, save_path=None):
        """Plot accuracy-speed tradeoff"""
        
        summary_file = self.results_dir / 'summary.json'
        with open(summary_file, 'r') as f:
            results = json.load(f)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        methods = ['exact_gp', 'rff', 'nystrom']
        colors = {'exact_gp': 'C0', 'rff': 'C1', 'nystrom': 'C2'}
        names = {'exact_gp': 'Exact GP', 'rff': 'RFF', 'nystrom': 'Nyström'}
        markers = {'exact_gp': 'o', 'rff': 's', 'nystrom': '^'}
        
        for dataset_name, dataset_results in results.items():
            if 'error' in dataset_results:
                continue
            
            for method in methods:
                if method in dataset_results and 'error' not in dataset_results[method]:
                    m = dataset_results[method]
                    
                    # ECE vs Time
                    ax1.scatter(m['training_time'], m['ece'], 
                              color=colors[method], marker=markers[method],
                              s=100, alpha=0.6, label=names[method] if dataset_name == list(results.keys())[0] else '')
                    
                    # RMSE vs Time
                    ax2.scatter(m['training_time'], m['rmse'],
                              color=colors[method], marker=markers[method],
                              s=100, alpha=0.6)
        
        ax1.set_xlabel('Training Time (seconds)')
        ax1.set_ylabel('ECE (lower is better)')
        ax1.set_title('Calibration vs Speed Tradeoff')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Training Time (seconds)')
        ax2.set_ylabel('RMSE (lower is better)')
        ax2.set_title('Accuracy vs Speed Tradeoff')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_all_plots(self):
        """Generate all comparison plots"""
        print("\n" + "="*60)
        print("GENERATING APPROXIMATION COMPARISON PLOTS")
        print("="*60)
        
        plot_dir = self.results_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        
        # 1. Accuracy comparison
        print("\n1. Accuracy comparison...")
        self.plot_accuracy_comparison(save_path=plot_dir / 'accuracy_comparison.png')
        
        # 2. Speed comparison
        print("2. Speed comparison...")
        self.plot_speedup_comparison(save_path=plot_dir / 'speed_comparison.png')
        
        # 3. Tradeoff plots
        print("3. Accuracy-speed tradeoff...")
        self.plot_accuracy_vs_speed_tradeoff(save_path=plot_dir / 'tradeoff.png')
        
        print(f"\n[OK] All plots saved to {plot_dir}")


if __name__ == "__main__":
    viz = ApproximationVisualizer()
    viz.generate_all_plots()