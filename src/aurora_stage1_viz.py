"""
AURORA - Stage 1: Visualization Tools
Visualize uncertainty maps, density, importance scores, and regions
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Get project root (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent

sns.set_style("whitegrid")


class Stage1Visualizer:
    """Visualization tools for AURORA Stage 1"""
    
    def __init__(self, results_dir=None):
        if results_dir is None:
            results_dir = PROJECT_ROOT / 'results' / 'region_identification'
        self.results_dir = Path(results_dir)
    
    def plot_2d_regions(self, X_train, importance_scores, 
                       high_threshold, low_threshold,
                       dataset_name, save_path=None):
        """
        Plot importance scores and regions for 2D data
        
        Args:
            X_train: Training data (n_samples, 2)
            importance_scores: Importance scores (n_samples,)
            high_threshold: High importance threshold
            low_threshold: Low importance threshold
            dataset_name: Name of dataset
            save_path: Path to save figure
        """
        if X_train.shape[1] != 2:
            print(f"Can only visualize 2D data, got {X_train.shape[1]}D")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Importance scores (continuous)
        ax = axes[0]
        scatter = ax.scatter(X_train[:, 0], X_train[:, 1], 
                           c=importance_scores, cmap='RdYlBu_r',
                           s=30, alpha=0.6, edgecolors='none')
        plt.colorbar(scatter, ax=ax, label='Importance Score')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'{dataset_name}: Importance Scores')
        ax.grid(True, alpha=0.3)
        
        # 2. Regions (discrete)
        ax = axes[1]
        regions = np.zeros(len(importance_scores))
        regions[importance_scores >= high_threshold] = 2  # High
        regions[importance_scores <= low_threshold] = 0   # Low
        regions[(importance_scores > low_threshold) & 
                (importance_scores < high_threshold)] = 1  # Medium
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
        labels = ['Low Importance', 'Medium Importance', 'High Importance']
        
        for region_id, color, label in zip([0, 1, 2], colors, labels):
            mask = regions == region_id
            ax.scatter(X_train[mask, 0], X_train[mask, 1],
                      c=color, label=label, s=30, alpha=0.6, edgecolors='none')
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'{dataset_name}: Identified Regions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Histogram of importance scores
        ax = axes[2]
        ax.hist(importance_scores, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(high_threshold, color='r', linestyle='--', linewidth=2, 
                  label=f'High threshold ({high_threshold:.3f})')
        ax.axvline(low_threshold, color='g', linestyle='--', linewidth=2,
                  label=f'Low threshold ({low_threshold:.3f})')
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Importance Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_components(self, X_train, uncertainty_map, density_map,
                       importance_scores, dataset_name, save_path=None):
        """
        Plot the components that make up importance scores
        
        Args:
            X_train: Training data (n_samples, 2)
            uncertainty_map: GP uncertainties (n_samples,)
            density_map: Density scores (n_samples,)
            importance_scores: Final importance scores (n_samples,)
            dataset_name: Name of dataset
            save_path: Path to save figure
        """
        if X_train.shape[1] != 2:
            print(f"Can only visualize 2D data, got {X_train.shape[1]}D")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Normalize for visualization
        uncertainty_norm = (uncertainty_map - uncertainty_map.min()) / \
                          (uncertainty_map.max() - uncertainty_map.min() + 1e-10)
        
        # 1. Uncertainty map
        ax = axes[0, 0]
        scatter = ax.scatter(X_train[:, 0], X_train[:, 1],
                           c=uncertainty_norm, cmap='Reds',
                           s=30, alpha=0.6, edgecolors='none')
        plt.colorbar(scatter, ax=ax, label='Normalized Uncertainty')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('GP Uncertainty Map')
        ax.grid(True, alpha=0.3)
        
        # 2. Density map
        ax = axes[0, 1]
        scatter = ax.scatter(X_train[:, 0], X_train[:, 1],
                           c=density_map, cmap='Blues',
                           s=30, alpha=0.6, edgecolors='none')
        plt.colorbar(scatter, ax=ax, label='Density Score (inverse)')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Data Density Map')
        ax.grid(True, alpha=0.3)
        
        # 3. Combined importance
        ax = axes[1, 0]
        scatter = ax.scatter(X_train[:, 0], X_train[:, 1],
                           c=importance_scores, cmap='RdYlBu_r',
                           s=30, alpha=0.6, edgecolors='none')
        plt.colorbar(scatter, ax=ax, label='Importance Score')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Combined Importance Score')
        ax.grid(True, alpha=0.3)
        
        # 4. Correlation plot
        ax = axes[1, 1]
        ax.scatter(uncertainty_norm, density_map, alpha=0.3, s=20)
        ax.set_xlabel('Normalized Uncertainty')
        ax.set_ylabel('Density Score')
        ax.set_title('Uncertainty vs Density')
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = np.corrcoef(uncertainty_norm, density_map)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'{dataset_name}: Importance Score Components', fontsize=14, y=1.00)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_region_summary(self, save_path=None):
        """Plot summary of regions across all datasets"""
        
        summary_file = self.results_dir / 'summary.json'
        if not summary_file.exists():
            print(f"No summary file found at {summary_file}")
            return None
        
        with open(summary_file, 'r') as f:
            results = json.load(f)
        
        # Extract data
        datasets = []
        pct_high = []
        pct_medium = []
        pct_low = []
        
        for name, stats in results.items():
            if 'error' not in stats:
                datasets.append(name)
                pct_high.append(stats['pct_high'])
                pct_medium.append(stats['pct_medium'])
                pct_low.append(stats['pct_low'])
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(datasets))
        width = 0.6
        
        p1 = ax.bar(x, pct_low, width, label='Low Importance', color='#2ecc71', alpha=0.8)
        p2 = ax.bar(x, pct_medium, width, bottom=pct_low, label='Medium Importance', 
                   color='#f39c12', alpha=0.8)
        p3 = ax.bar(x, pct_high, width, 
                   bottom=np.array(pct_low) + np.array(pct_medium),
                   label='High Importance', color='#e74c3c', alpha=0.8)
        
        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('Dataset')
        ax.set_title('Region Distribution Across Datasets')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for i, (d, l, m, h) in enumerate(zip(datasets, pct_low, pct_medium, pct_high)):
            if l > 5:  # Only label if > 5%
                ax.text(i, l/2, f'{l:.0f}%', ha='center', va='center', fontsize=9)
            if m > 5:
                ax.text(i, l + m/2, f'{m:.0f}%', ha='center', va='center', fontsize=9)
            if h > 5:
                ax.text(i, l + m + h/2, f'{h:.0f}%', ha='center', va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_importance_statistics(self, save_path=None):
        """Plot importance score statistics across datasets"""
        
        summary_file = self.results_dir / 'summary.json'
        with open(summary_file, 'r') as f:
            results = json.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        datasets = []
        importance_means = []
        importance_stds = []
        uncertainty_means = []
        density_means = []
        
        for name, stats in results.items():
            if 'error' not in stats:
                datasets.append(name)
                importance_means.append(stats['importance_mean'])
                importance_stds.append(stats['importance_std'])
                uncertainty_means.append(stats['uncertainty_mean'])
                density_means.append(stats['density_mean'])
        
        x = np.arange(len(datasets))
        width = 0.35
        
        # Plot 1: Mean importance with error bars
        ax = axes[0]
        ax.bar(x, importance_means, width, yerr=importance_stds, 
              alpha=0.7, capsize=5, color='steelblue')
        ax.set_ylabel('Importance Score')
        ax.set_xlabel('Dataset')
        ax.set_title('Mean Importance Score ± Std')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Uncertainty vs Density contribution
        ax = axes[1]
        ax.bar(x - width/2, uncertainty_means, width, label='Uncertainty', 
              alpha=0.7, color='coral')
        ax.bar(x + width/2, density_means, width, label='Density (inverse)',
              alpha=0.7, color='skyblue')
        ax.set_ylabel('Contribution (normalized)')
        ax.set_xlabel('Dataset')
        ax.set_title('Uncertainty vs Density Contributions')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_all_plots(self):
        """Generate all Stage 1 visualizations"""
        print("\n" + "="*60)
        print("GENERATING AURORA STAGE 1 VISUALIZATIONS")
        print("="*60)
        
        plot_dir = self.results_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        
        # 1. Region summary
        print("\n1. Region distribution summary...")
        self.plot_region_summary(save_path=plot_dir / 'region_summary.png')
        
        # 2. Importance statistics
        print("2. Importance statistics...")
        self.plot_importance_statistics(save_path=plot_dir / 'importance_stats.png')
        
        # 3. Individual dataset visualizations (for 2D datasets)
        print("\n3. Individual dataset visualizations (2D only)...")
        
        # Try to visualize synthetic_heteroscedastic (should be 2D)
        synth_dir = self.results_dir / 'synthetic_heteroscedastic'
        if synth_dir.exists():
            try:
                # Load data
                data_file = PROJECT_ROOT / 'data' / 'synthetic_heteroscedastic.npz'
                if data_file.exists():
                    data = np.load(data_file)
                    X = data['X']
                    
                    # Load region data
                    region_data = np.load(synth_dir / 'region_data.npz')
                    importance_scores = region_data['importance_scores']
                    uncertainty_map = region_data['uncertainty_map']
                    density_map = region_data['density_map']
                    high_thresh = region_data['high_threshold_value']
                    low_thresh = region_data['low_threshold_value']
                    
                    # Use subset for visualization
                    if len(X) > 2000:
                        idx = np.random.choice(len(X), 2000, replace=False)
                        X = X[idx]
                        importance_scores = importance_scores[idx]
                        uncertainty_map = uncertainty_map[idx]
                        density_map = density_map[idx]
                    
                    print("\n  Generating synthetic_heteroscedastic visualizations...")
                    self.plot_2d_regions(X, importance_scores, high_thresh, low_thresh,
                                        'synthetic_heteroscedastic',
                                        save_path=plot_dir / 'synthetic_regions.png')
                    
                    self.plot_components(X, uncertainty_map, density_map, importance_scores,
                                        'synthetic_heteroscedastic',
                                        save_path=plot_dir / 'synthetic_components.png')
            except Exception as e:
                print(f"  Could not visualize synthetic_heteroscedastic: {e}")
        
        print(f"\n[OK] All plots saved to {plot_dir}")
    
    def plot_components(self, X_train, uncertainty_map, density_map,
                       importance_scores, dataset_name, save_path=None):
        """Plot uncertainty, density, and combined importance"""
        if X_train.shape[1] != 2:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Normalize for viz
        unc_norm = (uncertainty_map - uncertainty_map.min()) / \
                   (uncertainty_map.max() - uncertainty_map.min() + 1e-10)
        
        # 1. Uncertainty
        ax = axes[0, 0]
        scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=unc_norm,
                           cmap='Reds', s=30, alpha=0.6, edgecolors='none')
        plt.colorbar(scatter, ax=ax, label='Uncertainty')
        ax.set_title('GP Uncertainty Map')
        ax.grid(True, alpha=0.3)
        
        # 2. Density
        ax = axes[0, 1]
        scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=density_map,
                           cmap='Blues', s=30, alpha=0.6, edgecolors='none')
        plt.colorbar(scatter, ax=ax, label='Density Score')
        ax.set_title('Data Density Map')
        ax.grid(True, alpha=0.3)
        
        # 3. Combined
        ax = axes[1, 0]
        scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=importance_scores,
                           cmap='RdYlBu_r', s=30, alpha=0.6, edgecolors='none')
        plt.colorbar(scatter, ax=ax, label='Importance')
        ax.set_title('Combined Importance Score')
        ax.grid(True, alpha=0.3)
        
        # 4. Correlation
        ax = axes[1, 1]
        ax.scatter(unc_norm, density_map, alpha=0.3, s=20)
        corr = np.corrcoef(unc_norm, density_map)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
               transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel('Uncertainty')
        ax.set_ylabel('Density Score')
        ax.set_title('Uncertainty vs Density')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{dataset_name}: Importance Components', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    viz = Stage1Visualizer()
    viz.generate_all_plots()