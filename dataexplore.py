import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import pdist
from pathlib import Path

class DatasetExplorer:
    """Explore dataset characteristics for region-aware approximation"""
    
    def __init__(self, data_dir='./data', results_dir='./results'):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.datasets = {}
        
    def load_datasets(self):
        """Load all saved datasets"""
        print("Loading datasets...")
        for npz_file in self.data_dir.glob('*.npz'):
            name = npz_file.stem
            data = np.load(npz_file, allow_pickle=True)
            # Ensure X and y are numeric numpy arrays
            X = np.asarray(data['X'], dtype=np.float64)
            y = np.asarray(data['y'], dtype=np.float64)
            self.datasets[name] = {
                'X': X,
                'y': y,
                'feature_names': data['feature_names'].tolist()
            }
            print(f"[OK] Loaded {name}: {X.shape}")
        return self.datasets
    
    def analyze_data_density(self, X, n_bins=50):
        """Analyze how uniformly distributed the data is"""
        # Ensure X is a numeric numpy array
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] == 2:
            # For 2D, use 2D histogram
            H, xedges, yedges = np.histogram2d(X[:, 0], X[:, 1], bins=n_bins)
            density_variance = H.var()
            density_cv = H.std() / (H.mean() + 1e-10)  # Coefficient of variation
            
            return {
                'density_variance': density_variance,
                'density_cv': density_cv,
                'uniformity_score': 1.0 / (1.0 + density_cv)  # Higher = more uniform
            }
        else:
            # For higher dimensions, use pairwise distances
            if X.shape[0] > 1000:
                # Sample for efficiency
                idx = np.random.choice(X.shape[0], 1000, replace=False)
                X_sample = X[idx]
            else:
                X_sample = X
            
            distances = pdist(X_sample)
            return {
                'distance_variance': distances.var(),
                'distance_cv': distances.std() / distances.mean(),
                'uniformity_score': 1.0 / (1.0 + distances.std() / distances.mean())
            }
    
    def analyze_output_variance(self, X, y, window_size=0.1):
        """Analyze variance in different regions of the input space"""
        from sklearn.neighbors import NearestNeighbors
        
        # Ensure X and y are numeric numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Sample points to analyze
        n_test = min(500, X.shape[0] // 2)
        test_idx = np.random.choice(X.shape[0], n_test, replace=False)
        X_test = X[test_idx]
        
        # For each test point, compute local variance
        nbrs = NearestNeighbors(n_neighbors=min(50, X.shape[0] // 10))
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X_test)
        
        local_variances = []
        for i in range(len(X_test)):
            neighbor_y = y[indices[i]]
            local_variances.append(neighbor_y.var())
        
        local_variances = np.array(local_variances)
        
        return {
            'mean_local_variance': local_variances.mean(),
            'variance_of_variance': local_variances.var(),
            'heteroscedasticity_score': local_variances.std() / (local_variances.mean() + 1e-10)
        }
    
    def analyze_function_smoothness(self, X, y):
        """Estimate function smoothness in different regions"""
        from sklearn.neighbors import NearestNeighbors
        
        # Ensure X and y are numeric numpy arrays
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Use nearest neighbors to estimate local gradients
        nbrs = NearestNeighbors(n_neighbors=min(10, X.shape[0] // 20))
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Compute local gradient magnitudes
        gradient_magnitudes = []
        for i in range(min(500, len(X))):
            neighbors_X = X[indices[i]]
            neighbors_y = y[indices[i]]
            
            # Simple gradient estimate
            diffs = np.abs(neighbors_y[1:] - neighbors_y[0])
            spatial_diffs = np.linalg.norm(neighbors_X[1:] - neighbors_X[0], axis=1)
            gradients = diffs / (spatial_diffs + 1e-10)
            gradient_magnitudes.append(gradients.mean())
        
        gradient_magnitudes = np.array(gradient_magnitudes)
        
        return {
            'mean_gradient': gradient_magnitudes.mean(),
            'gradient_variance': gradient_magnitudes.var(),
            'smoothness_variation': gradient_magnitudes.std() / (gradient_magnitudes.mean() + 1e-10)
        }
    
    def comprehensive_analysis(self):
        """Run comprehensive analysis on all datasets"""
        print("COMPREHENSIVE DATASET ANALYSIS")
        results = {}
        for name, data in self.datasets.items():
            print(f"\nAnalyzing {name}...")
            X, y = data['X'], data['y']
            
            density = self.analyze_data_density(X)
            variance = self.analyze_output_variance(X, y)
            smoothness = self.analyze_function_smoothness(X, y)
            
            results[name] = {
                **density,
                **variance,
                **smoothness
            }
            
            print(f"  Uniformity Score: {density['uniformity_score']:.3f} (higher = more uniform)")
            print(f"  Heteroscedasticity: {variance['heteroscedasticity_score']:.3f} (higher = more varying noise)")
            print(f"  Smoothness Variation: {smoothness['smoothness_variation']:.3f} (higher = more varying smoothness)")
        
        return results
    
    def visualize_characteristics(self, dataset_name):
        """Visualize key characteristics of a specific dataset"""
        if dataset_name not in self.datasets:
            print(f"Dataset {dataset_name} not found!")
            return
        
        data = self.datasets[dataset_name]
        # Ensure X and y are numeric numpy arrays
        X = np.asarray(data['X'], dtype=np.float64)
        y = np.asarray(data['y'], dtype=np.float64)
        
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Data distribution (first 2 features)
        ax1 = plt.subplot(2, 3, 1)
        scatter = ax1.scatter(X[:, 0], X[:, 1] if X.shape[1] > 1 else X[:, 0], 
                             c=y, cmap='viridis', alpha=0.6, s=20)
        ax1.set_xlabel(data['feature_names'][0])
        ax1.set_ylabel(data['feature_names'][1] if X.shape[1] > 1 else 'Index')
        ax1.set_title(f'{dataset_name}: Data Distribution')
        plt.colorbar(scatter, ax=ax1, label='Target Value')
        
        # 2. Target distribution
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(y, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Target Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Target Distribution')
        ax2.axvline(y.mean(), color='red', linestyle='--', label=f'Mean: {y.mean():.2f}')
        ax2.legend()
        
        # 3. Feature distributions
        ax3 = plt.subplot(2, 3, 3)
        for i in range(min(3, X.shape[1])):
            ax3.hist(X[:, i], bins=30, alpha=0.5, label=data['feature_names'][i])
        ax3.set_xlabel('Feature Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Feature Distributions')
        ax3.legend()
        
        # 4. Local variance visualization (2D projection using first 2 features)
        from sklearn.neighbors import NearestNeighbors
        
        ax4 = plt.subplot(2, 3, 4)
        
        # Use first 2 features for visualization (works for any dimensionality)
        X_2d = X[:, :2]
        
        # Create grid
        x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
        y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                            np.linspace(y_min, y_max, 50))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Compute local variance at each grid point using full feature space
        nbrs = NearestNeighbors(n_neighbors=min(50, X.shape[0] // 10))
        nbrs.fit(X)  # Use full X for nearest neighbors
        
        # Find neighbors in full space, but project grid to 2D for visualization
        # For each grid point, find nearest neighbors in full space
        grid_points_full = np.zeros((grid_points.shape[0], X.shape[1]))
        grid_points_full[:, :2] = grid_points
        # Fill remaining dimensions with median values
        if X.shape[1] > 2:
            for i in range(2, X.shape[1]):
                grid_points_full[:, i] = np.median(X[:, i])
        
        _, indices = nbrs.kneighbors(grid_points_full)
        
        local_vars = np.array([y[idx].var() for idx in indices])
        local_vars = local_vars.reshape(xx.shape)
        
        im = ax4.contourf(xx, yy, local_vars, levels=15, cmap='RdYlBu_r')
        ax4.scatter(X_2d[:, 0], X_2d[:, 1], c='black', s=1, alpha=0.3)
        ax4.set_xlabel(data['feature_names'][0])
        ax4.set_ylabel(data['feature_names'][1])
        title_suffix = " (2D projection)" if X.shape[1] > 2 else ""
        ax4.set_title(f'Local Variance Map{title_suffix}')
        plt.colorbar(im, ax=ax4, label='Local Variance')
        
        # 5. Correlation matrix
        ax5 = plt.subplot(2, 3, 5)
        X_with_y = np.column_stack([X, y])
        corr = np.corrcoef(X_with_y.T)
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   xticklabels=data['feature_names'] + ['target'],
                   yticklabels=data['feature_names'] + ['target'],
                   ax=ax5)
        ax5.set_title('Feature Correlations')
        
        # 6. Residual plot (simple linear baseline)
        from sklearn.linear_model import LinearRegression
        ax6 = plt.subplot(2, 3, 6)
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred = lr.predict(X)
        residuals = y - y_pred
        ax6.scatter(y_pred, residuals, alpha=0.5, s=20)
        ax6.axhline(0, color='red', linestyle='--')
        ax6.set_xlabel('Predicted Value')
        ax6.set_ylabel('Residual')
        ax6.set_title('Residuals (Linear Baseline)')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'{dataset_name}_analysis.png', dpi=150, bbox_inches='tight')
        print(f"[OK] Visualization saved to {self.results_dir / f'{dataset_name}_analysis.png'}")
        
        return fig
    
    def recommend_datasets(self, analysis_results):
        """Recommend which datasets are best for testing region-aware approximation"""
        print("DATASET RECOMMENDATIONS FOR REGION-AWARE TESTING")
        
        scores = {}
        for name, metrics in analysis_results.items():
            # Score based on: non-uniformity + heteroscedasticity + varying smoothness
            score = (
                (1 - metrics['uniformity_score']) * 0.4 +  # Want non-uniform
                metrics['heteroscedasticity_score'] * 0.3 +  # Want varying noise
                metrics['smoothness_variation'] * 0.3  # Want varying smoothness
            )
            scores[name] = score
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        print("\nDatasets ranked by suitability (higher = better for region-aware approach):\n")
        for i, (name, score) in enumerate(ranked, 1):
            metrics = analysis_results[name]
            print(f"{i}. {name.upper()} (Score: {score:.3f})")
            print(f"   - Uniformity: {metrics['uniformity_score']:.3f} (lower is better)")
            print(f"   - Heteroscedasticity: {metrics['heteroscedasticity_score']:.3f}")
            print(f"   - Smoothness Variation: {metrics['smoothness_variation']:.3f}")
            
            # Interpretation
            if score > 0.5:
                print(f"   -> EXCELLENT candidate for region-aware approximation")
            elif score > 0.3:
                print(f"   -> GOOD candidate for region-aware approximation")
            else:
                print(f"   -> Moderate candidate - may show smaller improvements")
            print()
        
        return ranked


if __name__ == "__main__":
    # Create explorer
    explorer = DatasetExplorer()
    
    # Load datasets
    datasets = explorer.load_datasets()
    
    if not datasets:
        print("\n[WARNING] No datasets found. Run the dataset collection script first!")
    else:
        # Run comprehensive analysis
        results = explorer.comprehensive_analysis()
        
        # Get recommendations
        recommendations = explorer.recommend_datasets(results)
        
        # Visualize each dataset
        print("\n" + "="*80)
        print("GENERATING DETAILED VISUALIZATIONS")
        print("="*80)
        for name in datasets.keys():
            print(f"\nGenerating visualization for {name}...")
            explorer.visualize_characteristics(name)
        
        print("\n[OK] Analysis complete!")