"""
AURORA - Stage 1: Region Identification
Identify high/medium/low importance regions for adaptive approximation
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.preprocessing import StandardScaler
import time
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

# Get project root (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent

from gp_baseline import GaussianProcessBaseline


class RegionIdentifier:
    """
    Identify importance regions for adaptive kernel approximation
    
    Importance is based on:
    1. Prediction uncertainty (variance from GP)
    2. Data density (sparse regions need more attention)
    """
    
    def __init__(self, 
                 max_gp_samples=5000,
                 n_neighbors=50,
                 uncertainty_weight=0.5,
                 density_weight=0.5,
                 high_threshold=0.7,
                 low_threshold=0.3,
                 random_state=42):
        """
        Args:
            max_gp_samples: Max samples for GP uncertainty map
            n_neighbors: Neighbors for density estimation
            uncertainty_weight: Weight for uncertainty in importance score
            density_weight: Weight for density in importance score
            high_threshold: Percentile for high importance (0.7 = top 30%)
            low_threshold: Percentile for low importance (0.3 = bottom 30%)
            random_state: Random seed
        """
        self.max_gp_samples = max_gp_samples
        self.n_neighbors = n_neighbors
        self.uncertainty_weight = uncertainty_weight
        self.density_weight = density_weight
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.random_state = random_state
        
        self.gp = None
        self.scaler = StandardScaler()
        
        self.uncertainty_map = None
        self.density_map = None
        self.importance_scores = None
        
        self.high_threshold_value = None
        self.low_threshold_value = None
        
        self.X_train_scaled = None
        
        self.fitting_time = 0
        
    def fit(self, X_train, y_train, n_gp_iter=30):
        """
        Identify regions based on importance
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples,)
            n_gp_iter: GP optimization iterations (fewer for speed)
        """
        start_time = time.time()
        
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.gp = GaussianProcessBaseline(
            max_train_size=self.max_gp_samples,
            random_state=self.random_state
        )
        self.gp.fit(X_train, y_train, n_iter=n_gp_iter)
        self.uncertainty_map = self.gp.get_uncertainty_map(X_train)
        
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nbrs.fit(self.X_train_scaled)
        distances, _ = nbrs.kneighbors(self.X_train_scaled)
        avg_distances = distances[:, 1:].mean(axis=1)
        density_scores = (avg_distances - avg_distances.min()) / (avg_distances.max() - avg_distances.min() + 1e-10)
        self.density_map = density_scores
        
        uncertainty_normalized = (self.uncertainty_map - self.uncertainty_map.min()) / \
                                (self.uncertainty_map.max() - self.uncertainty_map.min() + 1e-10)
        
        self.importance_scores = (
            self.uncertainty_weight * uncertainty_normalized +
            self.density_weight * self.density_map
        )
        
        self.high_threshold_value = np.percentile(self.importance_scores, self.high_threshold * 100)
        self.low_threshold_value = np.percentile(self.importance_scores, self.low_threshold * 100)
        
        self.fitting_time = time.time() - start_time
        
        return self
    
    def predict_regions(self, X_test, adaptive_thresholds=False):
        """
        Predict region labels for test points
        
        Args:
            X_test: Test features (n_test, n_features)
            adaptive_thresholds: If True, compute thresholds based on test data 
                                 distribution (recommended). If False, use training
                                 thresholds (can cause imbalanced regions).
            
        Returns:
            regions: Array of region labels (0=low, 1=medium, 2=high)
            importance_scores: Importance scores for test points
        """
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get uncertainty for test points
        uncertainties = self.gp.get_uncertainty_map(X_test)
        uncertainty_normalized = (uncertainties - self.uncertainty_map.min()) / \
                                (self.uncertainty_map.max() - self.uncertainty_map.min() + 1e-10)
        
        # Get density scores for test points
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors)
        nbrs.fit(self.X_train_scaled)
        distances, _ = nbrs.kneighbors(X_test_scaled)
        avg_distances = distances.mean(axis=1)
        
        density_scores = (avg_distances - self.density_map.min()) / \
                        (self.density_map.max() - self.density_map.min() + 1e-10)
        
        # Compute importance scores
        test_importance = (
            self.uncertainty_weight * uncertainty_normalized +
            self.density_weight * density_scores
        )
        
        # Determine thresholds
        if adaptive_thresholds:
            # Compute thresholds based on TEST data distribution
            high_thresh = np.percentile(test_importance, self.high_threshold * 100)
            low_thresh = np.percentile(test_importance, self.low_threshold * 100)
        else:
            # Use original training thresholds
            high_thresh = self.high_threshold_value
            low_thresh = self.low_threshold_value
        
        # Assign regions
        regions = np.ones(len(X_test), dtype=int)  # Default to medium (1)
        regions[test_importance >= high_thresh] = 2  # High
        regions[test_importance <= low_thresh] = 0   # Low
        
        return regions, test_importance
    
    def get_region_statistics(self):
        """Get statistics about identified regions"""
        n_high = np.sum(self.importance_scores >= self.high_threshold_value)
        n_low = np.sum(self.importance_scores <= self.low_threshold_value)
        n_medium = len(self.importance_scores) - n_high - n_low
        
        stats = {
            'n_total': len(self.importance_scores),
            'n_high': int(n_high),
            'n_medium': int(n_medium),
            'n_low': int(n_low),
            'pct_high': float(n_high / len(self.importance_scores) * 100),
            'pct_medium': float(n_medium / len(self.importance_scores) * 100),
            'pct_low': float(n_low / len(self.importance_scores) * 100),
            'importance_mean': float(self.importance_scores.mean()),
            'importance_std': float(self.importance_scores.std()),
            'importance_min': float(self.importance_scores.min()),
            'importance_max': float(self.importance_scores.max()),
            'uncertainty_mean': float(self.uncertainty_map.mean()),
            'uncertainty_std': float(self.uncertainty_map.std()),
            'density_mean': float(self.density_map.mean()),
            'density_std': float(self.density_map.std()),
            'fitting_time': float(self.fitting_time)
        }
        
        return stats
    
    def save(self, save_dir):
        """Save region identifier"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save GP model
        self.gp.save(save_dir / 'gp_model')
        
        # Save other attributes
        np.savez(
            save_dir / 'region_data.npz',
            X_train_scaled=self.X_train_scaled,
            uncertainty_map=self.uncertainty_map,
            density_map=self.density_map,
            importance_scores=self.importance_scores,
            high_threshold_value=self.high_threshold_value,
            low_threshold_value=self.low_threshold_value
        )
        
        # Save configuration
        config = {
            'max_gp_samples': self.max_gp_samples,
            'n_neighbors': self.n_neighbors,
            'uncertainty_weight': self.uncertainty_weight,
            'density_weight': self.density_weight,
            'high_threshold': self.high_threshold,
            'low_threshold': self.low_threshold,
            'random_state': self.random_state
        }
        
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save statistics
        stats = self.get_region_statistics()
        with open(save_dir / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        


def run_region_identification(data_dir=None,
                              results_dir=None,
                              max_gp_samples=5000,
                              uncertainty_weight=0.5,
                              density_weight=0.5):
    """
    Run region identification on all datasets
    
    Args:
        data_dir: Directory with .npz dataset files (default: PROJECT_ROOT/data)
        results_dir: Directory to save results (default: PROJECT_ROOT/results/region_identification)
        max_gp_samples: Max samples for GP training
        uncertainty_weight: Weight for uncertainty in importance
        density_weight: Weight for density in importance
    """
    if data_dir is None:
        data_dir = PROJECT_ROOT / 'data'
    if results_dir is None:
        results_dir = PROJECT_ROOT / 'results' / 'region_identification'
    
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = list(data_dir.glob('*.npz'))
    
    if not datasets:
        print(f"\n[ERROR] No datasets found in {data_dir}")
        return None
    
    all_stats = {}
    
    for dataset_file in datasets:
        dataset_name = dataset_file.stem
        
        try:
            data = np.load(dataset_file, allow_pickle=True)
            X = data['X']
            y = data['y']
            
            if len(X) > 10000:
                idx = np.random.RandomState(42).choice(len(X), 10000, replace=False)
                X = X[idx]
                y = y[idx]
            
            identifier = RegionIdentifier(
                max_gp_samples=max_gp_samples,
                uncertainty_weight=uncertainty_weight,
                density_weight=density_weight,
                random_state=42
            )
            
            identifier.fit(X, y, n_gp_iter=30)
            stats = identifier.get_region_statistics()
            all_stats[dataset_name] = stats
            
            dataset_results_dir = results_dir / dataset_name
            identifier.save(dataset_results_dir)
            
        except Exception as e:
            print(f"\n[ERROR] Error processing {dataset_name}: {e}")
            all_stats[dataset_name] = {'error': str(e)}
    
    # Save summary
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n[OK] Results saved to {results_dir}")
    
    return all_stats


if __name__ == "__main__":
    results = run_region_identification()