"""
AURORA - Stage 1: Region Identification (FIXED - with clipping)
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import time
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent

from gp_baseline import GaussianProcessBaseline


class RegionIdentifier:
    """Identify importance regions for adaptive kernel approximation"""
    
    def __init__(self, 
                 max_gp_samples=5000,
                 n_neighbors=50,
                 uncertainty_weight=0.5,
                 density_weight=0.5,
                 high_threshold=0.7,
                 low_threshold=0.3,
                 random_state=42):
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
        """Identify regions based on importance"""
        print("\n" + "="*60)
        print("AURORA STAGE 1: REGION IDENTIFICATION")
        print("="*60)
        
        start_time = time.time()
        
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        
        print(f"\nDataset: {len(X_train)} samples, {X_train.shape[1]} features")
        
        # Train GP for uncertainty map
        print("\nSTEP 1: Training GP for Uncertainty Map")
        self.gp = GaussianProcessBaseline(
            max_train_size=self.max_gp_samples,
            random_state=self.random_state
        )
        self.gp.fit(X_train, y_train, n_iter=n_gp_iter)
        self.uncertainty_map = self.gp.get_uncertainty_map(X_train)
        
        print(f"  Uncertainty range: [{self.uncertainty_map.min():.4f}, {self.uncertainty_map.max():.4f}]")
        
        # Compute data density
        print("\nSTEP 2: Computing Data Density")
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        nbrs.fit(self.X_train_scaled)
        
        distances, _ = nbrs.kneighbors(self.X_train_scaled)
        avg_distances = distances[:, 1:].mean(axis=1)
        
        density_scores = (avg_distances - avg_distances.min()) / (avg_distances.max() - avg_distances.min() + 1e-10)
        self.density_map = density_scores
        
        print(f"  Density range: [{self.density_map.min():.4f}, {self.density_map.max():.4f}]")
        
        # Combine into importance scores
        print("\nSTEP 3: Computing Importance Scores")
        uncertainty_normalized = (self.uncertainty_map - self.uncertainty_map.min()) / \
                                (self.uncertainty_map.max() - self.uncertainty_map.min() + 1e-10)
        
        self.importance_scores = (
            self.uncertainty_weight * uncertainty_normalized +
            self.density_weight * self.density_map
        )
        
        print(f"  Importance range: [{self.importance_scores.min():.4f}, {self.importance_scores.max():.4f}]")
        
        # Partition into regions
        print("\nSTEP 4: Partitioning Regions")
        self.high_threshold_value = np.percentile(self.importance_scores, self.high_threshold * 100)
        self.low_threshold_value = np.percentile(self.importance_scores, self.low_threshold * 100)
        
        print(f"  Thresholds: high >= {self.high_threshold_value:.4f}, low <= {self.low_threshold_value:.4f}")
        
        n_high = np.sum(self.importance_scores >= self.high_threshold_value)
        n_low = np.sum(self.importance_scores <= self.low_threshold_value)
        n_medium = len(self.importance_scores) - n_high - n_low
        
        print(f"  High:   {n_high} ({n_high/len(X_train)*100:.1f}%)")
        print(f"  Medium: {n_medium} ({n_medium/len(X_train)*100:.1f}%)")
        print(f"  Low:    {n_low} ({n_low/len(X_train)*100:.1f}%)")
        
        self.fitting_time = time.time() - start_time
        print(f"\n✓ Region identification complete in {self.fitting_time:.2f}s")
        
        return self
    
    def predict_regions(self, X_test, adaptive_thresholds=False):
        """
        Predict region labels for test points
        
        CRITICAL FIX: Clip test values to training range to prevent explosion
        """
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get uncertainty for test points
        uncertainties = self.gp.get_uncertainty_map(X_test)
        
        # CRITICAL FIX: Clip to training range
        uncertainties_clipped = np.clip(uncertainties, 
                                       self.uncertainty_map.min(), 
                                       self.uncertainty_map.max())
        
        uncertainty_normalized = (uncertainties_clipped - self.uncertainty_map.min()) / \
                                (self.uncertainty_map.max() - self.uncertainty_map.min() + 1e-10)
        
        # Get density scores
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors)
        nbrs.fit(self.X_train_scaled)
        distances, _ = nbrs.kneighbors(X_test_scaled)
        avg_distances = distances.mean(axis=1)
        
        # CRITICAL FIX: Compute training distance stats and normalize properly
        train_nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
        train_nbrs.fit(self.X_train_scaled)
        train_distances, _ = train_nbrs.kneighbors(self.X_train_scaled)
        train_avg_distances = train_distances[:, 1:].mean(axis=1)
        train_dist_min = train_avg_distances.min()
        train_dist_max = train_avg_distances.max()
        
        density_scores = (avg_distances - train_dist_min) / (train_dist_max - train_dist_min + 1e-10)
        density_scores = np.clip(density_scores, 0, 1)
        
        # Compute importance
        test_importance = (
            self.uncertainty_weight * uncertainty_normalized +
            self.density_weight * density_scores
        )
        test_importance = np.clip(test_importance, 0, 1)
        
        # Use fixed thresholds
        if adaptive_thresholds:
            print("  [WARNING] adaptive_thresholds=True causes data leakage!")
            high_thresh = np.percentile(test_importance, self.high_threshold * 100)
            low_thresh = np.percentile(test_importance, self.low_threshold * 100)
        else:
            high_thresh = self.high_threshold_value
            low_thresh = self.low_threshold_value
        
        # Assign regions
        regions = np.ones(len(X_test), dtype=int)
        regions[test_importance >= high_thresh] = 2
        regions[test_importance <= low_thresh] = 0
        
        return regions, test_importance
    
    def get_region_statistics(self):
        """Get statistics about identified regions"""
        n_high = np.sum(self.importance_scores >= self.high_threshold_value)
        n_low = np.sum(self.importance_scores <= self.low_threshold_value)
        n_medium = len(self.importance_scores) - n_high - n_low
        
        return {
            'n_total': len(self.importance_scores),
            'n_high': int(n_high),
            'n_medium': int(n_medium),
            'n_low': int(n_low),
            'pct_high': float(n_high / len(self.importance_scores) * 100),
            'pct_medium': float(n_medium / len(self.importance_scores) * 100),
            'pct_low': float(n_low / len(self.importance_scores) * 100),
            'fitting_time': float(self.fitting_time)
        }
    
    def save(self, save_dir):
        """Save region identifier"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.gp.save(save_dir / 'gp_model')
        
        np.savez(
            save_dir / 'region_data.npz',
            X_train_scaled=self.X_train_scaled,
            uncertainty_map=self.uncertainty_map,
            density_map=self.density_map,
            importance_scores=self.importance_scores,
            high_threshold_value=self.high_threshold_value,
            low_threshold_value=self.low_threshold_value
        )


if __name__ == "__main__":
    from pathlib import Path
    # Run on all datasets
    pass