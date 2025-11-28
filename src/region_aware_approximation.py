"""
AURORA - Stage 2: Region-Aware Approximation
Allocate different approximation quality based on region importance
"""

import numpy as np
from pathlib import Path
import json
import time
import tracemalloc
from scipy.stats import norm
from sklearn.metrics import mean_squared_error

import sys
sys.path.insert(0, str(Path(__file__).parent))

from region_identification import RegionIdentifier
from approximations import RandomFourierFeatures, NystromApproximation

PROJECT_ROOT = Path(__file__).parent.parent


class RegionAwareApproximation:
    """
    Region-aware kernel approximation that allocates different 
    approximation quality to different importance regions.
    
    Strategy:
        - High importance (2): Nyström with more landmarks
        - Medium importance (1): Standard Nyström
        - Low importance (0): Sparse RFF (fewer components)
    """
    
    def __init__(self,
                 low_n_components=200,
                 medium_n_landmarks=500,
                 high_n_landmarks=1000,
                 lengthscale=1.0,
                 sigma_noise=0.1,
                 random_state=42):
        """
        Args:
            low_n_components: RFF components for low importance regions
            medium_n_landmarks: Nyström landmarks for medium importance
            high_n_landmarks: Nyström landmarks for high importance
            lengthscale: Kernel lengthscale
            sigma_noise: Observation noise
            random_state: Random seed
        """
        self.low_n_components = low_n_components
        self.medium_n_landmarks = medium_n_landmarks
        self.high_n_landmarks = high_n_landmarks
        self.lengthscale = lengthscale
        self.sigma_noise = sigma_noise
        self.random_state = random_state
        
        # Models for each region
        # Original allocation: more compute for high importance regions
        self.models = {
            0: RandomFourierFeatures(  # LOW importance → Sparse RFF
                n_components=low_n_components,
                lengthscale=lengthscale,
                sigma_noise=sigma_noise,
                random_state=random_state
            ),
            1: NystromApproximation(  # MEDIUM importance → Standard Nyström
                n_landmarks=medium_n_landmarks,
                lengthscale=lengthscale,
                sigma_noise=sigma_noise,
                random_state=random_state
            ),
            2: NystromApproximation(  # HIGH importance → More Nyström
                n_landmarks=high_n_landmarks,
                lengthscale=lengthscale,
                sigma_noise=sigma_noise,
                random_state=random_state
            )
        }
        
        # Region identifier (Stage 1)
        self.region_identifier = None
        
        # Timing and memory
        self.training_time = 0
        self.region_id_time = 0
        self.peak_memory_mb = 0
        
    def fit(self, X_train, y_train, region_identifier=None, n_gp_iter=30):
        """
        Train the region-aware approximation system.
        
        Args:
            X_train: Training features
            y_train: Training targets
            region_identifier: Pre-fitted RegionIdentifier (optional)
            n_gp_iter: GP iterations for region identification
        """
        start_time = time.time()
        
        # Start memory tracking
        tracemalloc.start()
        
        # Step 1: Region identification (Stage 1)
        if region_identifier is not None:
            self.region_identifier = region_identifier
            self.region_id_time = region_identifier.fitting_time
        else:
            print("  Fitting region identifier...")
            self.region_identifier = RegionIdentifier(random_state=self.random_state)
            self.region_identifier.fit(X_train, y_train, n_gp_iter=n_gp_iter)
            self.region_id_time = self.region_identifier.fitting_time
        
        # Step 2: Train all models on ALL training data
        print("  Training low importance model (RFF)...")
        self.models[0].fit(X_train, y_train)
        
        print("  Training medium importance model (Nyström)...")
        self.models[1].fit(X_train, y_train)
        
        print("  Training high importance model (Nyström+)...")
        self.models[2].fit(X_train, y_train)
        
        # Get peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_memory_mb = peak / (1024 * 1024)  # Convert to MB
        
        self.training_time = time.time() - start_time
        
        return self
    
    def predict(self, X_test, return_std=True):
        """
        Make predictions by routing test points to appropriate models.
        
        Args:
            X_test: Test features (n_test, n_features)
            return_std: Whether to return uncertainty estimates
            
        Returns:
            y_pred: Predictions
            y_std: Standard deviations (if return_std=True)
            regions: Region assignments for each test point
        """
        # Get region assignments (use training thresholds, not adaptive)
        regions, importance_scores = self.region_identifier.predict_regions(X_test, adaptive_thresholds=False)
        
        n_test = len(X_test)
        y_pred = np.zeros(n_test)
        y_std = np.zeros(n_test) if return_std else None
        
        # Route to appropriate models
        for region_label in [0, 1, 2]:
            mask = (regions == region_label)
            if mask.sum() > 0:
                X_region = X_test[mask]
                
                if return_std:
                    pred, std = self.models[region_label].predict(X_region, return_std=True)
                    y_pred[mask] = pred
                    y_std[mask] = std
                else:
                    y_pred[mask] = self.models[region_label].predict(X_region, return_std=False)
        
        if return_std:
            return y_pred, y_std, regions
        return y_pred, regions
    
    def evaluate(self, X_test, y_test):
        """
        Comprehensive evaluation of the region-aware approximation.
        
        Returns metrics overall and per-region.
        """
        start_time = time.time()
        
        y_pred, y_std, regions = self.predict(X_test, return_std=True)
        
        inference_time = time.time() - start_time
        
        # Overall metrics
        overall_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'nll': self._negative_log_likelihood(y_test, y_pred, y_std),
            'ece': self._expected_calibration_error(y_test, y_pred, y_std),
            'mae': np.mean(np.abs(y_test - y_pred)),
            'training_time': self.training_time,
            'region_id_time': self.region_id_time,
            'inference_time': inference_time,
            'peak_memory_mb': self.peak_memory_mb,
            'mean_std': y_std.mean(),
            'std_std': y_std.std()
        }
        
        # Per-region metrics
        region_names = {0: 'low', 1: 'medium', 2: 'high'}
        region_metrics = {}
        
        for region_label, region_name in region_names.items():
            mask = (regions == region_label)
            n_points = mask.sum()
            
            if n_points > 0:
                y_true_region = y_test[mask]
                y_pred_region = y_pred[mask]
                y_std_region = y_std[mask]
                
                region_metrics[region_name] = {
                    'n_points': int(n_points),
                    'pct_points': float(n_points / len(y_test) * 100),
                    'rmse': np.sqrt(mean_squared_error(y_true_region, y_pred_region)),
                    'nll': self._negative_log_likelihood(y_true_region, y_pred_region, y_std_region),
                    'ece': self._expected_calibration_error(y_true_region, y_pred_region, y_std_region),
                    'mean_std': float(y_std_region.mean())
                }
            else:
                region_metrics[region_name] = {'n_points': 0}
        
        return {
            'overall': overall_metrics,
            'per_region': region_metrics
        }
    
    def _negative_log_likelihood(self, y_true, y_pred, y_std):
        """Compute NLL assuming Gaussian likelihood"""
        var = y_std ** 2
        nll = 0.5 * np.log(2 * np.pi * var) + (y_true - y_pred)**2 / (2 * var)
        return float(np.mean(nll))
    
    def _expected_calibration_error(self, y_true, y_pred, y_std, n_bins=10):
        """Compute ECE for regression"""
        errors = np.abs(y_true - y_pred)
        confidence_levels = np.linspace(0.1, 0.9, n_bins)
        
        ece = 0.0
        for conf in confidence_levels:
            z_score = norm.ppf((1 + conf) / 2)
            predicted_interval = y_std * z_score
            within_interval = (errors <= predicted_interval).astype(float)
            observed_conf = within_interval.mean()
            ece += np.abs(observed_conf - conf) / n_bins
        
        return float(ece)
    
    def get_config(self):
        """Return configuration dict"""
        return {
            'low_n_components': self.low_n_components,
            'medium_n_landmarks': self.medium_n_landmarks,
            'high_n_landmarks': self.high_n_landmarks,
            'lengthscale': self.lengthscale,
            'sigma_noise': self.sigma_noise,
            'random_state': self.random_state
        }


def load_baseline_hyperparameters(dataset_name):
    """
    Load tuned hyperparameters from baseline approximation results.
    """
    baseline_path = PROJECT_ROOT / 'results' / 'approximations' / 'summary.json'
    
    if not baseline_path.exists():
        print(f"  [WARNING] Baseline results not found, using default hyperparameters")
        return {'lengthscale': 1.0, 'sigma_noise': 0.1}
    
    with open(baseline_path, 'r') as f:
        baseline_results = json.load(f)
    
    if dataset_name not in baseline_results:
        print(f"  [WARNING] No baseline for {dataset_name}, using defaults")
        return {'lengthscale': 1.0, 'sigma_noise': 0.1}
    
    # Get lengthscale from Nyström results (or RFF if Nyström not available)
    dataset_baseline = baseline_results[dataset_name]
    
    if 'nystrom' in dataset_baseline and 'lengthscale' in dataset_baseline['nystrom']:
        lengthscale = dataset_baseline['nystrom']['lengthscale']
    elif 'rff' in dataset_baseline and 'lengthscale' in dataset_baseline['rff']:
        lengthscale = dataset_baseline['rff']['lengthscale']
    else:
        lengthscale = 1.0
    
    return {'lengthscale': lengthscale, 'sigma_noise': 0.1}


def run_region_aware_benchmark(data_dir=None, results_dir=None):
    """
    Run region-aware approximation on all datasets and compare with baselines.
    """
    if data_dir is None:
        data_dir = PROJECT_ROOT / 'data'
    if results_dir is None:
        results_dir = PROJECT_ROOT / 'results' / 'region_aware'
    
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = list(data_dir.glob('*.npz'))
    
    if not datasets:
        print(f"[ERROR] No datasets found in {data_dir}")
        return None
    
    all_results = {}
    
    for dataset_file in datasets:
        dataset_name = dataset_file.stem
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print('='*60)
        
        try:
            # Load data
            data = np.load(dataset_file, allow_pickle=True)
            X = data['X']
            y = data['y']
            
            # Limit size for tractability
            if len(X) > 10000:
                idx = np.random.RandomState(42).choice(len(X), 10000, replace=False)
                X = X[idx]
                y = y[idx]
            
            # Train/test split
            n_train = int(0.8 * len(X))
            indices = np.random.RandomState(42).permutation(len(X))
            train_idx, test_idx = indices[:n_train], indices[n_train:]
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Load tuned hyperparameters from baseline
            hyperparams = load_baseline_hyperparameters(dataset_name)
            print(f"  Using lengthscale: {hyperparams['lengthscale']:.4f}")
            
            # Fit and evaluate region-aware model
            print("\nFitting Region-Aware Approximation...")
            model = RegionAwareApproximation(
                low_n_components=200,
                medium_n_landmarks=500,
                high_n_landmarks=1000,
                lengthscale=hyperparams['lengthscale'],
                sigma_noise=hyperparams['sigma_noise'],
                random_state=42
            )
            model.fit(X_train, y_train)
            
            print("\nEvaluating...")
            metrics = model.evaluate(X_test, y_test)
            
            # Print results
            print(f"\n  Overall Results:")
            print(f"    RMSE: {metrics['overall']['rmse']:.4f}")
            print(f"    NLL:  {metrics['overall']['nll']:.4f}")
            print(f"    ECE:  {metrics['overall']['ece']:.4f}")
            print(f"    Peak Memory: {metrics['overall']['peak_memory_mb']:.2f} MB")
            
            print(f"\n  Per-Region Results:")
            for region_name, region_data in metrics['per_region'].items():
                if region_data['n_points'] > 0:
                    print(f"    {region_name.upper()}: "
                          f"n={region_data['n_points']}, "
                          f"RMSE={region_data['rmse']:.4f}, "
                          f"NLL={region_data['nll']:.4f}, "
                          f"ECE={region_data['ece']:.4f}")
            
            # Save results
            all_results[dataset_name] = metrics
            
            dataset_dir = results_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            with open(dataset_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            with open(dataset_dir / 'config.json', 'w') as f:
                json.dump(model.get_config(), f, indent=2)
                
        except Exception as e:
            print(f"[ERROR] Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset_name] = {'error': str(e)}
    
    # Save summary
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {results_dir}")
    print('='*60)
    
    return all_results


if __name__ == "__main__":
    run_region_aware_benchmark()