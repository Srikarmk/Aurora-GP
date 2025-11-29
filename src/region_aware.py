"""
AURORA - Stage 2: Region-Aware Approximation V2 (FIXED)
Train all models on all data with different complexities
Route test points to appropriate model
"""

import numpy as np
from pathlib import Path
import json
import time
import tracemalloc
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).parent))

from region_identification import RegionIdentifier
from approximations import RandomFourierFeatures, NystromApproximation

PROJECT_ROOT = Path(__file__).parent.parent


def load_baseline_hyperparameters(dataset_name):
    """Load tuned hyperparameters from baseline approximation results"""
    baseline_path = PROJECT_ROOT / 'results' / 'approximations' / 'summary.json'
    
    if not baseline_path.exists():
        return {'lengthscale': 1.0, 'sigma_noise': 0.1}
    
    with open(baseline_path, 'r') as f:
        baseline_results = json.load(f)
    
    if dataset_name not in baseline_results:
        return {'lengthscale': 1.0, 'sigma_noise': 0.1}
    
    dataset_baseline = baseline_results[dataset_name]
    
    if 'nystrom' in dataset_baseline and 'lengthscale' in dataset_baseline['nystrom']:
        lengthscale = dataset_baseline['nystrom']['lengthscale']
    elif 'rff' in dataset_baseline and 'lengthscale' in dataset_baseline['rff']:
        lengthscale = dataset_baseline['rff']['lengthscale']
    else:
        lengthscale = 1.0
    
    return {'lengthscale': lengthscale, 'sigma_noise': 0.1}


class RegionAwareApproximation:
    """
    AURORA V2: Region-aware kernel approximation
    
    STRATEGY:
    - Train ALL models on ALL data (no data splitting)
    - Use DIFFERENT complexities:
        Low importance: 100 RFF features (cheap)
        Medium importance: 300 Nyström landmarks (moderate)
        High importance: 800 Nyström landmarks (expensive)
    - Route test points to model matching their importance
    """
    
    def __init__(self,
                 low_n_components=100,
                 medium_n_landmarks=300,
                 high_n_landmarks=800,
                 lengthscale=1.0,
                 sigma_noise=0.1,
                 random_state=42):
        """
        Args:
            low_n_components: RFF components for low importance (cheap)
            medium_n_landmarks: Nyström landmarks for medium (moderate)
            high_n_landmarks: Nyström landmarks for high (expensive)
        """
        self.low_n_components = low_n_components
        self.medium_n_landmarks = medium_n_landmarks
        self.high_n_landmarks = high_n_landmarks
        self.lengthscale = lengthscale
        self.sigma_noise = sigma_noise
        self.random_state = random_state
        
        self.models = {}
        self.region_identifier = None
        
        self.training_time = 0
        self.region_id_time = 0
        self.peak_memory_mb = 0
        
    def fit(self, X_train, y_train, region_identifier=None, n_gp_iter=30):
        """Train region-aware approximation"""
        print("\n" + "="*60)
        print("AURORA V2: REGION-AWARE APPROXIMATION")
        print("Strategy: Train all models on all data")
        print("="*60)
        
        start_time = time.time()
        tracemalloc.start()
        
        # Step 1: Region identification
        print("\nSTEP 1: Region Identification")
        if region_identifier is not None:
            self.region_identifier = region_identifier
            self.region_id_time = region_identifier.fitting_time
        else:
            self.region_identifier = RegionIdentifier(random_state=self.random_state)
            self.region_identifier.fit(X_train, y_train, n_gp_iter=n_gp_iter)
            self.region_id_time = self.region_identifier.fitting_time
        
        # Get training region stats (for reporting)
        importance_train = self.region_identifier.importance_scores
        high_thresh = self.region_identifier.high_threshold_value
        low_thresh = self.region_identifier.low_threshold_value
        
        regions_train = np.ones(len(X_train), dtype=int)
        regions_train[importance_train >= high_thresh] = 2
        regions_train[importance_train <= low_thresh] = 0
        
        n_low = np.sum(regions_train == 0)
        n_med = np.sum(regions_train == 1)
        n_high = np.sum(regions_train == 2)
        
        print(f"  Training regions: Low={n_low} ({n_low/len(X_train)*100:.1f}%), "
              f"Med={n_med} ({n_med/len(X_train)*100:.1f}%), "
              f"High={n_high} ({n_high/len(X_train)*100:.1f}%)")
        
        # Step 2: Train all models on ALL data with different complexities
        print("\nSTEP 2: Training Models (All on Full Data)")
        
        print(f"  LOW complexity (RFF, D={self.low_n_components}) on {len(X_train)} points...")
        self.models[0] = RandomFourierFeatures(
            n_components=self.low_n_components,
            lengthscale=self.lengthscale,
            sigma_noise=self.sigma_noise,
            random_state=self.random_state
        )
        self.models[0].fit(X_train, y_train)
        
        print(f"  MEDIUM complexity (Nyström, m={self.medium_n_landmarks}) on {len(X_train)} points...")
        self.models[1] = NystromApproximation(
            n_landmarks=self.medium_n_landmarks,
            lengthscale=self.lengthscale,
            sigma_noise=self.sigma_noise,
            random_state=self.random_state
        )
        self.models[1].fit(X_train, y_train)
        
        print(f"  HIGH complexity (Nyström, m={self.high_n_landmarks}) on {len(X_train)} points...")
        self.models[2] = NystromApproximation(
            n_landmarks=self.high_n_landmarks,
            lengthscale=self.lengthscale,
            sigma_noise=self.sigma_noise,
            random_state=self.random_state
        )
        self.models[2].fit(X_train, y_train)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_memory_mb = peak / (1024 * 1024)
        
        self.training_time = time.time() - start_time
        
        print(f"\n✓ Training complete in {self.training_time:.2f}s")
        
        return self
    
    def predict(self, X_test, return_std=True):
        """Make predictions by routing to appropriate model"""
        
        # Get region assignments (with clipping fix!)
        regions, _ = self.region_identifier.predict_regions(X_test, adaptive_thresholds=False)
        
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
    
    def evaluate(self, X_test, y_test, verbose=True):
        """Comprehensive evaluation"""
        start_time = time.time()
        
        y_pred, y_std, regions = self.predict(X_test, return_std=True)
        
        inference_time = time.time() - start_time
        
        # Overall metrics
        overall_metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'nll': self._negative_log_likelihood(y_test, y_pred, y_std),
            'ece': self._expected_calibration_error(y_test, y_pred, y_std),
            'mae': float(np.mean(np.abs(y_test - y_pred))),
            'training_time': float(self.training_time),
            'region_id_time': float(self.region_id_time),
            'inference_time': float(inference_time),
            'peak_memory_mb': float(self.peak_memory_mb),
            'mean_std': float(y_std.mean()),
            'std_std': float(y_std.std())
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
                    'rmse': float(np.sqrt(mean_squared_error(y_true_region, y_pred_region))),
                    'nll': self._negative_log_likelihood(y_true_region, y_pred_region, y_std_region),
                    'ece': self._expected_calibration_error(y_true_region, y_pred_region, y_std_region),
                    'mean_std': float(y_std_region.mean())
                }
            else:
                region_metrics[region_name] = {'n_points': 0}
        
        if verbose:
            print("\n" + "="*60)
            print("AURORA V2 EVALUATION")
            print("="*60)
            print(f"Overall:")
            print(f"  RMSE: {overall_metrics['rmse']:.4f}")
            print(f"  NLL:  {overall_metrics['nll']:.4f}")
            print(f"  ECE:  {overall_metrics['ece']:.4f}")
            
            print(f"\nPer-region:")
            for region_name, metrics in region_metrics.items():
                if metrics['n_points'] > 0:
                    print(f"  {region_name.upper()} ({metrics['n_points']} pts, {metrics['pct_points']:.1f}%):")
                    print(f"    ECE: {metrics['ece']:.4f}")
            print("="*60)
        
        return {
            'overall': overall_metrics,
            'per_region': region_metrics
        }
    
    def _negative_log_likelihood(self, y_true, y_pred, y_std):
        var = y_std ** 2
        nll = 0.5 * np.log(2 * np.pi * var) + (y_true - y_pred)**2 / (2 * var)
        return float(np.mean(nll))
    
    def _expected_calibration_error(self, y_true, y_pred, y_std, n_bins=10):
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
        return {
            'low_n_components': self.low_n_components,
            'medium_n_landmarks': self.medium_n_landmarks,
            'high_n_landmarks': self.high_n_landmarks,
            'lengthscale': self.lengthscale,
            'sigma_noise': self.sigma_noise,
            'strategy': 'train_on_all_data_v2',
            'random_state': self.random_state
        }


def run_region_aware_benchmark(data_dir=None, results_dir=None):
    """Run region-aware approximation on all datasets"""
    if data_dir is None:
        data_dir = PROJECT_ROOT / 'data'
    if results_dir is None:
        results_dir = PROJECT_ROOT / 'results' / 'region_aware_v2'
    
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("AURORA V2 - REGION-AWARE APPROXIMATION BENCHMARK")
    print("="*70)
    
    datasets = list(data_dir.glob('*.npz'))
    
    if not datasets:
        print(f"[ERROR] No datasets found in {data_dir}")
        return None
    
    print(f"\nFound {len(datasets)} dataset(s)")
    
    all_results = {}
    
    for dataset_file in datasets:
        dataset_name = dataset_file.stem
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        try:
            # Load data
            data = np.load(dataset_file, allow_pickle=True)
            X = data['X']
            y = data['y']
            
            # Limit size
            if len(X) > 10000:
                print(f"  Subsampling from {len(X)} to 10000")
                idx = np.random.RandomState(42).choice(len(X), 10000, replace=False)
                X = X[idx]
                y = y[idx]
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Load hyperparameters
            hyperparams = load_baseline_hyperparameters(dataset_name)
            print(f"  Lengthscale: {hyperparams['lengthscale']:.4f}")
            
            # Fit AURORA
            print("\nFitting AURORA V2...")
            model = RegionAwareApproximation(
                low_n_components=100,
                medium_n_landmarks=300,
                high_n_landmarks=800,
                lengthscale=hyperparams['lengthscale'],
                sigma_noise=hyperparams['sigma_noise'],
                random_state=42
            )
            model.fit(X_train, y_train, n_gp_iter=30)
            
            print("\nEvaluating...")
            metrics = model.evaluate(X_test, y_test, verbose=True)
            
            # Save results
            all_results[dataset_name] = metrics
            
            dataset_dir = results_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            with open(dataset_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            with open(dataset_dir / 'config.json', 'w') as f:
                json.dump(model.get_config(), f, indent=2)
                
        except Exception as e:
            print(f"[ERROR] {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset_name] = {'error': str(e)}
    
    # Save summary
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("AURORA V2 BENCHMARK SUMMARY")
    print("="*70)
    print(f"{'Dataset':<20} {'RMSE':<10} {'NLL':<10} {'ECE':<10} {'Time(s)':<10}")
    print("-"*70)
    
    for name, results in all_results.items():
        if 'error' not in results:
            m = results['overall']
            print(f"{name:<20} {m['rmse']:<10.4f} {m['nll']:<10.4f} {m['ece']:<10.4f} {m['training_time']:<10.2f}")
    
    print("="*70)
    print(f"\n✓ Results saved to {results_dir}")
    
    return all_results


if __name__ == "__main__":
    results = run_region_aware_benchmark()
    
    print("\n✓ AURORA V2 benchmark complete!")
    print("\nNext: Compare with baselines to verify improvement")
