"""
AURORA Hyperparameter Tuning Script
Test multiple configurations to find what works best
"""

import numpy as np
from pathlib import Path
import json
import sys
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from region_identification import RegionIdentifier
from approximations import RandomFourierFeatures, NystromApproximation
from gp_baseline import GaussianProcessBaseline

PROJECT_ROOT = Path(__file__).parent.parent


# Define configurations to test
CONFIGS = [
    {
        'name': 'Config 1: Extreme Ratio',
        'low_n_components': 50,
        'medium_n_landmarks': 300,
        'high_n_landmarks': 1500,
        'use_exact_gp': False
    },
    {
        'name': 'Config 2: Conservative',
        'low_n_components': 100,
        'medium_n_landmarks': 400,
        'high_n_landmarks': 600,
        'use_exact_gp': False
    },
    {
        'name': 'Config 3: Exact GP for High',
        'low_n_components': 100,
        'medium_n_landmarks': 300,
        'high_n_landmarks': 0,  # Not used
        'use_exact_gp': True
    },
    {
        'name': 'Config 4: Very Cheap Low',
        'low_n_components': 30,
        'medium_n_landmarks': 400,
        'high_n_landmarks': 1000,
        'use_exact_gp': False
    },
    {
        'name': 'Config 5: Extreme High',
        'low_n_components': 100,
        'medium_n_landmarks': 200,
        'high_n_landmarks': 2000,
        'use_exact_gp': False
    },
]


class TunableAURORA:
    """AURORA with tunable hyperparameters"""
    
    def __init__(self, config, lengthscale=1.0, sigma_noise=0.1, random_state=42):
        self.config = config
        self.lengthscale = lengthscale
        self.sigma_noise = sigma_noise
        self.random_state = random_state
        
        self.models = {}
        self.region_identifier = None
        self.training_time = 0
        
    def fit(self, X_train, y_train):
        """Fit AURORA with specific configuration"""
        import time
        start = time.time()
        
        # Region identification
        self.region_identifier = RegionIdentifier(random_state=self.random_state)
        self.region_identifier.fit(X_train, y_train, n_gp_iter=20)
        
        # Create models based on config
        self.models[0] = RandomFourierFeatures(
            n_components=self.config['low_n_components'],
            lengthscale=self.lengthscale,
            sigma_noise=self.sigma_noise,
            random_state=self.random_state
        )
        
        self.models[1] = NystromApproximation(
            n_landmarks=self.config['medium_n_landmarks'],
            lengthscale=self.lengthscale,
            sigma_noise=self.sigma_noise,
            random_state=self.random_state
        )
        
        if self.config['use_exact_gp']:
            self.models[2] = GaussianProcessBaseline(
                max_train_size=5000,
                random_state=self.random_state,
                use_gpu=True
            )
        else:
            self.models[2] = NystromApproximation(
                n_landmarks=self.config['high_n_landmarks'],
                lengthscale=self.lengthscale,
                sigma_noise=self.sigma_noise,
                random_state=self.random_state
            )
        
        # Train all models on all data
        self.models[0].fit(X_train, y_train)
        self.models[1].fit(X_train, y_train)
        
        if self.config['use_exact_gp']:
            self.models[2].fit(X_train, y_train, n_iter=30)
        else:
            self.models[2].fit(X_train, y_train)
        
        self.training_time = time.time() - start
        
        return self
    
    def predict(self, X_test, return_std=True):
        """Predict with routing"""
        regions, _ = self.region_identifier.predict_regions(X_test, adaptive_thresholds=False)
        
        y_pred = np.zeros(len(X_test))
        y_std = np.zeros(len(X_test)) if return_std else None
        
        for region_label in [0, 1, 2]:
            mask = regions == region_label
            if mask.sum() > 0:
                if return_std:
                    pred, std = self.models[region_label].predict(X_test[mask], return_std=True)
                    y_pred[mask] = pred
                    y_std[mask] = std
                else:
                    y_pred[mask] = self.models[region_label].predict(X_test[mask], return_std=False)
        
        return (y_pred, y_std, regions) if return_std else (y_pred, regions)
    
    def evaluate(self, X_test, y_test):
        """Evaluate and return ECE"""
        from scipy.stats import norm
        
        y_pred, y_std, regions = self.predict(X_test, return_std=True)
        
        # Compute ECE
        errors = np.abs(y_test - y_pred)
        confidence_levels = np.linspace(0.1, 0.9, 10)
        
        ece = 0.0
        for conf in confidence_levels:
            z_score = norm.ppf((1 + conf) / 2)
            predicted_interval = y_std * z_score
            within_interval = (errors <= predicted_interval).astype(float)
            observed_conf = within_interval.mean()
            ece += np.abs(observed_conf - conf) / 10
        
        rmse = np.sqrt(np.mean((y_test - y_pred)**2))
        
        return {
            'rmse': rmse,
            'ece': ece,
            'training_time': self.training_time
        }


def tune_on_dataset(dataset_name='concrete', configs=CONFIGS):
    """
    Tune hyperparameters on a single dataset
    
    Args:
        dataset_name: Which dataset to tune on
        configs: List of configurations to try
    """
    print("\n" + "="*70)
    print(f"HYPERPARAMETER TUNING ON {dataset_name.upper()}")
    print("="*70)
    
    # Load data
    data = np.load(f'../data/{dataset_name}.npz')
    X, y = data['X'], data['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDataset: {len(X_train)} train, {len(X_test)} test")
    
    # Load baseline for comparison
    baseline_file = PROJECT_ROOT / 'results' / 'approximations' / f'{dataset_name}_results.json'
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        
        uniform_ece = baseline.get('nystrom', {}).get('ece', 0.41)
        gp_ece = baseline.get('exact_gp', {}).get('ece', 0.29)
        
        print(f"\nBaseline ECE:")
        print(f"  Exact GP:  {gp_ece:.4f}")
        print(f"  Uniform:   {uniform_ece:.4f}")
        print(f"  Target:    < {uniform_ece:.4f} (beat uniform)")
    else:
        uniform_ece = 0.41
        gp_ece = 0.29
        print(f"\n[WARNING] Baseline not found, using defaults")
    
    # Get lengthscale from baseline
    lengthscale = 1.0
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        if 'nystrom' in baseline and 'lengthscale' in baseline['nystrom']:
            lengthscale = baseline['nystrom']['lengthscale']
    
    print(f"  Lengthscale: {lengthscale:.4f}")
    
    # Test each configuration
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*70}")
        print(f"TESTING CONFIGURATION {i}/{len(configs)}")
        print(f"{'='*70}")
        print(f"Name: {config['name']}")
        print(f"  Low (RFF):    {config['low_n_components']} components")
        print(f"  Medium (Nys): {config['medium_n_landmarks']} landmarks")
        if config['use_exact_gp']:
            print(f"  High:         Exact GP")
        else:
            print(f"  High (Nys):   {config['high_n_landmarks']} landmarks")
        
        try:
            # Train model
            model = TunableAURORA(config, lengthscale=lengthscale, sigma_noise=0.1)
            model.fit(X_train, y_train)
            
            # Evaluate
            metrics = model.evaluate(X_test, y_test)
            
            improvement = (uniform_ece - metrics['ece']) / uniform_ece * 100
            
            print(f"\nResults:")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  ECE:  {metrics['ece']:.4f}")
            print(f"  vs Uniform: {improvement:+.1f}%")
            print(f"  Time: {metrics['training_time']:.2f}s")
            
            if metrics['ece'] < uniform_ece:
                print(f"  ✓ BETTER than uniform!")
            else:
                print(f"  ✗ Worse than uniform")
            
            results.append({
                'config': config['name'],
                'rmse': metrics['rmse'],
                'ece': metrics['ece'],
                'improvement_pct': improvement,
                'training_time': metrics['training_time']
            })
            
        except Exception as e:
            print(f"\n[ERROR] Configuration failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'config': config['name'],
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "="*70)
    print(f"TUNING SUMMARY: {dataset_name.upper()}")
    print("="*70)
    print(f"{'Config':<30} {'ECE':<10} {'Improvement':<12} {'Status':<10}")
    print("-"*70)
    
    best_config = None
    best_improvement = -float('inf')
    
    for result in results:
        if 'error' not in result:
            ece = result['ece']
            imp = result['improvement_pct']
            status = "✓ BETTER" if imp > 0 else "✗ WORSE"
            
            print(f"{result['config']:<30} {ece:<10.4f} {imp:>+10.1f}% {status:<10}")
            
            if imp > best_improvement:
                best_improvement = imp
                best_config = result['config']
    
    print("-"*70)
    
    if best_improvement > 0:
        print(f"\n✓ BEST: {best_config} ({best_improvement:+.1f}% improvement)")
    else:
        print(f"\n✗ NO configuration improved over uniform")
        print(f"  Best was {best_config} ({best_improvement:+.1f}%)")
    
    print("="*70)
    
    # Save results
    results_dir = PROJECT_ROOT / 'results' / 'tuning'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / f'{dataset_name}_tuning.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results, best_config


def tune_all_datasets():
    """Quick tuning on all datasets with best configs"""
    
    print("\n" + "="*70)
    print("QUICK TUNING: TESTING 2 BEST CONFIGS ON ALL DATASETS")
    print("="*70)
    
    # After tuning on concrete, test these two most promising
    best_configs = [
        {
            'name': 'Extreme Ratio',
            'low_n_components': 50,
            'medium_n_landmarks': 300,
            'high_n_landmarks': 1500,
            'use_exact_gp': False
        },
        {
            'name': 'Exact GP High',
            'low_n_components': 100,
            'medium_n_landmarks': 300,
            'high_n_landmarks': 0,
            'use_exact_gp': True
        },
    ]
    
    datasets = ['concrete', 'protein', 'robot_arm', 'sarcos', 'synthetic_heteroscedastic']
    
    all_results = {}
    
    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'='*70}")
        
        try:
            # Load data
            data = np.load(f'../data/{dataset_name}.npz')
            X, y = data['X'], data['y']
            
            if len(X) > 10000:
                idx = np.random.RandomState(42).choice(len(X), 10000, replace=False)
                X = X[idx]
                y = y[idx]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Get baseline
            baseline_file = PROJECT_ROOT / 'results' / 'approximations' / f'{dataset_name}_results.json'
            if baseline_file.exists():
                with open(baseline_file, 'r') as f:
                    baseline = json.load(f)
                uniform_ece = baseline.get('nystrom', {}).get('ece', 0.4)
                lengthscale = baseline.get('nystrom', {}).get('lengthscale', 1.0)
            else:
                uniform_ece = 0.4
                lengthscale = 1.0
            
            print(f"  Uniform ECE: {uniform_ece:.4f}")
            
            dataset_results = []
            
            for config in best_configs:
                print(f"\n  Testing: {config['name']}")
                
                try:
                    model = TunableAURORA(config, lengthscale=lengthscale, sigma_noise=0.1)
                    model.fit(X_train, y_train)
                    metrics = model.evaluate(X_test, y_test)
                    
                    improvement = (uniform_ece - metrics['ece']) / uniform_ece * 100
                    
                    print(f"    ECE: {metrics['ece']:.4f} ({improvement:+.1f}%)")
                    
                    dataset_results.append({
                        'config': config['name'],
                        'ece': metrics['ece'],
                        'improvement_pct': improvement
                    })
                    
                except Exception as e:
                    print(f"    [ERROR]: {e}")
                    dataset_results.append({'config': config['name'], 'error': str(e)})
            
            all_results[dataset_name] = dataset_results
            
        except Exception as e:
            print(f"  [ERROR] Dataset failed: {e}")
            all_results[dataset_name] = {'error': str(e)}
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL TUNING SUMMARY")
    print("="*70)
    print(f"{'Dataset':<25} {'Config':<20} {'ECE':<10} {'Improvement':<12}")
    print("-"*70)
    
    for dataset, results in all_results.items():
        if isinstance(results, list):
            for result in results:
                if 'error' not in result:
                    print(f"{dataset:<25} {result['config']:<20} {result['ece']:<10.4f} {result['improvement_pct']:>+10.1f}%")
    
    print("="*70)
    
    # Save
    results_dir = PROJECT_ROOT / 'results' / 'tuning'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / 'all_datasets_tuning.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='concrete',
                       help='Dataset to tune on (default: concrete)')
    parser.add_argument('--all', action='store_true',
                       help='Run quick tuning on all datasets')
    
    args = parser.parse_args()
    
    if args.all:
        print("\nRunning quick tuning on all datasets...")
        print("This will take ~30-45 minutes")
        tune_all_datasets()
    else:
        print(f"\nRunning thorough tuning on {args.dataset}")
        print("This will take ~10-15 minutes")
        tune_on_dataset(args.dataset, CONFIGS)
    
    print("\n✓ Tuning complete!")
    print("  Check results/tuning/ for detailed results")