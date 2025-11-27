"""
AURORA - Step 3: Benchmark Approximation Methods
Compare Exact GP, RFF, and Nyström on all datasets
"""

import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent

from gp_baseline import GaussianProcessBaseline
from approximations import RandomFourierFeatures, NystromApproximation


def benchmark_single_dataset(dataset_name, X, y, 
                            n_rff_components=1000,
                            n_nystrom_landmarks=500,
                            max_gp_size=5000,
                            test_size=0.2,
                            random_state=42):
    """Benchmark all three methods on a single dataset"""
    
    print("\n" + "="*70)
    print(f"BENCHMARKING: {dataset_name.upper()}")
    print("="*70)
    print(f"Dataset size: {len(X)} samples, {X.shape[1]} features")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = {}
    
    # =========================================================================
    # METHOD 1: EXACT GP (Baseline)
    # =========================================================================
    print("\n" + "-"*70)
    print("METHOD 1: EXACT GP")
    print("-"*70)
    
    try:
        gp = GaussianProcessBaseline(max_train_size=max_gp_size, random_state=random_state)
        gp.fit(X_train, y_train, n_iter=50)
        gp_metrics = gp.evaluate(X_test, y_test, verbose=False)
        
        results['exact_gp'] = {
            'rmse': gp_metrics['rmse'],
            'nll': gp_metrics['nll'],
            'ece': gp_metrics['ece'],
            'training_time': gp.training_time,
            'inference_time': gp_metrics['inference_time'],
            'n_train': gp.n_train
        }
        
        print(f"[OK] Exact GP Complete")
        print(f"  RMSE: {gp_metrics['rmse']:.4f}")
        print(f"  NLL:  {gp_metrics['nll']:.4f}")
        print(f"  ECE:  {gp_metrics['ece']:.4f}")
        print(f"  Time: {gp.training_time:.2f}s")
        
    except Exception as e:
        print(f"[ERROR] Exact GP Failed: {e}")
        results['exact_gp'] = {'error': str(e)}
    
    # =========================================================================
    # METHOD 2: RANDOM FOURIER FEATURES
    # =========================================================================
    print("\n" + "-"*70)
    print("METHOD 2: RANDOM FOURIER FEATURES (RFF)")
    print("-"*70)
    
    try:
        # Estimate lengthscale from data (simple heuristic)
        from scipy.spatial.distance import pdist
        if len(X_train) > 1000:
            sample_idx = np.random.choice(len(X_train), 1000, replace=False)
            distances = pdist(X_train[sample_idx])
        else:
            distances = pdist(X_train)
        lengthscale = np.median(distances)
        
        rff = RandomFourierFeatures(
            n_components=n_rff_components,
            lengthscale=lengthscale,
            sigma_noise=0.1,
            random_state=random_state
        )
        rff.fit(X_train, y_train)
        rff_metrics = rff.evaluate(X_test, y_test, verbose=False)
        
        results['rff'] = {
            'rmse': rff_metrics['rmse'],
            'nll': rff_metrics['nll'],
            'ece': rff_metrics['ece'],
            'training_time': rff.training_time,
            'inference_time': rff_metrics['inference_time'],
            'n_components': n_rff_components,
            'lengthscale': lengthscale
        }
        
        print(f"[OK] RFF Complete (D={n_rff_components})")
        print(f"  RMSE: {rff_metrics['rmse']:.4f}")
        print(f"  NLL:  {rff_metrics['nll']:.4f}")
        print(f"  ECE:  {rff_metrics['ece']:.4f}")
        print(f"  Time: {rff.training_time:.2f}s")
        
    except Exception as e:
        print(f"[ERROR] RFF Failed: {e}")
        results['rff'] = {'error': str(e)}
    
    # =========================================================================
    # METHOD 3: NYSTRÖM APPROXIMATION
    # =========================================================================
    print("\n" + "-"*70)
    print("METHOD 3: NYSTRÖM APPROXIMATION")
    print("-"*70)
    
    try:
        nystrom = NystromApproximation(
            n_landmarks=n_nystrom_landmarks,
            lengthscale=lengthscale,
            sigma_noise=0.1,
            random_state=random_state
        )
        nystrom.fit(X_train, y_train)
        nystrom_metrics = nystrom.evaluate(X_test, y_test, verbose=False)
        
        results['nystrom'] = {
            'rmse': nystrom_metrics['rmse'],
            'nll': nystrom_metrics['nll'],
            'ece': nystrom_metrics['ece'],
            'training_time': nystrom.training_time,
            'inference_time': nystrom_metrics['inference_time'],
            'n_landmarks': n_nystrom_landmarks,
            'lengthscale': lengthscale
        }
        
        print(f"[OK] Nyström Complete (m={n_nystrom_landmarks})")
        print(f"  RMSE: {nystrom_metrics['rmse']:.4f}")
        print(f"  NLL:  {nystrom_metrics['nll']:.4f}")
        print(f"  ECE:  {nystrom_metrics['ece']:.4f}")
        print(f"  Time: {nystrom.training_time:.2f}s")
        
    except Exception as e:
        print(f"[ERROR] Nyström Failed: {e}")
        results['nystrom'] = {'error': str(e)}
    
    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print(f"RESULTS SUMMARY: {dataset_name}")
    print("="*70)
    print(f"{'Method':<15} {'RMSE':<10} {'NLL':<10} {'ECE':<10} {'Time(s)':<10} {'Speedup':<10}")
    print("-"*70)
    
    gp_time = results['exact_gp'].get('training_time', np.inf)
    
    for method_name, method_results in results.items():
        if 'error' not in method_results:
            rmse = method_results['rmse']
            nll = method_results['nll']
            ece = method_results['ece']
            time_val = method_results['training_time']
            speedup = gp_time / time_val if time_val > 0 else 0
            
            display_name = {
                'exact_gp': 'Exact GP',
                'rff': 'RFF',
                'nystrom': 'Nyström'
            }[method_name]
            
            print(f"{display_name:<15} {rmse:<10.4f} {nll:<10.4f} {ece:<10.4f} {time_val:<10.2f} {speedup:<10.1f}x")
    
    print("="*70)
    
    return results


def run_all_benchmarks(data_dir=None, 
                       results_dir=None,
                       n_rff_components=1000,
                       n_nystrom_landmarks=500):
    """Run benchmarks on all datasets"""
    
    if data_dir is None:
        data_dir = PROJECT_ROOT / 'data'
    if results_dir is None:
        results_dir = PROJECT_ROOT / 'results' / 'approximations'
    
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all datasets
    datasets = list(data_dir.glob('*.npz'))
    
    if not datasets:
        print(f"\n[ERROR] No datasets found in {data_dir}")
        return None
    
    print(f"\nFound {len(datasets)} dataset(s)")
    
    all_results = {}
    
    for dataset_file in datasets:
        dataset_name = dataset_file.stem
        
        try:
            # Load dataset
            data = np.load(dataset_file, allow_pickle=True)
            X = data['X']
            y = data['y']
            
            # Run benchmark
            results = benchmark_single_dataset(
                dataset_name, X, y,
                n_rff_components=n_rff_components,
                n_nystrom_landmarks=n_nystrom_landmarks
            )
            
            all_results[dataset_name] = results
            
            # Save individual results
            with open(results_dir / f'{dataset_name}_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
        except Exception as e:
            print(f"\n[ERROR] Error processing {dataset_name}: {e}")
            all_results[dataset_name] = {'error': str(e)}
    
    # Save summary
    with open(results_dir / 'summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - ALL DATASETS")
    print("="*70)
    print(f"{'Dataset':<20} {'Method':<12} {'RMSE':<10} {'NLL':<10} {'ECE':<10}")
    print("-"*70)
    
    for dataset_name, results in all_results.items():
        if 'error' not in results:
            for method in ['exact_gp', 'rff', 'nystrom']:
                if method in results and 'error' not in results[method]:
                    m = results[method]
                    method_name = {'exact_gp': 'GP', 'rff': 'RFF', 'nystrom': 'Nyström'}[method]
                    print(f"{dataset_name:<20} {method_name:<12} {m['rmse']:<10.4f} {m['nll']:<10.4f} {m['ece']:<10.4f}")
    
    print("="*70)
    print(f"\n[OK] Results saved to {results_dir}")
    
    return all_results


if __name__ == "__main__":
    results = run_all_benchmarks()