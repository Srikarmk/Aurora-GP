"""
Quick test script to verify GP implementation works
Run this before full experiments to catch any issues early
"""

import numpy as np
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_gp_basic():
    """Test basic GP functionality on toy data"""
    print("\n" + "="*60)
    print("TEST 1: Basic GP Functionality")
    print("="*60)
    
    try:
        from gp_baseline import GaussianProcessBaseline
        
        # Generate simple 1D data
        np.random.seed(42)
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = np.sin(X.ravel()) + np.random.randn(100) * 0.1
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train GP
        print("\nTraining GP on 1D toy problem...")
        gp = GaussianProcessBaseline(max_train_size=100)
        gp.fit(X_train, y_train, n_iter=20)
        
        # Predict
        print("\nMaking predictions...")
        y_pred, y_std = gp.predict(X_test, return_std=True)
        
        # Evaluate
        metrics = gp.evaluate(X_test, y_test, verbose=True)
        
        # Check metrics are reasonable
        assert metrics['rmse'] < 1.0, "RMSE too high!"
        assert metrics['nll'] < 2.0, "NLL too high!"
        assert 0 < metrics['ece'] < 0.5, "ECE out of reasonable range!"
        
        print("\n[PASS] Test 1 PASSED: Basic GP works!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gp_2d():
    """Test GP on 2D data (for visualization testing)"""
    print("\n" + "="*60)
    print("TEST 2: 2D GP with Uncertainty Map")
    print("="*60)
    
    try:
        from gp_baseline import GaussianProcessBaseline
        
        # Generate 2D data
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 2)
        y = np.sin(X[:, 0]) + 0.5 * np.cos(X[:, 1]) + np.random.randn(n) * 0.1
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train
        print("\nTraining GP on 2D problem...")
        gp = GaussianProcessBaseline(max_train_size=200)
        gp.fit(X_train, y_train, n_iter=20)
        
        # Get uncertainty map
        print("\nComputing uncertainty map...")
        uncertainties = gp.get_uncertainty_map(X_test)
        
        assert uncertainties.shape == (len(X_test),), "Uncertainty shape mismatch!"
        assert np.all(uncertainties > 0), "Uncertainties should be positive!"
        
        print(f"\nUncertainty statistics:")
        print(f"  Mean: {uncertainties.mean():.4f}")
        print(f"  Std:  {uncertainties.std():.4f}")
        print(f"  Min:  {uncertainties.min():.4f}")
        print(f"  Max:  {uncertainties.max():.4f}")
        
        print("\n[PASS] Test 2 PASSED: 2D GP and uncertainty map work!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_load():
    """Test model saving and loading"""
    print("\n" + "="*60)
    print("TEST 3: Model Save/Load")
    print("="*60)
    
    try:
        from gp_baseline import GaussianProcessBaseline
        import tempfile
        import shutil
        
        # Generate data
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X[:, 0] + X[:, 1] + np.random.randn(50) * 0.1
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train
        print("\nTraining original GP...")
        gp1 = GaussianProcessBaseline(max_train_size=50)
        gp1.fit(X_train, y_train, n_iter=10)
        pred1, std1 = gp1.predict(X_test, return_std=True)
        
        # Save
        temp_dir = tempfile.mkdtemp()
        print(f"\nSaving to {temp_dir}...")
        gp1.save(temp_dir)
        
        # Load
        print("\nLoading GP...")
        gp2 = GaussianProcessBaseline()
        gp2.load(temp_dir)
        pred2, std2 = gp2.predict(X_test, return_std=True)
        
        # Check predictions match
        assert np.allclose(pred1, pred2, atol=1e-5), "Predictions don't match!"
        assert np.allclose(std1, std2, atol=1e-5), "Uncertainties don't match!"
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print("\n[PASS] Test 3 PASSED: Save/load works correctly!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_large_dataset():
    """Test GP handles larger datasets with subsampling"""
    print("\n" + "="*60)
    print("TEST 4: Large Dataset Handling")
    print("="*60)
    
    try:
        from gp_baseline import GaussianProcessBaseline
        
        # Generate large dataset
        np.random.seed(42)
        n = 10000
        X = np.random.randn(n, 5)
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.2
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"\nDataset: {len(X_train)} train, {len(X_test)} test")
        print("Testing subsampling to 2000 points...")
        
        # Train with subsampling
        gp = GaussianProcessBaseline(max_train_size=2000)
        gp.fit(X_train, y_train, n_iter=30)
        
        print(f"\nActual training size: {gp.n_train}")
        assert gp.n_train == 2000, "Subsampling didn't work!"
        
        # Test on large test set
        print(f"\nPredicting on {len(X_test)} test points...")
        metrics = gp.evaluate(X_test, y_test, verbose=True)
        
        print("\n[PASS] Test 4 PASSED: Large dataset handling works!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization functions"""
    print("\n" + "="*60)
    print("TEST 5: Visualization Functions")
    print("="*60)
    
    try:
        from gp_visualization import GPVisualizer
        from gp_baseline import GaussianProcessBaseline
        import tempfile
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        # Generate 2D data
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.sin(X[:, 0]) + np.random.randn(100) * 0.1
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train GP
        gp = GaussianProcessBaseline(max_train_size=100)
        gp.fit(X_train, y_train, n_iter=10)
        y_pred, y_std = gp.predict(X_test, return_std=True)
        
        # Create visualizer
        visualizer = GPVisualizer()
        
        # Test plots (don't save, just create)
        print("\nTesting prediction plot...")
        fig1 = visualizer.plot_predictions_vs_true('test', X_test, y_test, y_pred, y_std)
        assert fig1 is not None
        
        print("Testing calibration plot...")
        fig2 = visualizer.plot_calibration_curve(y_test, y_pred, y_std, 'test')
        assert fig2 is not None
        
        print("Testing uncertainty map (2D)...")
        fig3 = visualizer.plot_uncertainty_map_2d(gp, X_train, X_test, y_test, 'test')
        assert fig3 is not None
        
        print("\n[PASS] Test 5 PASSED: Visualization functions work!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Test 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_datasets():
    """Test GP on all .npz datasets in the data directory"""
    print("\n" + "="*60)
    print("TEST 6: All .npz Datasets")
    print("="*60)
    
    try:
        from gp_baseline import GaussianProcessBaseline
        
        # Find all .npz files in data directory
        data_dir = Path(__file__).parent.parent / 'data'
        npz_files = list(data_dir.glob('*.npz'))
        
        if not npz_files:
            print(f"\n[WARN] No .npz files found in {data_dir}")
            return False
        
        print(f"\nFound {len(npz_files)} dataset(s):")
        for f in npz_files:
            print(f"  - {f.stem}")
        
        results = {}
        
        for npz_file in npz_files:
            dataset_name = npz_file.stem
            print(f"\n{'='*60}")
            print(f"Testing on: {dataset_name}")
            print(f"{'='*60}")
            
            try:
                # Load dataset
                data = np.load(npz_file, allow_pickle=True)
                X = data['X']
                y = data['y']
                
                print(f"Dataset shape: X={X.shape}, y={y.shape}")
                
                # Use smaller subset for testing (max 2000 points)
                max_test_size = min(2000, len(X))
                if len(X) > max_test_size:
                    idx = np.random.RandomState(42).choice(len(X), max_test_size, replace=False)
                    X = X[idx]
                    y = y[idx]
                    print(f"Using subset: {X.shape[0]} samples")
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Train GP (with fewer iterations for testing)
                print("\nTraining GP...")
                gp = GaussianProcessBaseline(max_train_size=min(1000, len(X_train)), random_state=42)
                gp.fit(X_train, y_train, n_iter=20)
                
                # Evaluate
                print("\nEvaluating...")
                metrics = gp.evaluate(X_test, y_test, verbose=False)
                
                results[dataset_name] = {
                    'rmse': metrics['rmse'],
                    'nll': metrics['nll'],
                    'ece': metrics['ece'],
                    'training_time': metrics.get('training_time', 0),
                    'n_train': gp.n_train,
                    'n_test': len(X_test)
                }
                
                print(f"\n[OK] {dataset_name}: RMSE={metrics['rmse']:.4f}, NLL={metrics['nll']:.4f}, ECE={metrics['ece']:.4f}")
                
            except Exception as e:
                print(f"\n[FAIL] Error testing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                results[dataset_name] = {'error': str(e)}
        
        # Summary
        print("\n" + "="*60)
        print("DATASET TEST SUMMARY")
        print("="*60)
        print(f"{'Dataset':<25} {'RMSE':<10} {'NLL':<10} {'ECE':<10} {'Status':<10}")
        print("-"*60)
        
        all_passed = True
        for name, res in results.items():
            if 'error' in res:
                print(f"{name:<25} {'ERROR':<10}")
                all_passed = False
            else:
                print(f"{name:<25} {res['rmse']:<10.4f} {res['nll']:<10.4f} {res['ece']:<10.4f} {'OK':<10}")
        
        if all_passed:
            print("\n[PASS] Test 6 PASSED: All datasets tested successfully!")
            return True
        else:
            print("\n[FAIL] Test 6 FAILED: Some datasets had errors")
            return False
        
    except Exception as e:
        print(f"\n[FAIL] Test 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_datasets():
    """Test GP on all .npz datasets in the data directory"""
    print("\n" + "="*60)
    print("TEST 6: All .npz Datasets")
    print("="*60)
    
    try:
        from gp_baseline import GaussianProcessBaseline
        
        # Find all .npz files in data directory
        data_dir = Path(__file__).parent.parent / 'data'
        npz_files = list(data_dir.glob('*.npz'))
        
        if not npz_files:
            print(f"\n[WARN] No .npz files found in {data_dir}")
            return False
        
        print(f"\nFound {len(npz_files)} dataset(s):")
        for f in npz_files:
            print(f"  - {f.stem}")
        
        results = {}
        
        for npz_file in npz_files:
            dataset_name = npz_file.stem
            print(f"\n{'='*60}")
            print(f"Testing on: {dataset_name}")
            print(f"{'='*60}")
            
            try:
                # Load dataset
                data = np.load(npz_file, allow_pickle=True)
                X = data['X']
                y = data['y']
                
                print(f"Dataset shape: X={X.shape}, y={y.shape}")
                
                # Use smaller subset for testing (max 2000 points)
                max_test_size = min(2000, len(X))
                if len(X) > max_test_size:
                    idx = np.random.RandomState(42).choice(len(X), max_test_size, replace=False)
                    X = X[idx]
                    y = y[idx]
                    print(f"Using subset: {X.shape[0]} samples")
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Train GP (with fewer iterations for testing)
                print("\nTraining GP...")
                gp = GaussianProcessBaseline(max_train_size=min(1000, len(X_train)), random_state=42)
                gp.fit(X_train, y_train, n_iter=20)
                
                # Evaluate
                print("\nEvaluating...")
                metrics = gp.evaluate(X_test, y_test, verbose=False)
                
                results[dataset_name] = {
                    'rmse': metrics['rmse'],
                    'nll': metrics['nll'],
                    'ece': metrics['ece'],
                    'training_time': metrics.get('training_time', 0),
                    'n_train': gp.n_train,
                    'n_test': len(X_test)
                }
                
                print(f"\n[OK] {dataset_name}: RMSE={metrics['rmse']:.4f}, NLL={metrics['nll']:.4f}, ECE={metrics['ece']:.4f}")
                
            except Exception as e:
                print(f"\n[FAIL] Error testing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                results[dataset_name] = {'error': str(e)}
        
        # Summary
        print("\n" + "="*60)
        print("DATASET TEST SUMMARY")
        print("="*60)
        print(f"{'Dataset':<25} {'RMSE':<10} {'NLL':<10} {'ECE':<10} {'Status':<10}")
        print("-"*60)
        
        all_passed = True
        for name, res in results.items():
            if 'error' in res:
                print(f"{name:<25} {'ERROR':<10}")
                all_passed = False
            else:
                print(f"{name:<25} {res['rmse']:<10.4f} {res['nll']:<10.4f} {res['ece']:<10.4f} {'OK':<10}")
        
        if all_passed:
            print("\n[PASS] Test 6 PASSED: All datasets tested successfully!")
            return True
        else:
            print("\n[FAIL] Test 6 FAILED: Some datasets had errors")
            return False
        
    except Exception as e:
        print(f"\n[FAIL] Test 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("AURORA GP BASELINE - TEST SUITE")
    print("="*70)
    print("\nRunning 6 tests to verify GP implementation...")
    
    tests = [
        ("Basic GP Functionality", test_gp_basic),
        ("2D GP with Uncertainty", test_gp_2d),
        ("Model Save/Load", test_save_load),
        ("Large Dataset Handling", test_large_dataset),
        ("Visualization Functions", test_visualization),
        ("All .npz Datasets", test_all_datasets),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[WARN]  Test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "[PASS] PASSED" if passed else "[FAIL] FAILED"
        print(f"{status}: {name}")
    
    n_passed = sum(1 for _, passed in results if passed)
    n_total = len(results)
    
    print(f"\n{n_passed}/{n_total} tests passed")
    
    if n_passed == n_total:
        print("\n[SUCCESS] All tests passed!")
        return True
    else:
        print("\n[WARN]  Some tests failed. Please fix issues before proceeding.")
        print("\nCommon issues:")
        print("  - GPyTorch not installed: pip install gpytorch")
        print("  - PyTorch not installed: pip install torch")
        print("  - CUDA issues: Install CPU-only version or check CUDA setup")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)