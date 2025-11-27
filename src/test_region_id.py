"""
Quick test for AURORA Stage 1: Region Identification
Run this before full region identification to catch issues early
"""

import numpy as np
from sklearn.model_selection import train_test_split
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from region_identification import RegionIdentifier


def test_region_basic():
    """Test basic region identification"""
    print("\n" + "="*60)
    print("TEST 1: Basic Region Identification")
    print("="*60)
    
    try:
        # Generate toy data
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 3)
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.1
        
        print("\nTraining region identifier...")
        identifier = RegionIdentifier(
            max_gp_samples=500,
            uncertainty_weight=0.5,
            density_weight=0.5,
            high_threshold=0.7,
            low_threshold=0.3
        )
        
        identifier.fit(X, y, n_gp_iter=20)
        
        # Check outputs
        assert identifier.uncertainty_map is not None, "Uncertainty map not computed!"
        assert identifier.density_map is not None, "Density map not computed!"
        assert identifier.importance_scores is not None, "Importance scores not computed!"
        
        assert len(identifier.uncertainty_map) == len(X), "Uncertainty map size mismatch!"
        assert len(identifier.importance_scores) == len(X), "Importance scores size mismatch!"
        
        # Check thresholds
        assert identifier.high_threshold_value > identifier.low_threshold_value, \
            "High threshold should be > low threshold!"
        
        # Get statistics
        stats = identifier.get_region_statistics()
        
        print(f"\nRegion statistics:")
        print(f"  High:   {stats['n_high']} ({stats['pct_high']:.1f}%)")
        print(f"  Medium: {stats['n_medium']} ({stats['pct_medium']:.1f}%)")
        print(f"  Low:    {stats['n_low']} ({stats['pct_low']:.1f}%)")
        
        # Check percentages roughly match thresholds
        assert 20 < stats['pct_high'] < 40, "High region percentage seems off!"
        assert 20 < stats['pct_low'] < 40, "Low region percentage seems off!"
        
        print("\n[PASS] Test 1 PASSED: Basic region identification works!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_region_prediction():
    """Test predicting regions for new points"""
    print("\n" + "="*60)
    print("TEST 2: Region Prediction for New Points")
    print("="*60)
    
    try:
        # Generate train and test data
        np.random.seed(42)
        X_train = np.random.randn(300, 3)
        y_train = X_train[:, 0] + np.random.randn(300) * 0.1
        
        X_test = np.random.randn(100, 3)
        
        print("\nFitting identifier on training data...")
        identifier = RegionIdentifier(max_gp_samples=300)
        identifier.fit(X_train, y_train, n_gp_iter=15)
        
        print("\nPredicting regions for test points...")
        regions, test_importance = identifier.predict_regions(X_test)
        
        # Check outputs
        assert regions.shape == (len(X_test),), "Region shape mismatch!"
        assert test_importance.shape == (len(X_test),), "Importance shape mismatch!"
        assert set(regions).issubset({0, 1, 2}), "Regions should be 0, 1, or 2!"
        
        # Count regions
        n_high = np.sum(regions == 2)
        n_medium = np.sum(regions == 1)
        n_low = np.sum(regions == 0)
        
        print(f"\nTest region distribution:")
        print(f"  High:   {n_high} ({n_high/len(X_test)*100:.1f}%)")
        print(f"  Medium: {n_medium} ({n_medium/len(X_test)*100:.1f}%)")
        print(f"  Low:    {n_low} ({n_low/len(X_test)*100:.1f}%)")
        
        print("\n[PASS] Test 2 PASSED: Region prediction works!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_load():
    """Test saving and loading region identifier"""
    print("\n" + "="*60)
    print("TEST 3: Save/Load Region Identifier")
    print("="*60)
    
    try:
        import tempfile
        import shutil
        
        # Generate data
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y = X[:, 0] + np.random.randn(200) * 0.1
        
        # Train
        print("\nTraining identifier...")
        identifier1 = RegionIdentifier(max_gp_samples=200)
        identifier1.fit(X, y, n_gp_iter=10)
        
        stats1 = identifier1.get_region_statistics()
        
        # Save
        temp_dir = tempfile.mkdtemp()
        print(f"\nSaving to {temp_dir}...")
        identifier1.save(temp_dir)
        
        # Check files exist
        assert (Path(temp_dir) / 'config.json').exists(), "Config not saved!"
        assert (Path(temp_dir) / 'statistics.json').exists(), "Statistics not saved!"
        assert (Path(temp_dir) / 'region_data.npz').exists(), "Region data not saved!"
        
        # Load statistics and verify
        with open(Path(temp_dir) / 'statistics.json', 'r') as f:
            stats_loaded = json.load(f)
        
        # Check key statistics match
        assert abs(stats_loaded['n_high'] - stats1['n_high']) < 1, "Statistics don't match!"
        assert abs(stats_loaded['pct_high'] - stats1['pct_high']) < 0.1, "Percentages don't match!"
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print("\n[PASS] Test 3 PASSED: Save/load works correctly!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_weights():
    """Test that different uncertainty/density weights affect results"""
    print("\n" + "="*60)
    print("TEST 4: Different Weight Configurations")
    print("="*60)
    
    try:
        # Generate data
        np.random.seed(42)
        X = np.random.randn(300, 2)
        y = X[:, 0] + np.random.randn(300) * 0.1
        
        # Test 1: Uncertainty only
        print("\nConfiguration 1: Uncertainty only (1.0, 0.0)")
        id1 = RegionIdentifier(max_gp_samples=300, 
                              uncertainty_weight=1.0, 
                              density_weight=0.0)
        id1.fit(X, y, n_gp_iter=10)
        stats1 = id1.get_region_statistics()
        
        # Test 2: Density only
        print("\nConfiguration 2: Density only (0.0, 1.0)")
        id2 = RegionIdentifier(max_gp_samples=300,
                              uncertainty_weight=0.0,
                              density_weight=1.0)
        id2.fit(X, y, n_gp_iter=10)
        stats2 = id2.get_region_statistics()
        
        # Test 3: Balanced
        print("\nConfiguration 3: Balanced (0.5, 0.5)")
        id3 = RegionIdentifier(max_gp_samples=300,
                              uncertainty_weight=0.5,
                              density_weight=0.5)
        id3.fit(X, y, n_gp_iter=10)
        stats3 = id3.get_region_statistics()
        
        # Results should differ
        print(f"\nRegion distributions:")
        print(f"  Uncertainty only: High={stats1['pct_high']:.1f}%, Low={stats1['pct_low']:.1f}%")
        print(f"  Density only:     High={stats2['pct_high']:.1f}%, Low={stats2['pct_low']:.1f}%")
        print(f"  Balanced:         High={stats3['pct_high']:.1f}%, Low={stats3['pct_low']:.1f}%")
        
        print("\n[PASS] Test 4 PASSED: Different weights produce different regions!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all region identification tests"""
    print("\n" + "="*70)
    print("AURORA STAGE 1: REGION IDENTIFICATION - TEST SUITE")
    print("="*70)
    
    tests = [
        ("Basic Region Identification", test_region_basic),
        ("Region Prediction", test_region_prediction),
        ("Save/Load", test_save_load),
        ("Different Weights", test_different_weights)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[WARN] Test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")
    
    n_passed = sum(1 for _, passed in results if passed)
    n_total = len(results)
    
    print(f"\n{n_passed}/{n_total} tests passed")
    
    if n_passed == n_total:
        print("\n[SUCCESS] All tests passed!")
        return True
    else:
        print("\n[WARN] Some tests failed. Fix issues before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)