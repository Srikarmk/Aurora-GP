"""
Diagnose why AURORA isn't improving results
"""

import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from region_aware import RegionAwareApproximation


def diagnose_concrete():
    """Detailed diagnosis on Concrete dataset"""
    
    print("\n" + "="*70)
    print("DIAGNOSING AURORA ON CONCRETE DATASET")
    print("="*70)
    
    # Load data
    data = np.load('../data/concrete.npz')
    X, y = data['X'], data['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDataset sizes:")
    print(f"  Total: {len(X)}")
    print(f"  Train: {len(X_train)}")
    print(f"  Test:  {len(X_test)}")
    
    # Create AURORA model
    print("\n" + "-"*70)
    print("CREATING AURORA MODEL")
    print("-"*70)
    
    aurora = RegionAwareApproximation(
        low_n_components=200,
        medium_n_landmarks=500,
        high_n_landmarks=1000,
        lengthscale=1.0,
        sigma_noise=0.1,
        random_state=42
    )
    
    print(f"\nModel configuration:")
    print(f"  Low (RFF):      {aurora.low_n_components} components")
    print(f"  Medium (Nyström): {aurora.medium_n_landmarks} landmarks")
    print(f"  High (Nyström):   {aurora.high_n_landmarks} landmarks")
    
    # Fit with diagnostics
    print("\n" + "-"*70)
    print("FITTING AURORA")
    print("-"*70)
    
    aurora.fit(X_train, y_train, n_gp_iter=20)
    
    # DIAGNOSTIC 1: Check if models were created
    print("\n" + "-"*70)
    print("DIAGNOSTIC 1: Model Creation")
    print("-"*70)
    
    print(f"\nNumber of models created: {len(aurora.models)}")
    
    for region_id in [0, 1, 2]:
        if region_id in aurora.models:
            model = aurora.models[region_id]
            print(f"\nRegion {region_id}:")
            print(f"  Type: {type(model).__name__}")
            
            if hasattr(model, 'n_components'):
                print(f"  RFF components: {model.n_components}")
            
            if hasattr(model, 'n_landmarks'):
                print(f"  Nyström landmarks: {model.n_landmarks}")
            
            if hasattr(model, 'n_train'):
                print(f"  GP training size: {model.n_train}")
            
            # Check if model was actually fitted
            if hasattr(model, 'scaler_X') and hasattr(model.scaler_X, 'mean_'):
                print(f"  ✓ Model fitted (scaler has {len(model.scaler_X.mean_)} features)")
            else:
                print(f"  ❌ Model NOT fitted!")
        else:
            print(f"\n❌ Region {region_id}: No model created!")
    
    # DIAGNOSTIC 2: Test region prediction
    print("\n" + "-"*70)
    print("DIAGNOSTIC 2: Test Region Prediction")
    print("-"*70)
    
    y_pred, y_std, regions_test = aurora.predict(X_test, return_std=True)
    
    print(f"\nTest region distribution:")
    print(f"  Low (0):    {np.sum(regions_test == 0)} ({np.sum(regions_test == 0)/len(X_test)*100:.1f}%)")
    print(f"  Medium (1): {np.sum(regions_test == 1)} ({np.sum(regions_test == 1)/len(X_test)*100:.1f}%)")
    print(f"  High (2):   {np.sum(regions_test == 2)} ({np.sum(regions_test == 2)/len(X_test)*100:.1f}%)")
    
    # DIAGNOSTIC 3: Check predictions per region
    print("\n" + "-"*70)
    print("DIAGNOSTIC 3: Predictions Per Region")
    print("-"*70)
    
    for region_id in [0, 1, 2]:
        mask = regions_test == region_id
        if mask.sum() > 0:
            region_pred = y_pred[mask]
            region_std = y_std[mask]
            region_true = y_test[mask]
            
            rmse = np.sqrt(np.mean((region_pred - region_true)**2))
            mean_std = region_std.mean()
            
            print(f"\nRegion {region_id} ({mask.sum()} points):")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  Mean uncertainty: {mean_std:.4f}")
            print(f"  Predictions range: [{region_pred.min():.2f}, {region_pred.max():.2f}]")
    
    # DIAGNOSTIC 4: Compare with uniform Nyström behavior
    print("\n" + "-"*70)
    print("DIAGNOSTIC 4: Comparison Check")
    print("-"*70)
    
    # If all regions have similar performance, models aren't specialized
    errors_by_region = {}
    for region_id in [0, 1, 2]:
        mask = regions_test == region_id
        if mask.sum() > 0:
            errors = np.abs(y_pred[mask] - y_test[mask])
            errors_by_region[region_id] = errors.mean()
    
    print(f"\nMean error by region:")
    for region_id, error in errors_by_region.items():
        print(f"  Region {region_id}: {error:.4f}")
    
    # If errors are very similar, models aren't specialized
    error_std = np.std(list(errors_by_region.values()))
    print(f"\nError variation across regions: {error_std:.4f}")
    if error_std < 0.5:
        print("  ⚠️ Very low variation - models may not be specialized!")
    else:
        print("  ✓ Good variation - models are different")
    
    # DIAGNOSTIC 5: Final metrics
    print("\n" + "-"*70)
    print("DIAGNOSTIC 5: Final Metrics")
    print("-"*70)
    
    metrics = aurora.evaluate(X_test, y_test, verbose=True)
    
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)
    
    # Key checks
    print("\nKey Checks:")
    checks_passed = 0
    
    # Check 1: Models exist
    if len(aurora.models) == 3:
        print("  ✓ All 3 models created")
        checks_passed += 1
    else:
        print(f"  ❌ Only {len(aurora.models)} models created (need 3)")
    
    # Check 2: Different model types
    types = [type(aurora.models[i]).__name__ for i in [0, 1, 2] if i in aurora.models]
    if 'RandomFourierFeatures' in types and 'NystromApproximation' in types:
        print(f"  ✓ Different model types: {set(types)}")
        checks_passed += 1
    else:
        print(f"  ❌ Wrong model types: {types}")
    
    # Check 3: ECE not too low
    if metrics['overall']['ece'] > 0.15:
        print(f"  ✓ ECE realistic ({metrics['overall']['ece']:.4f} > 0.15)")
        checks_passed += 1
    else:
        print(f"  ❌ ECE too low ({metrics['overall']['ece']:.4f} < 0.15)")
    
    # Check 4: Test regions distributed
    if np.sum(regions_test == 0) > 0 and np.sum(regions_test == 1) > 0 and np.sum(regions_test == 2) > 0:
        print(f"  ✓ All regions have test points")
        checks_passed += 1
    else:
        print(f"  ❌ Some regions have no test points")
    
    print(f"\nPassed {checks_passed}/4 checks")
    
    if checks_passed == 4:
        print("\n✅ Diagnostics look good - implementation seems correct")
        print("   If ECE still not improving, may be dataset-specific issue")
    else:
        print("\n❌ Diagnostics show issues - implementation has bugs")


if __name__ == "__main__":
    diagnose_concrete()