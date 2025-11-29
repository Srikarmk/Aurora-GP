"""
Diagnose test region classification
Check if all test points are being classified as same region
"""

import numpy as np
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from region_identification import RegionIdentifier


def diagnose_test_regions():
    """Check how test points are being classified"""
    
    print("\n" + "="*70)
    print("DIAGNOSING TEST REGION CLASSIFICATION")
    print("="*70)
    
    # Load concrete
    data = np.load('../data/concrete.npz')
    X, y = data['X'], data['y']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nDataset: Concrete")
    print(f"  Train: {len(X_train)}")
    print(f"  Test:  {len(X_test)}")
    
    # Fit region identifier
    print("\nFitting region identifier...")
    identifier = RegionIdentifier(random_state=42)
    identifier.fit(X_train, y_train, n_gp_iter=20)
    
    print("\n" + "-"*70)
    print("TRAINING DATA REGIONS (using stored scores)")
    print("-"*70)
    
    importance_train = identifier.importance_scores
    high_thresh = identifier.high_threshold_value
    low_thresh = identifier.low_threshold_value
    
    print(f"\nThresholds from training:")
    print(f"  High: {high_thresh:.4f}")
    print(f"  Low:  {low_thresh:.4f}")
    
    print(f"\nTraining importance scores:")
    print(f"  Min:  {importance_train.min():.4f}")
    print(f"  Mean: {importance_train.mean():.4f}")
    print(f"  Max:  {importance_train.max():.4f}")
    
    regions_train_direct = np.ones(len(X_train), dtype=int)
    regions_train_direct[importance_train >= high_thresh] = 2
    regions_train_direct[importance_train <= low_thresh] = 0
    
    print(f"\nTraining regions (direct from scores):")
    print(f"  Low:    {np.sum(regions_train_direct == 0)} ({np.sum(regions_train_direct == 0)/len(X_train)*100:.1f}%)")
    print(f"  Medium: {np.sum(regions_train_direct == 1)} ({np.sum(regions_train_direct == 1)/len(X_train)*100:.1f}%)")
    print(f"  High:   {np.sum(regions_train_direct == 2)} ({np.sum(regions_train_direct == 2)/len(X_train)*100:.1f}%)")
    
    # Now test both methods on TEST data
    print("\n" + "-"*70)
    print("TEST DATA REGIONS")
    print("-"*70)
    
    # Method 1: Using predict_regions with adaptive_thresholds=False
    print("\nMethod 1: predict_regions(X_test, adaptive_thresholds=False)")
    regions_test_fixed, importance_test = identifier.predict_regions(X_test, adaptive_thresholds=False)
    
    print(f"\nTest importance scores:")
    print(f"  Min:  {importance_test.min():.4f}")
    print(f"  Mean: {importance_test.mean():.4f}")
    print(f"  Max:  {importance_test.max():.4f}")
    
    print(f"\nUsing training thresholds (high={high_thresh:.4f}, low={low_thresh:.4f}):")
    print(f"  Low:    {np.sum(regions_test_fixed == 0)} ({np.sum(regions_test_fixed == 0)/len(X_test)*100:.1f}%)")
    print(f"  Medium: {np.sum(regions_test_fixed == 1)} ({np.sum(regions_test_fixed == 1)/len(X_test)*100:.1f}%)")
    print(f"  High:   {np.sum(regions_test_fixed == 2)} ({np.sum(regions_test_fixed == 2)/len(X_test)*100:.1f}%)")
    
    # Check if all in one region
    if np.sum(regions_test_fixed == 0) == 0 and np.sum(regions_test_fixed == 1) == 0:
        print("\n  🚨 ALL TEST POINTS CLASSIFIED AS HIGH!")
        print("  This explains why AURORA = uniform Nyström")
    elif np.sum(regions_test_fixed == 2) == 0:
        print("\n  🚨 NO TEST POINTS CLASSIFIED AS HIGH!")
    else:
        print("\n  ✓ Test points distributed across regions")
    
    # Method 2: Check what adaptive does
    print("\n\nMethod 2: predict_regions(X_test, adaptive_thresholds=True)")
    regions_test_adaptive, _ = identifier.predict_regions(X_test, adaptive_thresholds=True)
    
    print(f"\nUsing ADAPTIVE thresholds (computed from test data):")
    test_high_thresh = np.percentile(importance_test, 70)
    test_low_thresh = np.percentile(importance_test, 30)
    print(f"  Test high threshold: {test_high_thresh:.4f} (training was {high_thresh:.4f})")
    print(f"  Test low threshold:  {test_low_thresh:.4f} (training was {low_thresh:.4f})")
    
    print(f"\n  Low:    {np.sum(regions_test_adaptive == 0)} ({np.sum(regions_test_adaptive == 0)/len(X_test)*100:.1f}%)")
    print(f"  Medium: {np.sum(regions_test_adaptive == 1)} ({np.sum(regions_test_adaptive == 1)/len(X_test)*100:.1f}%)")
    print(f"  High:   {np.sum(regions_test_adaptive == 2)} ({np.sum(regions_test_adaptive == 2)/len(X_test)*100:.1f}%)")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    # Check importance score distributions
    print(f"\nImportance score comparison:")
    print(f"  Training: min={importance_train.min():.4f}, mean={importance_train.mean():.4f}, max={importance_train.max():.4f}")
    print(f"  Test:     min={importance_test.min():.4f}, mean={importance_test.mean():.4f}, max={importance_test.max():.4f}")
    
    # If test importance is shifted up, everything becomes "high"
    if importance_test.mean() > importance_train.mean() + 0.1:
        print("\n  ⚠️ Test importance scores shifted UP")
        print("     → More test points classified as high importance")
        print("     → This could explain routing to one model")
    
    # Check how many test points exceed training thresholds
    above_high = np.sum(importance_test >= high_thresh)
    below_low = np.sum(importance_test <= low_thresh)
    
    print(f"\nTest points relative to training thresholds:")
    print(f"  Above high threshold: {above_high} ({above_high/len(X_test)*100:.1f}%)")
    print(f"  Below low threshold:  {below_low} ({below_low/len(X_test)*100:.1f}%)")
    print(f"  Between:              {len(X_test) - above_high - below_low} ({(len(X_test)-above_high-below_low)/len(X_test)*100:.1f}%)")
    
    if above_high > 0.5 * len(X_test):
        print("\n  🚨 PROBLEM: >50% of test points exceed high threshold!")
        print("     All being classified as high importance")
        print("     All routed to same model → no benefit from region-aware")


if __name__ == "__main__":
    diagnose_test_regions()