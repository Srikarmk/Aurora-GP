"""
Diagnostic script: Understand why test points cluster into one region
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from region_identification import RegionIdentifier

PROJECT_ROOT = Path(__file__).parent.parent


def diagnose_dataset(dataset_name):
    """Analyze region distribution for a dataset."""
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSING: {dataset_name}")
    print('='*60)
    
    # Load data
    data_path = PROJECT_ROOT / 'data' / f'{dataset_name}.npz'
    data = np.load(data_path, allow_pickle=True)
    X = data['X']
    y = data['y']
    
    # Limit size
    if len(X) > 10000:
        idx = np.random.RandomState(42).choice(len(X), 10000, replace=False)
        X = X[idx]
        y = y[idx]
    
    # Train/test split (same as benchmark)
    n_train = int(0.8 * len(X))
    indices = np.random.RandomState(42).permutation(len(X))
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Fit region identifier
    identifier = RegionIdentifier(random_state=42)
    identifier.fit(X_train, y_train, n_gp_iter=30)
    
    # Get training importance scores
    train_scores = identifier.importance_scores
    
    # Get test importance scores
    test_regions, test_scores = identifier.predict_regions(X_test)
    
    # Print distributions
    print(f"\nTRAINING DATA importance scores:")
    print(f"  Min:    {train_scores.min():.4f}")
    print(f"  Max:    {train_scores.max():.4f}")
    print(f"  Mean:   {train_scores.mean():.4f}")
    print(f"  Median: {np.median(train_scores):.4f}")
    print(f"  Std:    {train_scores.std():.4f}")
    
    print(f"\nTEST DATA importance scores:")
    print(f"  Min:    {test_scores.min():.4f}")
    print(f"  Max:    {test_scores.max():.4f}")
    print(f"  Mean:   {test_scores.mean():.4f}")
    print(f"  Median: {np.median(test_scores):.4f}")
    print(f"  Std:    {test_scores.std():.4f}")
    
    print(f"\nTHRESHOLDS (from training):")
    print(f"  Low threshold:  {identifier.low_threshold_value:.4f}")
    print(f"  High threshold: {identifier.high_threshold_value:.4f}")
    
    # Where do test scores fall?
    pct_below_low = (test_scores <= identifier.low_threshold_value).mean() * 100
    pct_above_high = (test_scores >= identifier.high_threshold_value).mean() * 100
    pct_medium = 100 - pct_below_low - pct_above_high
    
    print(f"\nTEST POINTS distribution:")
    print(f"  Below low threshold (→ LOW):    {pct_below_low:.1f}%")
    print(f"  Between thresholds (→ MEDIUM):  {pct_medium:.1f}%")
    print(f"  Above high threshold (→ HIGH):  {pct_above_high:.1f}%")
    
    # The key question: are test scores shifted?
    train_median = np.median(train_scores)
    test_median = np.median(test_scores)
    shift = test_median - train_median
    
    print(f"\nSCORE SHIFT:")
    print(f"  Test median - Train median = {shift:+.4f}")
    if shift > 0.1:
        print(f"  ⚠️  Test scores are HIGHER than training (explains HIGH clustering)")
    elif shift < -0.1:
        print(f"  ⚠️  Test scores are LOWER than training")
    else:
        print(f"  ✓  Distributions are similar")


if __name__ == "__main__":
    datasets = ['concrete', 'protein', 'robot_arm', 'sarcos', 'synthetic_heteroscedastic']
    
    for dataset in datasets:
        try:
            diagnose_dataset(dataset)
        except Exception as e:
            print(f"Error with {dataset}: {e}")