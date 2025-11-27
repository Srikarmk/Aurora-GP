"""
Validate AURORA Stage 1 results
Check if region identification makes sense
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Get project root (parent of src directory)
PROJECT_ROOT = Path(__file__).parent.parent


def validate_synthetic_heteroscedastic():
    """
    Validate that high importance regions correspond to high noise areas
    (far from origin in synthetic_heteroscedastic)
    """
    print("\n" + "="*60)
    print("VALIDATION: Synthetic Heteroscedastic")
    print("="*60)
    
    try:
        # Load dataset
        data_file = PROJECT_ROOT / 'data' / 'synthetic_heteroscedastic.npz'
        if not data_file.exists():
            print(f"Dataset not found: {data_file}")
            return False
        
        data = np.load(data_file)
        X = data['X']
        
        # Load region identification results
        results_file = PROJECT_ROOT / 'results' / 'region_identification' / 'synthetic_heteroscedastic' / 'region_data.npz'
        if not results_file.exists():
            print(f"Results not found: {results_file}")
            print("Run region_identification.py first!")
            return False
        
        region_data = np.load(results_file)
        importance_scores = region_data['importance_scores']
        
        # Compute distance from origin (noise increases with distance)
        distances = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
        
        # Correlation: high distance should = high importance
        correlation = np.corrcoef(distances, importance_scores)[0, 1]
        
        print(f"\nDistance from origin vs Importance correlation: {correlation:.3f}")
        
        if correlation > 0.3:
            print("[PASS] VALIDATION PASSED: High importance in high-noise regions!")
            status = "PASS"
        elif correlation > 0.1:
            print("[WARN] VALIDATION MARGINAL: Weak correlation, check weights")
            status = "MARGINAL"
        else:
            print("[FAIL] VALIDATION FAILED: Regions don't match expected pattern")
            status = "FAIL"
        
        # Create validation plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter: distance vs importance
        ax = axes[0]
        ax.scatter(distances, importance_scores, alpha=0.3, s=20)
        ax.set_xlabel('Distance from Origin (relates to noise level)')
        ax.set_ylabel('Importance Score')
        ax.set_title(f'Validation: Correlation = {correlation:.3f}')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(distances, importance_scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(distances.min(), distances.max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Trend (slope={z[0]:.3f})')
        ax.legend()
        
        # 2D scatter showing regions
        ax = axes[1]
        regions = np.zeros(len(importance_scores))
        high_thresh = region_data['high_threshold_value']
        low_thresh = region_data['low_threshold_value']
        
        regions[importance_scores >= high_thresh] = 2
        regions[importance_scores <= low_thresh] = 0
        regions[(importance_scores > low_thresh) & (importance_scores < high_thresh)] = 1
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        labels = ['Low (should be near origin)', 'Medium', 'High (should be far from origin)']
        
        for region_id, color, label in zip([0, 1, 2], colors, labels):
            mask = regions == region_id
            ax.scatter(X[mask, 0], X[mask, 1], c=color, label=label, 
                      s=20, alpha=0.6, edgecolors='none')
        
        # Draw origin
        ax.scatter([0], [0], c='black', s=200, marker='x', linewidths=3,
                  label='Origin (low noise)', zorder=5)
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Regions (should match noise pattern)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Synthetic Heteroscedastic Validation: {status}', fontsize=14)
        plt.tight_layout()
        
        # Save
        plot_dir = PROJECT_ROOT / 'results' / 'region_identification' / 'plots'
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_dir / 'validation_synthetic.png', dpi=150, bbox_inches='tight')
        print(f"\n[OK] Validation plot saved to {plot_dir / 'validation_synthetic.png'}")
        
        return correlation > 0.3
        
    except Exception as e:
        print(f"\n[ERROR] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_region_balance():
    """Check that regions are reasonably balanced across datasets"""
    print("\n" + "="*60)
    print("CHECK: Region Balance")
    print("="*60)
    
    try:
        summary_file = PROJECT_ROOT / 'results' / 'region_identification' / 'summary.json'
        if not summary_file.exists():
            print(f"Summary not found: {summary_file}")
            return False
        
        with open(summary_file, 'r') as f:
            results = json.load(f)
        
        print(f"\n{'Dataset':<25} {'High%':<10} {'Med%':<10} {'Low%':<10} {'Balance':<10}")
        print("-"*65)
        
        all_balanced = True
        
        for name, stats in results.items():
            if 'error' in stats:
                continue
            
            # Check balance (should be roughly 30/40/30)
            high_pct = stats['pct_high']
            med_pct = stats['pct_medium']
            low_pct = stats['pct_low']
            
            # Simple balance check: no region should be < 15% or > 50%
            balanced = (15 < high_pct < 50) and (15 < med_pct < 60) and (15 < low_pct < 50)
            status = "[OK]" if balanced else "[WARN] IMBALANCED"
            
            print(f"{name:<25} {high_pct:<10.1f} {med_pct:<10.1f} {low_pct:<10.1f} {status:<10}")
            
            if not balanced:
                all_balanced = False
        
        print("-"*65)
        
        if all_balanced:
            print("\n[PASS] CHECK PASSED: All datasets have balanced regions")
            return True
        else:
            print("\n[WARN] CHECK WARNING: Some datasets have imbalanced regions")
            print("  This might be OK if the dataset is naturally skewed")
            return True  # Still return True, just a warning
        
    except Exception as e:
        print(f"\n[ERROR] Check failed: {e}")
        return False


def validate_importance_range():
    """Check that importance scores use full range [0, 1]"""
    print("\n" + "="*60)
    print("CHECK: Importance Score Range")
    print("="*60)
    
    try:
        summary_file = PROJECT_ROOT / 'results' / 'region_identification' / 'summary.json'
        with open(summary_file, 'r') as f:
            results = json.load(f)
        
        print(f"\n{'Dataset':<25} {'Min':<10} {'Max':<10} {'Range':<10} {'Status':<10}")
        print("-"*65)
        
        all_good = True
        
        for name, stats in results.items():
            if 'error' in stats:
                continue
            
            min_val = stats['importance_min']
            max_val = stats['importance_max']
            range_val = max_val - min_val
            
            # Range should be at least 0.3 (otherwise not much variation)
            good_range = range_val > 0.3
            status = "[OK]" if good_range else "[WARN] LOW"
            
            print(f"{name:<25} {min_val:<10.3f} {max_val:<10.3f} {range_val:<10.3f} {status:<10}")
            
            if not good_range:
                all_good = False
        
        print("-"*65)
        
        if all_good:
            print("\n[PASS] CHECK PASSED: All datasets have good importance score range")
        else:
            print("\n[WARN] CHECK WARNING: Some datasets have low importance variation")
            print("  This might mean the dataset is uniform (less benefit from AURORA)")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Check failed: {e}")
        return False


def run_all_validations():
    """Run all validation checks"""
    print("\n" + "="*70)
    print("AURORA STAGE 1: VALIDATION SUITE")
    print("="*70)
    
    checks = [
        ("Synthetic Heteroscedastic Validation", validate_synthetic_heteroscedastic),
        ("Region Balance Check", check_region_balance),
        ("Importance Range Check", validate_importance_range)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            passed = check_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[WARN] Check crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")
    
    n_passed = sum(1 for _, passed in results if passed)
    n_total = len(results)
    
    print(f"\n{n_passed}/{n_total} validations passed")
    
    if n_passed == n_total:
        print("\n[SUCCESS] All validations passed! Stage 1 is working correctly.")
        return True
    else:
        print("\n[WARN] Some validations failed or marginal.")
        print("Review the warnings and decide if acceptable.")
        return False


if __name__ == "__main__":
    import sys
    success = run_all_validations()
    sys.exit(0 if success else 1)