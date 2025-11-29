"""
Compare Exact GP, Uniform Approximations, and AURORA
Generate comparison tables and validate improvements
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def load_all_results():
    """Load results from all three approaches"""
    
    results = {}
    
    # Load Exact GP baseline
    gp_file = Path('results/baseline_gp/summary.json')
    if gp_file.exists():
        with open(gp_file, 'r') as f:
            results['exact_gp'] = json.load(f)
    else:
        print(f"⚠ GP baseline not found at {gp_file}")
    
    # Load Approximations
    approx_file = Path('results/approximations/summary.json')
    if approx_file.exists():
        with open(approx_file, 'r') as f:
            results['approximations'] = json.load(f)
    else:
        print(f"⚠ Approximations not found at {approx_file}")
    
    # Load AURORA
    aurora_file = Path('results/region_aware/summary.json')
    if aurora_file.exists():
        with open(aurora_file, 'r') as f:
            results['aurora'] = json.load(f)
    else:
        print(f"⚠ AURORA results not found at {aurora_file}")
    
    return results


def create_comparison_table(results):
    """Create comprehensive comparison table"""
    
    data = []
    
    for dataset in results['exact_gp'].keys():
        # Exact GP
        if dataset in results['exact_gp'] and 'error' not in results['exact_gp'][dataset]:
            gp = results['exact_gp'][dataset]
            data.append({
                'Dataset': dataset,
                'Method': 'Exact GP',
                'RMSE': gp['rmse'],
                'NLL': gp['nll'],
                'ECE': gp['ece'],
                'Time': gp.get('training_time', 0)
            })
        
        # Uniform RFF
        if dataset in results['approximations'] and 'rff' in results['approximations'][dataset]:
            rff = results['approximations'][dataset]['rff']
            if 'error' not in rff:
                data.append({
                    'Dataset': dataset,
                    'Method': 'RFF (uniform)',
                    'RMSE': rff['rmse'],
                    'NLL': rff['nll'],
                    'ECE': rff['ece'],
                    'Time': rff['training_time']
                })
        
        # Uniform Nyström
        if dataset in results['approximations'] and 'nystrom' in results['approximations'][dataset]:
            nys = results['approximations'][dataset]['nystrom']
            if 'error' not in nys:
                data.append({
                    'Dataset': dataset,
                    'Method': 'Nyström (uniform)',
                    'RMSE': nys['rmse'],
                    'NLL': nys['nll'],
                    'ECE': nys['ece'],
                    'Time': nys['training_time']
                })
        
        # AURORA
        if dataset in results['aurora'] and 'overall' in results['aurora'][dataset]:
            aurora = results['aurora'][dataset]['overall']
            data.append({
                'Dataset': dataset,
                'Method': 'AURORA',
                'RMSE': aurora['rmse'],
                'NLL': aurora['nll'],
                'ECE': aurora['ece'],
                'Time': aurora['training_time']
            })
    
    df = pd.DataFrame(data)
    return df


def validate_aurora_improvement(df):
    """Validate that AURORA shows proper improvements"""
    
    print("\n" + "="*70)
    print("AURORA IMPROVEMENT VALIDATION")
    print("="*70)
    
    datasets = df['Dataset'].unique()
    
    validation_results = []
    
    for dataset in datasets:
        dataset_df = df[df['Dataset'] == dataset]
        
        # Get ECE values
        gp_ece = dataset_df[dataset_df['Method'] == 'Exact GP']['ECE'].values
        aurora_ece = dataset_df[dataset_df['Method'] == 'AURORA']['ECE'].values
        
        # Get uniform baseline (use best of RFF/Nyström)
        uniform_eces = dataset_df[dataset_df['Method'].str.contains('uniform')]['ECE'].values
        
        if len(gp_ece) == 0 or len(aurora_ece) == 0 or len(uniform_eces) == 0:
            continue
        
        gp_ece = gp_ece[0]
        aurora_ece = aurora_ece[0]
        uniform_ece = uniform_eces.min()  # Best uniform method
        
        print(f"\n{dataset}:")
        print(f"  GP ECE:      {gp_ece:.4f}")
        print(f"  Uniform ECE: {uniform_ece:.4f}")
        print(f"  AURORA ECE:  {aurora_ece:.4f}")
        
        # Validation checks
        checks = []
        
        # Check 1: Not too good
        if aurora_ece < 0.85 * gp_ece:
            checks.append(f"  ❌ AURORA beats GP ({aurora_ece:.4f} < 0.85*{gp_ece:.4f}={0.85*gp_ece:.4f})")
            checks.append(f"     → Likely data leakage or bug!")
            status = "FAIL"
        else:
            checks.append(f"  ✓ AURORA doesn't beat GP ({aurora_ece:.4f} >= {0.85*gp_ece:.4f})")
            status = "PASS"
        
        # Check 2: Better than uniform
        if aurora_ece < uniform_ece:
            improvement = (uniform_ece - aurora_ece) / uniform_ece * 100
            checks.append(f"  ✓ AURORA better than uniform ({improvement:.1f}% improvement)")
        else:
            checks.append(f"  ⚠ AURORA not better than uniform")
            status = "WARN"
        
        # Check 3: Realistic range
        if aurora_ece < 0.08:
            checks.append(f"  ❌ ECE unrealistically low ({aurora_ece:.4f} < 0.08)")
            status = "FAIL"
        elif aurora_ece > 0.50:
            checks.append(f"  ⚠ ECE quite high ({aurora_ece:.4f} > 0.50)")
        else:
            checks.append(f"  ✓ ECE in realistic range ({aurora_ece:.4f})")
        
        for check in checks:
            print(check)
        
        validation_results.append({
            'dataset': dataset,
            'status': status,
            'gp_ece': gp_ece,
            'uniform_ece': uniform_ece,
            'aurora_ece': aurora_ece,
            'improvement_pct': (uniform_ece - aurora_ece) / uniform_ece * 100 if aurora_ece < uniform_ece else 0
        })
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    n_pass = sum(1 for r in validation_results if r['status'] == 'PASS')
    n_warn = sum(1 for r in validation_results if r['status'] == 'WARN')
    n_fail = sum(1 for r in validation_results if r['status'] == 'FAIL')
    
    print(f"✓ Passed: {n_pass}/{len(validation_results)}")
    print(f"⚠ Warnings: {n_warn}/{len(validation_results)}")
    print(f"❌ Failed: {n_fail}/{len(validation_results)}")
    
    if n_fail > 0:
        print("\n⚠️  VALIDATION FAILED")
        print("Likely issues:")
        print("  - Still using adaptive_thresholds=True")
        print("  - Evaluating on training data instead of test")
        print("  - Not training separate models per region")
    elif n_pass >= 3:
        print("\n✅ VALIDATION PASSED")
        print("AURORA is working correctly!")
    else:
        print("\n⚠️  MARGINAL VALIDATION")
        print("Some datasets working, others need investigation")
    
    return validation_results


def plot_comparison(df, save_path='results/aurora_comparison.png'):
    """Create visual comparison of all methods"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # ECE comparison
    ax = axes[0]
    datasets = df['Dataset'].unique()
    methods = ['Exact GP', 'RFF (uniform)', 'Nyström (uniform)', 'AURORA']
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, method in enumerate(methods):
        method_data = []
        for dataset in datasets:
            ece = df[(df['Dataset'] == dataset) & (df['Method'] == method)]['ECE'].values
            method_data.append(ece[0] if len(ece) > 0 else np.nan)
        
        offset = (i - 1.5) * width
        ax.bar(x + offset, method_data, width, label=method, alpha=0.8)
    
    ax.set_ylabel('ECE (lower is better)')
    ax.set_title('Calibration Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Training time
    ax = axes[1]
    for i, method in enumerate(methods):
        method_data = []
        for dataset in datasets:
            time_val = df[(df['Dataset'] == dataset) & (df['Method'] == method)]['Time'].values
            method_data.append(time_val[0] if len(time_val) > 0 else np.nan)
        
        offset = (i - 1.5) * width
        ax.bar(x + offset, method_data, width, label=method, alpha=0.8)
    
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Speed Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    # ECE improvement over uniform
    ax = axes[2]
    improvements = []
    for dataset in datasets:
        uniform_ece = df[(df['Dataset'] == dataset) & (df['Method'].str.contains('uniform'))]['ECE'].min()
        aurora_ece = df[(df['Dataset'] == dataset) & (df['Method'] == 'AURORA')]['ECE'].values
        
        if len(aurora_ece) > 0:
            improvement = (uniform_ece - aurora_ece[0]) / uniform_ece * 100
            improvements.append(improvement)
        else:
            improvements.append(0)
    
    colors = ['green' if i > 0 else 'red' for i in improvements]
    ax.bar(x, improvements, width=0.6, color=colors, alpha=0.7)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('ECE Improvement (%)')
    ax.set_title('AURORA Improvement over Uniform')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    print("\n" + "="*70)
    print("AURORA RESULTS COMPARISON & VALIDATION")
    print("="*70)
    
    # Load all results
    print("\nLoading results...")
    results = load_all_results()
    
    # Create comparison table
    print("\nCreating comparison table...")
    df = create_comparison_table(results)
    
    print("\n" + "="*70)
    print("COMPLETE COMPARISON TABLE")
    print("="*70)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('results/complete_comparison.csv', index=False)
    print(f"\n✓ Saved to results/complete_comparison.csv")
    
    # Validate AURORA improvements
    validation_results = validate_aurora_improvement(df)
    
    # Create plots
    print("\nGenerating comparison plots...")
    plot_comparison(df)
    
    print("\n" + "="*70)
    print("✓ Comparison complete!")
    print("="*70)