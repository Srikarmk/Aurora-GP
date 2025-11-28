"""
AURORA - Experimental Evaluation
Compare Region-Aware Approximation against baselines
"""

import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent


def load_results():
    """Load all results from baseline and region-aware experiments."""
    
    # Load baseline approximation results
    baseline_path = PROJECT_ROOT / 'results' / 'approximations' / 'summary.json'
    with open(baseline_path, 'r') as f:
        baseline_results = json.load(f)
    
    # Load region-aware results
    region_aware_path = PROJECT_ROOT / 'results' / 'region_aware' / 'summary.json'
    with open(region_aware_path, 'r') as f:
        region_aware_results = json.load(f)
    
    return baseline_results, region_aware_results


def print_comparison_table(baseline_results, region_aware_results):
    """Print a formatted comparison table."""
    
    datasets = list(baseline_results.keys())
    
    print("\n" + "=" * 90)
    print("EXPERIMENTAL COMPARISON: Region-Aware vs Baselines")
    print("=" * 90)
    
    for dataset in datasets:
        print(f"\n{'─' * 90}")
        print(f"Dataset: {dataset.upper()}")
        print('─' * 90)
        print(f"{'Method':<25} {'RMSE':>12} {'NLL':>12} {'ECE':>12} {'Train Time':>12} {'Memory':>12}")
        print('─' * 90)
        
        baseline = baseline_results[dataset]
        
        # Exact GP
        if 'exact_gp' in baseline:
            gp = baseline['exact_gp']
            mem = gp.get('peak_memory_mb', 'N/A')
            mem_str = f"{mem:.1f} MB" if isinstance(mem, (int, float)) else mem
            print(f"{'Exact GP':<25} {gp['rmse']:>12.4f} {gp['nll']:>12.4f} {gp['ece']:>12.4f} {gp['training_time']:>11.2f}s {mem_str:>12}")
        
        # RFF
        if 'rff' in baseline:
            rff = baseline['rff']
            mem = rff.get('peak_memory_mb', 'N/A')
            mem_str = f"{mem:.1f} MB" if isinstance(mem, (int, float)) else mem
            print(f"{'RFF (uniform)':<25} {rff['rmse']:>12.4f} {rff['nll']:>12.4f} {rff['ece']:>12.4f} {rff['training_time']:>11.2f}s {mem_str:>12}")
        
        # Nyström
        if 'nystrom' in baseline:
            nys = baseline['nystrom']
            mem = nys.get('peak_memory_mb', 'N/A')
            mem_str = f"{mem:.1f} MB" if isinstance(mem, (int, float)) else mem
            print(f"{'Nyström (uniform)':<25} {nys['rmse']:>12.4f} {nys['nll']:>12.4f} {nys['ece']:>12.4f} {nys['training_time']:>11.2f}s {mem_str:>12}")
        
        # Region-Aware
        if dataset in region_aware_results and 'overall' in region_aware_results[dataset]:
            ra = region_aware_results[dataset]['overall']
            mem = ra.get('peak_memory_mb', 'N/A')
            mem_str = f"{mem:.1f} MB" if isinstance(mem, (int, float)) else mem
            print(f"{'Region-Aware (ours)':<25} {ra['rmse']:>12.4f} {ra['nll']:>12.4f} {ra['ece']:>12.4f} {ra['training_time']:>11.2f}s {mem_str:>12}")
            
            # Calculate improvement over uniform Nyström
            if 'nystrom' in baseline:
                nys = baseline['nystrom']
                ece_improvement = (nys['ece'] - ra['ece']) / nys['ece'] * 100
                nll_improvement = (nys['nll'] - ra['nll']) / nys['nll'] * 100
                print(f"\n  → ECE improvement over uniform Nyström: {ece_improvement:+.1f}%")
                print(f"  → NLL improvement over uniform Nyström: {nll_improvement:+.1f}%")
        else:
            print(f"{'Region-Aware (ours)':<25} {'N/A':>12}")
    
    print("\n" + "=" * 90)


def generate_summary_statistics(baseline_results, region_aware_results):
    """Generate summary statistics across all datasets."""
    
    print("\n" + "=" * 90)
    print("SUMMARY: Average Improvement Across Datasets")
    print("=" * 90)
    
    ece_improvements = []
    nll_improvements = []
    
    for dataset in baseline_results.keys():
        if dataset in region_aware_results and 'overall' in region_aware_results[dataset]:
            ra = region_aware_results[dataset]['overall']
            nys = baseline_results[dataset].get('nystrom', {})
            
            if nys and 'ece' in nys and 'ece' in ra:
                ece_imp = (nys['ece'] - ra['ece']) / nys['ece'] * 100
                ece_improvements.append(ece_imp)
            
            if nys and 'nll' in nys and 'nll' in ra:
                nll_imp = (nys['nll'] - ra['nll']) / nys['nll'] * 100
                nll_improvements.append(nll_imp)
    
    if ece_improvements:
        print(f"\nECE Improvement over Uniform Nyström:")
        print(f"  Mean:   {np.mean(ece_improvements):+.1f}%")
        print(f"  Median: {np.median(ece_improvements):+.1f}%")
        print(f"  Range:  {min(ece_improvements):+.1f}% to {max(ece_improvements):+.1f}%")
    
    if nll_improvements:
        print(f"\nNLL Improvement over Uniform Nyström:")
        print(f"  Mean:   {np.mean(nll_improvements):+.1f}%")
        print(f"  Median: {np.median(nll_improvements):+.1f}%")
        print(f"  Range:  {min(nll_improvements):+.1f}% to {max(nll_improvements):+.1f}%")
    
    print("\n" + "=" * 90)


def save_comparison_csv(baseline_results, region_aware_results):
    """Save comparison results to CSV for easy import to reports."""
    
    output_path = PROJECT_ROOT / 'results' / 'comparison.csv'
    
    with open(output_path, 'w') as f:
        # Header
        f.write("Dataset,Method,RMSE,NLL,ECE,Training_Time,Peak_Memory_MB\n")
        
        for dataset in baseline_results.keys():
            baseline = baseline_results[dataset]
            
            # Exact GP
            if 'exact_gp' in baseline:
                gp = baseline['exact_gp']
                mem = gp.get('peak_memory_mb', '')
                f.write(f"{dataset},Exact GP,{gp['rmse']:.4f},{gp['nll']:.4f},{gp['ece']:.4f},{gp['training_time']:.2f},{mem}\n")
            
            # RFF
            if 'rff' in baseline:
                rff = baseline['rff']
                mem = rff.get('peak_memory_mb', '')
                f.write(f"{dataset},RFF,{rff['rmse']:.4f},{rff['nll']:.4f},{rff['ece']:.4f},{rff['training_time']:.2f},{mem}\n")
            
            # Nyström
            if 'nystrom' in baseline:
                nys = baseline['nystrom']
                mem = nys.get('peak_memory_mb', '')
                f.write(f"{dataset},Nystrom,{nys['rmse']:.4f},{nys['nll']:.4f},{nys['ece']:.4f},{nys['training_time']:.2f},{mem}\n")
            
            # Region-Aware
            if dataset in region_aware_results and 'overall' in region_aware_results[dataset]:
                ra = region_aware_results[dataset]['overall']
                mem = ra.get('peak_memory_mb', '')
                f.write(f"{dataset},Region-Aware,{ra['rmse']:.4f},{ra['nll']:.4f},{ra['ece']:.4f},{ra['training_time']:.2f},{mem}\n")
    
    print(f"\nComparison CSV saved to: {output_path}")


if __name__ == "__main__":
    baseline_results, region_aware_results = load_results()
    print_comparison_table(baseline_results, region_aware_results)
    generate_summary_statistics(baseline_results, region_aware_results)
    save_comparison_csv(baseline_results, region_aware_results)