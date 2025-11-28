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
    
    print("\n" + "=" * 95)
    print("EXPERIMENTAL COMPARISON: Region-Aware vs Baselines")
    print("=" * 95)
    
    for dataset in datasets:
        print(f"\n{'─' * 95}")
        print(f"Dataset: {dataset.upper()}")
        print('─' * 95)
        print(f"{'Method':<25} {'RMSE':>10} {'NLL':>10} {'ECE':>10} {'Train(s)':>10} {'Infer(s)':>10} {'Memory':>12}")
        print('─' * 95)
        
        baseline = baseline_results[dataset]
        
        # Exact GP
        if 'exact_gp' in baseline:
            gp = baseline['exact_gp']
            mem = gp.get('peak_memory_mb', 'N/A')
            mem_str = f"{mem:.1f} MB" if isinstance(mem, (int, float)) else mem
            infer = gp.get('inference_time', 'N/A')
            infer_str = f"{infer:.3f}" if isinstance(infer, (int, float)) else infer
            print(f"{'Exact GP':<25} {gp['rmse']:>10.4f} {gp['nll']:>10.4f} {gp['ece']:>10.4f} {gp['training_time']:>10.2f} {infer_str:>10} {mem_str:>12}")
        
        # RFF
        if 'rff' in baseline:
            rff = baseline['rff']
            mem = rff.get('peak_memory_mb', 'N/A')
            mem_str = f"{mem:.1f} MB" if isinstance(mem, (int, float)) else mem
            infer = rff.get('inference_time', 'N/A')
            infer_str = f"{infer:.3f}" if isinstance(infer, (int, float)) else infer
            print(f"{'RFF (uniform)':<25} {rff['rmse']:>10.4f} {rff['nll']:>10.4f} {rff['ece']:>10.4f} {rff['training_time']:>10.2f} {infer_str:>10} {mem_str:>12}")
        
        # Nyström
        if 'nystrom' in baseline:
            nys = baseline['nystrom']
            mem = nys.get('peak_memory_mb', 'N/A')
            mem_str = f"{mem:.1f} MB" if isinstance(mem, (int, float)) else mem
            infer = nys.get('inference_time', 'N/A')
            infer_str = f"{infer:.3f}" if isinstance(infer, (int, float)) else infer
            print(f"{'Nyström (uniform)':<25} {nys['rmse']:>10.4f} {nys['nll']:>10.4f} {nys['ece']:>10.4f} {nys['training_time']:>10.2f} {infer_str:>10} {mem_str:>12}")
        
        # Region-Aware
        if dataset in region_aware_results and 'overall' in region_aware_results[dataset]:
            ra = region_aware_results[dataset]['overall']
            mem = ra.get('peak_memory_mb', 'N/A')
            mem_str = f"{mem:.1f} MB" if isinstance(mem, (int, float)) else mem
            infer = ra.get('inference_time', 'N/A')
            infer_str = f"{infer:.3f}" if isinstance(infer, (int, float)) else infer
            print(f"{'Region-Aware (ours)':<25} {ra['rmse']:>10.4f} {ra['nll']:>10.4f} {ra['ece']:>10.4f} {ra['training_time']:>10.2f} {infer_str:>10} {mem_str:>12}")
            
            # Calculate improvement over uniform Nyström
            if 'nystrom' in baseline:
                nys = baseline['nystrom']
                ece_improvement = (nys['ece'] - ra['ece']) / nys['ece'] * 100
                nll_improvement = (nys['nll'] - ra['nll']) / nys['nll'] * 100
                rmse_improvement = (nys['rmse'] - ra['rmse']) / nys['rmse'] * 100
                train_change = (ra['training_time'] - nys['training_time']) / nys['training_time'] * 100
                
                print(f"\n  Improvement over uniform Nyström:")
                print(f"    RMSE: {rmse_improvement:+.1f}%  |  NLL: {nll_improvement:+.1f}%  |  ECE: {ece_improvement:+.1f}%")
                
                # Inference time comparison
                nys_infer = nys.get('inference_time')
                ra_infer = ra.get('inference_time')
                if nys_infer and ra_infer:
                    infer_change = (ra_infer - nys_infer) / nys_infer * 100
                    print(f"    Training Time: {train_change:+.1f}%  |  Inference Time: {infer_change:+.1f}%")
                else:
                    print(f"    Training Time: {train_change:+.1f}%")
        else:
            print(f"{'Region-Aware (ours)':<25} {'N/A':>10}")
    
    print("\n" + "=" * 95)


def generate_summary_statistics(baseline_results, region_aware_results):
    """Generate summary statistics across all datasets."""
    
    print("\n" + "=" * 95)
    print("SUMMARY: Average Improvement Across Datasets (vs Uniform Nyström)")
    print("=" * 95)
    
    ece_improvements = []
    nll_improvements = []
    rmse_improvements = []
    train_changes = []
    infer_changes = []
    
    for dataset in baseline_results.keys():
        if dataset in region_aware_results and 'overall' in region_aware_results[dataset]:
            ra = region_aware_results[dataset]['overall']
            nys = baseline_results[dataset].get('nystrom', {})
            
            if nys:
                if 'ece' in nys and 'ece' in ra:
                    ece_imp = (nys['ece'] - ra['ece']) / nys['ece'] * 100
                    ece_improvements.append(ece_imp)
                
                if 'nll' in nys and 'nll' in ra:
                    nll_imp = (nys['nll'] - ra['nll']) / nys['nll'] * 100
                    nll_improvements.append(nll_imp)
                
                if 'rmse' in nys and 'rmse' in ra:
                    rmse_imp = (nys['rmse'] - ra['rmse']) / nys['rmse'] * 100
                    rmse_improvements.append(rmse_imp)
                
                if 'training_time' in nys and 'training_time' in ra:
                    train_chg = (ra['training_time'] - nys['training_time']) / nys['training_time'] * 100
                    train_changes.append(train_chg)
                
                if 'inference_time' in nys and 'inference_time' in ra:
                    infer_chg = (ra['inference_time'] - nys['inference_time']) / nys['inference_time'] * 100
                    infer_changes.append(infer_chg)
    
    print("\nAccuracy & Calibration (positive = better):")
    if rmse_improvements:
        print(f"  RMSE: Mean {np.mean(rmse_improvements):+.1f}%, Median {np.median(rmse_improvements):+.1f}%, Range [{min(rmse_improvements):+.1f}% to {max(rmse_improvements):+.1f}%]")
    if nll_improvements:
        print(f"  NLL:  Mean {np.mean(nll_improvements):+.1f}%, Median {np.median(nll_improvements):+.1f}%, Range [{min(nll_improvements):+.1f}% to {max(nll_improvements):+.1f}%]")
    if ece_improvements:
        print(f"  ECE:  Mean {np.mean(ece_improvements):+.1f}%, Median {np.median(ece_improvements):+.1f}%, Range [{min(ece_improvements):+.1f}% to {max(ece_improvements):+.1f}%]")
    
    print("\nEfficiency (negative = faster/less overhead):")
    if train_changes:
        print(f"  Training Time:  Mean {np.mean(train_changes):+.1f}%, Median {np.median(train_changes):+.1f}%, Range [{min(train_changes):+.1f}% to {max(train_changes):+.1f}%]")
    if infer_changes:
        print(f"  Inference Time: Mean {np.mean(infer_changes):+.1f}%, Median {np.median(infer_changes):+.1f}%, Range [{min(infer_changes):+.1f}% to {max(infer_changes):+.1f}%]")
    
    print("\n" + "=" * 95)


def save_comparison_csv(baseline_results, region_aware_results):
    """Save comparison results to CSV for easy import to reports."""
    
    output_path = PROJECT_ROOT / 'results' / 'comparison.csv'
    
    with open(output_path, 'w') as f:
        # Header
        f.write("Dataset,Method,RMSE,NLL,ECE,Training_Time,Inference_Time,Peak_Memory_MB\n")
        
        for dataset in baseline_results.keys():
            baseline = baseline_results[dataset]
            
            # Exact GP
            if 'exact_gp' in baseline:
                gp = baseline['exact_gp']
                mem = gp.get('peak_memory_mb', '')
                infer = gp.get('inference_time', '')
                f.write(f"{dataset},Exact GP,{gp['rmse']:.4f},{gp['nll']:.4f},{gp['ece']:.4f},{gp['training_time']:.2f},{infer},{mem}\n")
            
            # RFF
            if 'rff' in baseline:
                rff = baseline['rff']
                mem = rff.get('peak_memory_mb', '')
                infer = rff.get('inference_time', '')
                f.write(f"{dataset},RFF,{rff['rmse']:.4f},{rff['nll']:.4f},{rff['ece']:.4f},{rff['training_time']:.2f},{infer},{mem}\n")
            
            # Nyström
            if 'nystrom' in baseline:
                nys = baseline['nystrom']
                mem = nys.get('peak_memory_mb', '')
                infer = nys.get('inference_time', '')
                f.write(f"{dataset},Nystrom,{nys['rmse']:.4f},{nys['nll']:.4f},{nys['ece']:.4f},{nys['training_time']:.2f},{infer},{mem}\n")
            
            # Region-Aware
            if dataset in region_aware_results and 'overall' in region_aware_results[dataset]:
                ra = region_aware_results[dataset]['overall']
                mem = ra.get('peak_memory_mb', '')
                infer = ra.get('inference_time', '')
                f.write(f"{dataset},Region-Aware,{ra['rmse']:.4f},{ra['nll']:.4f},{ra['ece']:.4f},{ra['training_time']:.2f},{infer},{mem}\n")
    
    print(f"\nComparison CSV saved to: {output_path}")


if __name__ == "__main__":
    baseline_results, region_aware_results = load_results()
    print_comparison_table(baseline_results, region_aware_results)
    generate_summary_statistics(baseline_results, region_aware_results)
    save_comparison_csv(baseline_results, region_aware_results)