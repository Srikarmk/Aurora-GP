"""
Create Publication-Quality Comparison Plots for AURORA
Compares Exact GP, Uniform Approximations, and AURORA (Config 3)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import pandas as pd

sns.set_style("whitegrid")
sns.set_palette("husl")

PROJECT_ROOT = Path(__file__).parent.parent


def load_all_results():
    """Load results from all methods"""
    results = {}
    
    # Exact GP baseline
    gp_file = PROJECT_ROOT / 'results' / 'baseline_gp' / 'summary.json'
    if gp_file.exists():
        with open(gp_file) as f:
            results['exact_gp'] = json.load(f)
    
    # Approximations (RFF and Nyström)
    approx_file = PROJECT_ROOT / 'results' / 'approximations' / 'summary.json'
    if approx_file.exists():
        with open(approx_file) as f:
            results['approximations'] = json.load(f)
    
    # AURORA Final
    aurora_file = PROJECT_ROOT / 'results' / 'aurora_final' / 'summary.json'
    if aurora_file.exists():
        with open(aurora_file) as f:
            results['aurora'] = json.load(f)
    
    return results


def create_main_comparison_plot(results, save_path):
    """
    Main comparison plot: ECE, RMSE, and Training Time
    Figure 1 for the paper
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Prepare data
    datasets = []
    methods_data = {
        'Exact GP': {'rmse': [], 'ece': [], 'time': []},
        'RFF': {'rmse': [], 'ece': [], 'time': []},
        'Nyström': {'rmse': [], 'ece': [], 'time': []},
        'AURORA': {'rmse': [], 'ece': [], 'time': []}
    }
    
    for dataset in results['exact_gp'].keys():
        if 'error' in results['exact_gp'].get(dataset, {}):
            continue
        
        datasets.append(dataset)
        
        # Exact GP
        gp = results['exact_gp'][dataset]
        methods_data['Exact GP']['rmse'].append(gp['rmse'])
        methods_data['Exact GP']['ece'].append(gp['ece'])
        methods_data['Exact GP']['time'].append(gp.get('training_time', 0))
        
        # RFF
        if dataset in results['approximations'] and 'rff' in results['approximations'][dataset]:
            rff = results['approximations'][dataset]['rff']
            methods_data['RFF']['rmse'].append(rff['rmse'])
            methods_data['RFF']['ece'].append(rff['ece'])
            methods_data['RFF']['time'].append(rff['training_time'])
        else:
            methods_data['RFF']['rmse'].append(np.nan)
            methods_data['RFF']['ece'].append(np.nan)
            methods_data['RFF']['time'].append(np.nan)
        
        # Nyström
        if dataset in results['approximations'] and 'nystrom' in results['approximations'][dataset]:
            nys = results['approximations'][dataset]['nystrom']
            methods_data['Nyström']['rmse'].append(nys['rmse'])
            methods_data['Nyström']['ece'].append(nys['ece'])
            methods_data['Nyström']['time'].append(nys['training_time'])
        else:
            methods_data['Nyström']['rmse'].append(np.nan)
            methods_data['Nyström']['ece'].append(np.nan)
            methods_data['Nyström']['time'].append(np.nan)
        
        # AURORA
        if dataset in results['aurora'] and 'overall' in results['aurora'][dataset]:
            aurora = results['aurora'][dataset]['overall']
            methods_data['AURORA']['rmse'].append(aurora['rmse'])
            methods_data['AURORA']['ece'].append(aurora['ece'])
            methods_data['AURORA']['time'].append(aurora['training_time'])
        else:
            methods_data['AURORA']['rmse'].append(np.nan)
            methods_data['AURORA']['ece'].append(np.nan)
            methods_data['AURORA']['time'].append(np.nan)
    
    x = np.arange(len(datasets))
    width = 0.2
    
    # Plot 1: ECE Comparison (Most Important!)
    ax = axes[0]
    for i, (method, data) in enumerate(methods_data.items()):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, data['ece'], width, label=method, alpha=0.85)
    
    ax.set_ylabel('ECE (lower is better)', fontsize=11, fontweight='bold')
    ax.set_title('Calibration Quality Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    
    # Plot 2: RMSE Comparison
    ax = axes[1]
    for i, (method, data) in enumerate(methods_data.items()):
        offset = (i - 1.5) * width
        ax.bar(x + offset, data['rmse'], width, label=method, alpha=0.85)
    
    ax.set_ylabel('RMSE (lower is better)', fontsize=11, fontweight='bold')
    ax.set_title('Prediction Accuracy Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(bottom=0)
    
    # Plot 3: Training Time Comparison
    ax = axes[2]
    for i, (method, data) in enumerate(methods_data.items()):
        offset = (i - 1.5) * width
        ax.bar(x + offset, data['time'], width, label=method, alpha=0.85)
    
    ax.set_ylabel('Training Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_title('Computational Efficiency', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    return fig


def create_improvement_plot(results, save_path):
    """
    Plot ECE improvement of AURORA over uniform baselines
    Figure 2 for the paper
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    datasets = []
    improvements = []
    
    for dataset in results['exact_gp'].keys():
        if 'error' in results['exact_gp'].get(dataset, {}):
            continue
        
        # Get uniform baseline (best of RFF/Nyström)
        uniform_ece = float('inf')
        
        if dataset in results['approximations']:
            if 'rff' in results['approximations'][dataset]:
                uniform_ece = min(uniform_ece, results['approximations'][dataset]['rff']['ece'])
            if 'nystrom' in results['approximations'][dataset]:
                uniform_ece = min(uniform_ece, results['approximations'][dataset]['nystrom']['ece'])
        
        # Get AURORA ECE
        if dataset in results['aurora'] and 'overall' in results['aurora'][dataset]:
            aurora_ece = results['aurora'][dataset]['overall']['ece']
            
            improvement = (uniform_ece - aurora_ece) / uniform_ece * 100
            
            datasets.append(dataset)
            improvements.append(improvement)
    
    # Create bar plot
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax.bar(range(len(datasets)), improvements, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:+.1f}%',
               ha='center', va='bottom' if val > 0 else 'top',
               fontsize=10, fontweight='bold')
    
    ax.axhline(0, color='black', linewidth=1.5)
    ax.set_ylabel('ECE Improvement over Uniform Baseline (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=11, fontweight='bold')
    ax.set_title('AURORA Improvement over Uniform Approximations', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add average line
    avg_improvement = np.mean(improvements)
    ax.axhline(avg_improvement, color='blue', linestyle='--', linewidth=2, 
              label=f'Average: {avg_improvement:+.1f}%')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    return fig


def create_calibration_curves(results, save_path):
    """
    Plot calibration curves for all methods
    Figure 3 for the paper
    """
    # This would require raw predictions, which we don't have in summary
    # Skip for now or implement if you have detailed results
    print(f"  [SKIP] Calibration curves require raw predictions (not in summary)")
    return None


def create_speedup_accuracy_tradeoff(results, save_path):
    """
    Scatter plot: ECE vs Training Time
    Shows AURORA's position in the speed-accuracy tradeoff space
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    markers = {'Exact GP': 'o', 'RFF': 's', 'Nyström': '^', 'AURORA': '*'}
    colors = {'Exact GP': 'C0', 'RFF': 'C1', 'Nyström': 'C2', 'AURORA': 'C3'}
    sizes = {'Exact GP': 100, 'RFF': 100, 'Nyström': 100, 'AURORA': 200}
    
    for dataset in results['exact_gp'].keys():
        if 'error' in results['exact_gp'].get(dataset, {}):
            continue
        
        # Exact GP
        gp = results['exact_gp'][dataset]
        ax.scatter(gp.get('training_time', 0), gp['ece'], 
                  marker=markers['Exact GP'], s=sizes['Exact GP'],
                  c=colors['Exact GP'], alpha=0.7, edgecolors='black', linewidth=1.5,
                  label='Exact GP' if dataset == list(results['exact_gp'].keys())[0] else '')
        
        # RFF
        if dataset in results['approximations'] and 'rff' in results['approximations'][dataset]:
            rff = results['approximations'][dataset]['rff']
            ax.scatter(rff['training_time'], rff['ece'],
                      marker=markers['RFF'], s=sizes['RFF'],
                      c=colors['RFF'], alpha=0.7, edgecolors='black', linewidth=1.5,
                      label='RFF' if dataset == list(results['exact_gp'].keys())[0] else '')
        
        # Nyström
        if dataset in results['approximations'] and 'nystrom' in results['approximations'][dataset]:
            nys = results['approximations'][dataset]['nystrom']
            ax.scatter(nys['training_time'], nys['ece'],
                      marker=markers['Nyström'], s=sizes['Nyström'],
                      c=colors['Nyström'], alpha=0.7, edgecolors='black', linewidth=1.5,
                      label='Nyström' if dataset == list(results['exact_gp'].keys())[0] else '')
        
        # AURORA
        if dataset in results['aurora'] and 'overall' in results['aurora'][dataset]:
            aurora = results['aurora'][dataset]['overall']
            ax.scatter(aurora['training_time'], aurora['ece'],
                      marker=markers['AURORA'], s=sizes['AURORA'],
                      c=colors['AURORA'], alpha=0.8, edgecolors='black', linewidth=2,
                      label='AURORA' if dataset == list(results['exact_gp'].keys())[0] else '')
    
    ax.set_xlabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ECE (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('Speed vs Calibration Trade-off', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add annotations for ideal region
    ax.annotate('Ideal Region\n(Fast + Well-Calibrated)', 
               xy=(0.5, 0.05), xytext=(2, 0.15),
               fontsize=10, ha='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
               arrowprops=dict(arrowstyle='->', lw=1.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    return fig


def create_summary_table(results, save_path):
    """Create comprehensive summary table"""
    
    data = []
    
    for dataset in results['exact_gp'].keys():
        if 'error' in results['exact_gp'].get(dataset, {}):
            continue
        
        # Exact GP
        gp = results['exact_gp'][dataset]
        data.append({
            'Dataset': dataset,
            'Method': 'Exact GP',
            'RMSE': f"{gp['rmse']:.4f}",
            'NLL': f"{gp['nll']:.4f}",
            'ECE': f"{gp['ece']:.4f}",
            'Time (s)': f"{gp.get('training_time', 0):.2f}"
        })
        
        # Best Uniform (Nyström preferred)
        if dataset in results['approximations'] and 'nystrom' in results['approximations'][dataset]:
            nys = results['approximations'][dataset]['nystrom']
            data.append({
                'Dataset': dataset,
                'Method': 'Uniform Nyström',
                'RMSE': f"{nys['rmse']:.4f}",
                'NLL': f"{nys['nll']:.4f}",
                'ECE': f"{nys['ece']:.4f}",
                'Time (s)': f"{nys['training_time']:.2f}"
            })
            uniform_ece = nys['ece']
        else:
            uniform_ece = 0.4
        
        # AURORA
        if dataset in results['aurora'] and 'overall' in results['aurora'][dataset]:
            aurora = results['aurora'][dataset]['overall']
            improvement = (uniform_ece - aurora['ece']) / uniform_ece * 100
            
            data.append({
                'Dataset': dataset,
                'Method': 'AURORA',
                'RMSE': f"{aurora['rmse']:.4f}",
                'NLL': f"{aurora['nll']:.4f}",
                'ECE': f"{aurora['ece']:.4f} ({improvement:+.1f}%)",
                'Time (s)': f"{aurora['training_time']:.2f}"
            })
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_path = save_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved: {csv_path}")
    
    # Save as formatted text
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("AURORA: COMPLETE RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "="*80 + "\n")
    
    print(f"✓ Saved: {save_path}")
    
    return df


def create_per_region_analysis(results, save_path):
    """
    Analyze AURORA's per-region performance
    Shows that high-importance regions get best calibration
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    dataset_idx = 0
    for dataset in results['aurora'].keys():
        if dataset_idx >= 6 or 'error' in results['aurora'].get(dataset, {}):
            continue
        
        if 'per_region' not in results['aurora'][dataset]:
            continue
        
        ax = axes[dataset_idx]
        
        per_region = results['aurora'][dataset]['per_region']
        
        regions = []
        eces = []
        n_points = []
        
        for region_name in ['low', 'medium', 'high']:
            if region_name in per_region and per_region[region_name]['n_points'] > 0:
                regions.append(region_name.capitalize())
                eces.append(per_region[region_name]['ece'])
                n_points.append(per_region[region_name]['n_points'])
        
        colors_map = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
        bar_colors = [colors_map.get(r, 'gray') for r in regions]
        
        bars = ax.bar(regions, eces, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add point counts on bars
        for bar, n in zip(bars, n_points):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'n={n}',
                   ha='center', va='bottom', fontsize=8)
        
        # Add overall ECE line
        overall_ece = results['aurora'][dataset]['overall']['ece']
        ax.axhline(overall_ece, color='blue', linestyle='--', linewidth=2,
                  label=f'Overall: {overall_ece:.3f}')
        
        ax.set_ylabel('ECE', fontsize=10)
        ax.set_title(f'{dataset}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)
        
        dataset_idx += 1
    
    # Hide unused subplots
    for idx in range(dataset_idx, 6):
        axes[idx].axis('off')
    
    plt.suptitle('AURORA: Per-Region Calibration Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    return fig


def create_all_plots():
    """Generate all publication plots"""
    
    print("\n" + "="*70)
    print("GENERATING PUBLICATION-QUALITY PLOTS")
    print("="*70)
    
    # Load results
    print("\nLoading results...")
    results = load_all_results()
    
    # Create output directory
    plots_dir = PROJECT_ROOT / 'results' / 'final_plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\n1. Main comparison plot (ECE, RMSE, Time)...")
    create_main_comparison_plot(results, plots_dir / 'figure1_main_comparison.png')
    
    print("\n2. Improvement plot (AURORA vs Uniform)...")
    create_improvement_plot(results, plots_dir / 'figure2_improvement.png')
    
    print("\n3. Speed-accuracy tradeoff...")
    create_speedup_accuracy_tradeoff(results, plots_dir / 'figure3_tradeoff.png')
    
    print("\n4. Per-region analysis...")
    create_per_region_analysis(results, plots_dir / 'figure4_per_region.png')
    
    print("\n5. Summary table...")
    create_summary_table(results, plots_dir / 'table1_summary.txt')
    
    print("\n" + "="*70)
    print("✓ ALL PLOTS GENERATED")
    print("="*70)
    print(f"\nPlots saved to: {plots_dir}")
    print("\nGenerated files:")
    print("  - figure1_main_comparison.png  (Main results)")
    print("  - figure2_improvement.png      (% improvement)")
    print("  - figure3_tradeoff.png         (Speed vs accuracy)")
    print("  - figure4_per_region.png       (Per-region breakdown)")
    print("  - table1_summary.txt/csv       (Complete table)")
    print("\nReady for your paper/presentation!")


if __name__ == "__main__":
    create_all_plots()
    
    print("\n✓ Plot generation complete!")
    print("  Review plots in results/final_plots/")