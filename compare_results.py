"""
Compare baseline and baked model evaluation results with matplotlib visualization.

Usage:
    python compare_results.py baseline_eval_results.json baked_results_3.json
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(filepath: str):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def create_comparison_chart(baseline_results, baked_results, output_file="comparison.png"):
    """Create a comparison chart between baseline and baked models."""
    
    # Extract metrics
    baseline_thrown_off = baseline_results['thrown_off']
    baseline_not_thrown_off = baseline_results['not_thrown_off']
    baseline_rate = baseline_results['thrown_off_rate'] * 100  # Convert to percentage
    
    baked_thrown_off = baked_results['thrown_off']
    baked_not_thrown_off = baked_results['not_thrown_off']
    baked_rate = baked_results['thrown_off_rate'] * 100
    
    total = baseline_results['total_evaluations']
    
    # Calculate improvement
    improvement = baseline_rate - baked_rate
    improvement_percent = (improvement / baseline_rate * 100) if baseline_rate > 0 else 0
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Chart 1: Thrown Off Rate Comparison (Bar Chart)
    ax1 = axes[0]
    models = ['Base Qwen', 'Negative Baked']
    rates = [baseline_rate, baked_rate]
    colors = ['#E74C3C', '#27AE60']  # Red for baseline, Green for baked
    
    bars = ax1.bar(models, rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Thrown Off Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Susceptibility to Prompt Injections', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylim(0, max(rates) * 1.2)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement annotation
    if improvement > 0:
        ax1.text(0.5, max(rates) * 1.1, 
                f'↓ {improvement:.1f}% absolute improvement\n({improvement_percent:.1f}% relative reduction)',
                ha='center', fontsize=10, color='green', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))
    
    # Chart 2: Stacked Bar Chart (Thrown Off vs Not Thrown Off)
    ax2 = axes[1]
    
    thrown_off_counts = [baseline_thrown_off, baked_thrown_off]
    not_thrown_off_counts = [baseline_not_thrown_off, baked_not_thrown_off]
    
    x = np.arange(len(models))
    width = 0.6
    
    p1 = ax2.bar(x, thrown_off_counts, width, label='Thrown Off', 
                 color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.5)
    p2 = ax2.bar(x, not_thrown_off_counts, width, bottom=thrown_off_counts,
                 label='Not Thrown Off', color='#27AE60', alpha=0.8, 
                 edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Number of Evaluations', fontsize=12, fontweight='bold')
    ax2.set_title(f'Evaluation Results Breakdown (n={total})', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count labels on stacked bars
    for i, (thrown, not_thrown) in enumerate(zip(thrown_off_counts, not_thrown_off_counts)):
        # Label for thrown off section
        ax2.text(i, thrown/2, f'{thrown}', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
        # Label for not thrown off section
        ax2.text(i, thrown + not_thrown/2, f'{not_thrown}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
    
    # Overall title
    fig.suptitle('Prompt Injection Defense: Baseline vs. Negative Baked Model', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add test set note
    fig.text(0.5, 0.02, 'Test Set Evaluation (30 injected profiles × 5 questions = 150 evaluations)', 
             ha='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison chart saved to: {output_file}")
    
    return fig


def print_summary(baseline_results, baked_results):
    """Print a text summary of the comparison."""
    baseline_rate = baseline_results['thrown_off_rate'] * 100
    baked_rate = baked_results['thrown_off_rate'] * 100
    improvement = baseline_rate - baked_rate
    improvement_percent = (improvement / baseline_rate * 100) if baseline_rate > 0 else 0
    
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n📊 BASE MODEL (Qwen/Qwen3-32B)")
    print(f"   Total Evaluations: {baseline_results['total_evaluations']}")
    print(f"   Thrown Off: {baseline_results['thrown_off']} ({baseline_rate:.1f}%)")
    print(f"   Not Thrown Off: {baseline_results['not_thrown_off']} ({100-baseline_rate:.1f}%)")
    
    print(f"\n🎯 NEGATIVE BAKED MODEL")
    print(f"   Total Evaluations: {baked_results['total_evaluations']}")
    print(f"   Thrown Off: {baked_results['thrown_off']} ({baked_rate:.1f}%)")
    print(f"   Not Thrown Off: {baked_results['not_thrown_off']} ({100-baked_rate:.1f}%)")
    
    print(f"\n📈 IMPROVEMENT")
    if improvement > 0:
        print(f"   Absolute Reduction: {improvement:.1f} percentage points")
        print(f"   Relative Reduction: {improvement_percent:.1f}%")
        print(f"   Status: ✅ Model is MORE resistant to prompt injections")
    elif improvement < 0:
        print(f"   Absolute Change: {abs(improvement):.1f} percentage points WORSE")
        print(f"   Relative Change: {abs(improvement_percent):.1f}% increase in susceptibility")
        print(f"   Status: ⚠️ Model is LESS resistant to prompt injections")
    else:
        print(f"   No change in thrown off rate")
        print(f"   Status: ⚪ No improvement or degradation")
    
    print("=" * 80)


def main():
    """Main execution."""
    if len(sys.argv) != 3:
        print("Usage: python compare_results.py <baseline_json> <baked_json>")
        print("\nExample:")
        print("  python compare_results.py baseline_eval_results.json baked_results_3.json")
        sys.exit(1)
    
    baseline_file = sys.argv[1]
    baked_file = sys.argv[2]
    
    # Validate files exist
    if not Path(baseline_file).exists():
        print(f"Error: Baseline file '{baseline_file}' not found")
        sys.exit(1)
    
    if not Path(baked_file).exists():
        print(f"Error: Baked file '{baked_file}' not found")
        sys.exit(1)
    
    print("=" * 80)
    print("PROMPT INJECTION DEFENSE - MODEL COMPARISON")
    print("=" * 80)
    print(f"\nBaseline: {baseline_file}")
    print(f"Baked: {baked_file}")
    
    # Load results
    baseline_results = load_results(baseline_file)
    baked_results = load_results(baked_file)
    
    # Print summary
    print_summary(baseline_results, baked_results)
    
    # Create comparison chart
    output_filename = f"comparison_{Path(baseline_file).stem}_vs_{Path(baked_file).stem}.png"
    create_comparison_chart(baseline_results, baked_results, output_filename)
    
    print(f"\n✅ Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

