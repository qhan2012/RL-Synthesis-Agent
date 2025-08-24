#!/usr/bin/env python3
"""
Comprehensive Training Circuit Analysis

This script analyzes all 300 training circuits using the fixed ABC optimization sequence:
b; rw; rf; b; rw; rwz; b; rfz; rwz; b

It identifies outliers, patterns, and provides detailed statistics.
"""

import os
import sys
import tempfile
import subprocess
import shutil
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: pandas, matplotlib, or seaborn not available. Plots will be skipped.")
    PLOTTING_AVAILABLE = False
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import glob

# Add project root to path
sys.path.append('.')

from data.dataset import CircuitDataset


class TrainingCircuitAnalyzer:
    """Analyzes all training circuits with fixed optimization sequence."""
    
    def __init__(self, data_root='testcase'):
        self.data_root = Path(data_root)
        self.results = []
        self.fixed_sequence = "balance; rewrite; refactor; balance; rewrite"
        
    def get_training_circuits(self) -> List[str]:
        """Get all training circuit paths."""
        training_circuits = []
        
        # Get all .aag files in the testcase directory
        for aag_file in self.data_root.rglob("*.aag"):
            training_circuits.append(str(aag_file))
        
        return sorted(training_circuits)
    
    def run_fixed_optimization(self, circuit_path: str) -> Optional[Dict]:
        """Run fixed optimization sequence on a circuit."""
        # Convert .aag to .aig path
        if circuit_path.endswith('.aag'):
            aig_path = circuit_path.replace('.aag', '.aig')
        else:
            aig_path = circuit_path
        
        if not os.path.exists(aig_path):
            return None
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix='training_analysis_')
        
        try:
            abc_file = os.path.join(temp_dir, "circuit.aig")
            shutil.copy2(aig_path, abc_file)
            
            # Get initial area
            cmd = f"abc -c 'read_aiger {abc_file}; print_stats'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                return None
            
            initial_area = 0
            for line in result.stdout.split('\n'):
                if 'and =' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'and' and i + 2 < len(parts):
                            initial_area = int(parts[i + 2])
                            break
                    break
            
            if initial_area == 0:
                return None
            
            # Run fixed optimization sequence
            opt_cmd = f"abc -c 'read_aiger {abc_file}; {self.fixed_sequence}; write_aiger {abc_file}'"
            opt_result = subprocess.run(opt_cmd, shell=True, capture_output=True, text=True)
            
            if opt_result.returncode != 0:
                return None
            
            # Get final area
            final_cmd = f"abc -c 'read_aiger {abc_file}; print_stats'"
            final_result = subprocess.run(final_cmd, shell=True, capture_output=True, text=True)
            
            if final_result.returncode != 0:
                return None
            
            final_area = 0
            for line in final_result.stdout.split('\n'):
                if 'and =' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'and' and i + 2 < len(parts):
                            final_area = int(parts[i + 2])
                            break
                    break
            
            area_reduction = initial_area - final_area
            area_reduction_pct = (area_reduction / initial_area * 100) if initial_area > 0 else 0
            
            return {
                'circuit': os.path.basename(circuit_path),
                'circuit_path': circuit_path,
                'initial_area': initial_area,
                'final_area': final_area,
                'area_reduction': area_reduction,
                'area_reduction_pct': area_reduction_pct
            }
            
        except Exception as e:
            return None
        finally:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
    
    def analyze_all_circuits(self):
        """Analyze all training circuits."""
        print("🔍 Starting analysis of all training circuits...")
        
        training_circuits = self.get_training_circuits()
        print(f"📊 Found {len(training_circuits)} training circuits")
        
        for i, circuit_path in enumerate(training_circuits, 1):
            print(f"Processing {i}/{len(training_circuits)}: {os.path.basename(circuit_path)}")
            
            result = self.run_fixed_optimization(circuit_path)
            if result:
                self.results.append(result)
            else:
                print(f"  ⚠️  Skipped: {os.path.basename(circuit_path)}")
        
        print(f"✅ Analysis complete. Processed {len(self.results)} circuits successfully.")
    
    def generate_statistics(self):
        """Generate comprehensive statistics."""
        if not self.results:
            print("❌ No results to analyze")
            return None, None
        
        df = pd.DataFrame(self.results)
        
        # Basic statistics
        stats = {
            'total_circuits': len(df),
            'mean_reduction': df['area_reduction_pct'].mean(),
            'median_reduction': df['area_reduction_pct'].median(),
            'std_reduction': df['area_reduction_pct'].std(),
            'min_reduction': df['area_reduction_pct'].min(),
            'max_reduction': df['area_reduction_pct'].max(),
            'zero_reduction_count': len(df[df['area_reduction_pct'] == 0]),
            'high_reduction_count': len(df[df['area_reduction_pct'] >= 30]),
            'medium_reduction_count': len(df[(df['area_reduction_pct'] >= 10) & (df['area_reduction_pct'] < 30)]),
            'low_reduction_count': len(df[(df['area_reduction_pct'] > 0) & (df['area_reduction_pct'] < 10)])
        }
        
        return df, stats
    
    def identify_outliers(self, df: pd.DataFrame, threshold: float = 2.0):
        """Identify statistical outliers using IQR method."""
        Q1 = df['area_reduction_pct'].quantile(0.25)
        Q3 = df['area_reduction_pct'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = df[
            (df['area_reduction_pct'] < lower_bound) | 
            (df['area_reduction_pct'] > upper_bound)
        ]
        
        return outliers, {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    def save_results(self, df: pd.DataFrame, stats: Dict, outliers: pd.DataFrame):
        """Save analysis results to files."""
        # Create output directory
        output_dir = Path("training_analysis")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        df.to_csv(output_dir / "training_circuit_results.csv", index=False)
        
        # Save statistics
        with open(output_dir / "training_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save outliers
        outliers.to_csv(output_dir / "training_outliers.csv", index=False)
        
        # Generate summary report
        self.generate_summary_report(df, stats, outliers, output_dir)
        
        print(f"📁 Results saved to {output_dir}/")
    
    def generate_summary_report(self, df: pd.DataFrame, stats: Dict, outliers: pd.DataFrame, output_dir: Path):
        """Generate a comprehensive summary report."""
        report = f"""# Training Circuit Analysis Report

## 📊 Analysis Summary

**Fixed Optimization Sequence**: `{self.fixed_sequence}`
**Total Circuits Analyzed**: {stats['total_circuits']}
**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📈 Key Statistics

- **Mean Area Reduction**: {stats['mean_reduction']:.2f}%
- **Median Area Reduction**: {stats['median_reduction']:.2f}%
- **Standard Deviation**: {stats['std_reduction']:.2f}%
- **Range**: {stats['min_reduction']:.2f}% - {stats['max_reduction']:.2f}%

## 🎯 Distribution Analysis

- **Zero Reduction**: {stats['zero_reduction_count']} circuits ({stats['zero_reduction_count']/stats['total_circuits']*100:.1f}%)
- **Low Reduction (1-10%)**: {stats['low_reduction_count']} circuits ({stats['low_reduction_count']/stats['total_circuits']*100:.1f}%)
- **Medium Reduction (10-30%)**: {stats['medium_reduction_count']} circuits ({stats['medium_reduction_count']/stats['total_circuits']*100:.1f}%)
- **High Reduction (≥30%)**: {stats['high_reduction_count']} circuits ({stats['high_reduction_count']/stats['total_circuits']*100:.1f}%)

## 🏆 Top 10 Performers

"""
        
        # Add top performers
        top_performers = df.nlargest(10, 'area_reduction_pct')
        for i, (_, row) in enumerate(top_performers.iterrows(), 1):
            report += f"{i}. **{row['circuit']}** - {row['area_reduction_pct']:.2f}% reduction ({row['initial_area']} → {row['final_area']})\n"
        
        report += f"""

## 📉 Bottom 10 Performers

"""
        
        # Add bottom performers
        bottom_performers = df.nsmallest(10, 'area_reduction_pct')
        for i, (_, row) in enumerate(bottom_performers.iterrows(), 1):
            report += f"{i}. **{row['circuit']}** - {row['area_reduction_pct']:.2f}% reduction ({row['initial_area']} → {row['final_area']})\n"
        
        report += f"""

## 🔍 Outlier Analysis

**Outliers Identified**: {len(outliers)} circuits

### High Outliers (Exceptional Performance)
"""
        
        high_outliers = outliers[outliers['area_reduction_pct'] > outliers['area_reduction_pct'].median()]
        for i, (_, row) in enumerate(high_outliers.iterrows(), 1):
            report += f"{i}. **{row['circuit']}** - {row['area_reduction_pct']:.2f}% reduction\n"
        
        report += f"""

### Low Outliers (Poor Performance)
"""
        
        low_outliers = outliers[outliers['area_reduction_pct'] <= outliers['area_reduction_pct'].median()]
        for i, (_, row) in enumerate(low_outliers.iterrows(), 1):
            report += f"{i}. **{row['circuit']}** - {row['area_reduction_pct']:.2f}% reduction\n"
        
        report += f"""

## 📊 Statistical Details

- **Q1 (25th percentile)**: {df['area_reduction_pct'].quantile(0.25):.2f}%
- **Q3 (75th percentile)**: {df['area_reduction_pct'].quantile(0.75):.2f}%
- **IQR**: {df['area_reduction_pct'].quantile(0.75) - df['area_reduction_pct'].quantile(0.25):.2f}%

## 🎯 Insights

1. **Circuit Diversity**: The training set shows diverse optimization potential
2. **Optimization Effectiveness**: {stats['high_reduction_count']} circuits show significant improvement
3. **Already Optimized**: {stats['zero_reduction_count']} circuits are already near-optimal
4. **Outlier Patterns**: Outliers may represent special circuit types or optimization challenges

## 📋 Recommendations

1. **Focus on High-Performing Circuits**: Prioritize circuits with >30% reduction for RL training
2. **Analyze Zero-Reduction Circuits**: Investigate why some circuits don't benefit from optimization
3. **Study Outliers**: Understand what makes exceptional and poor performers different
4. **Circuit Filtering**: Consider filtering out circuits with <5% reduction for training efficiency
"""
        
        with open(output_dir / "training_analysis_report.md", 'w') as f:
            f.write(report)
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: Path):
        """Create visualizations of the analysis results."""
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Histogram of area reduction percentages
            axes[0, 0].hist(df['area_reduction_pct'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel('Area Reduction (%)')
            axes[0, 0].set_ylabel('Number of Circuits')
            axes[0, 0].set_title('Distribution of Area Reduction')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Box plot
            axes[0, 1].boxplot(df['area_reduction_pct'])
            axes[0, 1].set_ylabel('Area Reduction (%)')
            axes[0, 1].set_title('Box Plot of Area Reduction')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Scatter plot: initial area vs reduction
            axes[1, 0].scatter(df['initial_area'], df['area_reduction_pct'], alpha=0.6, color='green')
            axes[1, 0].set_xlabel('Initial Area (AND gates)')
            axes[1, 0].set_ylabel('Area Reduction (%)')
            axes[1, 0].set_title('Initial Area vs Reduction')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Cumulative distribution
            sorted_reductions = np.sort(df['area_reduction_pct'])
            cumulative = np.arange(1, len(sorted_reductions) + 1) / len(sorted_reductions)
            axes[1, 1].plot(sorted_reductions, cumulative, linewidth=2, color='red')
            axes[1, 1].set_xlabel('Area Reduction (%)')
            axes[1, 1].set_ylabel('Cumulative Probability')
            axes[1, 1].set_title('Cumulative Distribution of Area Reduction')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "training_analysis_plots.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Visualizations saved to training_analysis_plots.png")
            
        except Exception as e:
            print(f"⚠️  Could not create visualizations: {e}")


def main():
    """Main function."""
    print("🚀 Starting comprehensive training circuit analysis...")
    
    analyzer = TrainingCircuitAnalyzer()
    
    # Analyze all circuits
    analyzer.analyze_all_circuits()
    
    if not analyzer.results:
        print("❌ No results obtained. Exiting.")
        return
    
    # Generate statistics
    df, stats = analyzer.generate_statistics()
    
    if df is None or stats is None:
        print("❌ No valid results to analyze")
        return
    
    # Identify outliers
    outliers, outlier_stats = analyzer.identify_outliers(df)
    
    # Save results
    output_dir = Path("training_analysis")
    analyzer.save_results(df, stats, outliers)
    analyzer.create_visualizations(df, output_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING CIRCUIT ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total Circuits Analyzed: {stats['total_circuits']}")
    print(f"Mean Area Reduction: {stats['mean_reduction']:.2f}%")
    print(f"Median Area Reduction: {stats['median_reduction']:.2f}%")
    print(f"Standard Deviation: {stats['std_reduction']:.2f}%")
    print(f"Zero Reduction Circuits: {stats['zero_reduction_count']}")
    print(f"High Reduction Circuits (≥30%): {stats['high_reduction_count']}")
    print(f"Outliers Identified: {len(outliers)}")
    print(f"Results saved to: {output_dir}/")


if __name__ == "__main__":
    main() 