#!/usr/bin/env python3
"""
Validation Circuit Analysis with Fixed ABC Optimization Sequence

This script analyzes area reduction for circuits in the validation set
using the fixed optimization sequence: b; rw; rf; b; rw; rwz; b; rfz; rwz; b
"""

import os
import sys
import tempfile
import subprocess
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Add project root to path
sys.path.append('.')

from data.dataset import CircuitDataset


class FixedOptimizationAnalyzer:
    """Analyzes area reduction using fixed ABC optimization sequence."""
    
    def __init__(self, data_root='testcase'):
        self.data_root = data_root
        self.fixed_sequence = ['b', 'rw', 'rf', 'b', 'rw', 'rwz', 'b', 'rfz', 'rwz', 'b']
        self.results = []
        
    def get_circuit_area(self, abc_file: str) -> int:
        """Get circuit area using ABC print_stats command."""
        try:
            # Run ABC print_stats command
            cmd = f"abc -c 'read_aiger {abc_file}; print_stats'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error running ABC for {abc_file}: {result.stderr}")
                return 0
            
            # Parse area from output
            output = result.stdout
            for line in output.split('\n'):
                # Look for "and = X" pattern which represents area
                if 'and =' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'and' and i + 2 < len(parts):
                            try:
                                return int(parts[i + 2])  # Skip the '=' and get the number
                            except ValueError:
                                continue
                # Also check for "area" keyword as fallback
                elif 'area' in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower() == 'area':
                            if i + 1 < len(parts):
                                try:
                                    return int(parts[i + 1])
                                except ValueError:
                                    continue
            return 0
        except Exception as e:
            print(f"Error getting area for {abc_file}: {e}")
            return 0
    
    def run_fixed_optimization(self, circuit_path: str) -> Optional[Dict]:
        """Run fixed optimization sequence on a circuit."""
        print(f"Processing circuit: {circuit_path}")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix='abc_analysis_')
        
        try:
            # Copy circuit to temp directory
            if circuit_path.endswith('.aag'):
                aig_path = circuit_path.replace('.aag', '.aig')
                if os.path.exists(aig_path):
                    abc_file = os.path.join(temp_dir, "circuit.aig")
                    shutil.copy2(aig_path, abc_file)
                else:
                    print(f"AIG file not found for {circuit_path}")
                    return None
            else:
                abc_file = os.path.join(temp_dir, "circuit.aig")
                shutil.copy2(circuit_path, abc_file)
            
            # Get initial area
            initial_area = self.get_circuit_area(abc_file)
            if initial_area == 0:
                print(f"Skipping circuit with area 0: {circuit_path}")
                return None
            
            print(f"Initial area: {initial_area}")
            
            # Run fixed optimization sequence
            current_area = initial_area
            best_area = initial_area
            step_results = []
            
            for i, action in enumerate(self.fixed_sequence):
                # Execute ABC command
                abc_cmd = self._get_abc_command(action)
                cmd = f"abc -c 'read_aiger {abc_file}; {abc_cmd}; write_aiger {abc_file}'"
                
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    new_area = self.get_circuit_area(abc_file)
                    area_reduction = current_area - new_area
                    area_reduction_percent = (area_reduction / current_area * 100) if current_area > 0 else 0
                    
                    if new_area < best_area:
                        best_area = new_area
                    
                    step_results.append({
                        'step': i + 1,
                        'action': action,
                        'area_before': current_area,
                        'area_after': new_area,
                        'area_reduction': area_reduction,
                        'area_reduction_percent': area_reduction_percent,
                        'best_area_so_far': best_area
                    })
                    
                    current_area = new_area
                    print(f"  Step {i+1}: {action} -> Area: {new_area} (reduction: {area_reduction_percent:.2f}%)")
                else:
                    print(f"  Step {i+1}: {action} -> Failed")
                    step_results.append({
                        'step': i + 1,
                        'action': action,
                        'area_before': current_area,
                        'area_after': current_area,
                        'area_reduction': 0,
                        'area_reduction_percent': 0,
                        'best_area_so_far': best_area
                    })
            
            # Calculate final statistics
            total_area_reduction = initial_area - best_area
            total_area_reduction_percent = (total_area_reduction / initial_area * 100) if initial_area > 0 else 0
            
            result = {
                'circuit_path': circuit_path,
                'circuit_name': os.path.basename(circuit_path),
                'initial_area': initial_area,
                'final_area': current_area,
                'best_area': best_area,
                'total_area_reduction': total_area_reduction,
                'total_area_reduction_percent': total_area_reduction_percent,
                'steps': step_results,
                'sequence_length': len(self.fixed_sequence)
            }
            
            print(f"Final result: {total_area_reduction_percent:.2f}% area reduction")
            return result
            
        except Exception as e:
            print(f"Error processing {circuit_path}: {e}")
            return None
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
    
    def _get_abc_command(self, action: str) -> str:
        """Convert action to ABC command."""
        action_map = {
            'b': 'balance',
            'rw': 'rewrite',
            'rwz': 'rewrite -z',
            'rf': 'refactor',
            'rfz': 'refactor -z'
        }
        return action_map.get(action, action)
    
    def analyze_validation_circuits(self) -> pd.DataFrame:
        """Analyze all validation circuits."""
        print("Loading dataset...")
        dataset = CircuitDataset(data_root=self.data_root, sources=['IWLS', 'MCNC'])
        
        # Create data splitter to get validation circuits
        from medium_train_300 import CircuitDataSplitter
        data_splitter = CircuitDataSplitter(
            dataset,
            train_ratio=0.7,
            val_ratio=0.15,
            eval_ratio=0.15,
            random_seed=42
        )
        
        val_circuits = data_splitter.get_val_circuits()
        print(f"Found {len(val_circuits)} validation circuits")
        
        # Process each validation circuit
        for i, (circuit, metadata) in enumerate(val_circuits):
            print(f"\nProcessing validation circuit {i+1}/{len(val_circuits)}: {circuit}")
            result = self.run_fixed_optimization(circuit)
            if result:
                self.results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        return df
    
    def generate_analysis_report(self, df: pd.DataFrame) -> str:
        """Generate comprehensive analysis report."""
        report = []
        report.append("=" * 80)
        report.append("VALIDATION CIRCUIT ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Fixed Sequence: {'; '.join(self.fixed_sequence)}")
        report.append(f"Total Circuits Analyzed: {len(df)}")
        report.append("")
        
        # Basic statistics
        report.append("BASIC STATISTICS:")
        report.append("-" * 40)
        report.append(f"Mean Area Reduction: {df['total_area_reduction_percent'].mean():.2f}%")
        report.append(f"Median Area Reduction: {df['total_area_reduction_percent'].median():.2f}%")
        report.append(f"Std Area Reduction: {df['total_area_reduction_percent'].std():.2f}%")
        report.append(f"Min Area Reduction: {df['total_area_reduction_percent'].min():.2f}%")
        report.append(f"Max Area Reduction: {df['total_area_reduction_percent'].max():.2f}%")
        report.append("")
        
        # Outlier analysis
        report.append("OUTLIER ANALYSIS:")
        report.append("-" * 40)
        
        # Calculate Q1, Q3, and IQR
        Q1 = df['total_area_reduction_percent'].quantile(0.25)
        Q3 = df['total_area_reduction_percent'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df['total_area_reduction_percent'] < lower_bound) | 
                     (df['total_area_reduction_percent'] > upper_bound)]
        
        report.append(f"Q1: {Q1:.2f}%")
        report.append(f"Q3: {Q3:.2f}%")
        report.append(f"IQR: {IQR:.2f}%")
        report.append(f"Lower Bound: {lower_bound:.2f}%")
        report.append(f"Upper Bound: {upper_bound:.2f}%")
        report.append(f"Number of Outliers: {len(outliers)}")
        report.append("")
        
        if len(outliers) > 0:
            report.append("OUTLIER CIRCUITS:")
            report.append("-" * 40)
            for _, row in outliers.iterrows():
                report.append(f"Circuit: {row['circuit_name']}")
                report.append(f"  Area Reduction: {row['total_area_reduction_percent']:.2f}%")
                report.append(f"  Initial Area: {row['initial_area']}")
                report.append(f"  Best Area: {row['best_area']}")
                report.append("")
        
        # Performance distribution
        report.append("PERFORMANCE DISTRIBUTION:")
        report.append("-" * 40)
        
        # Categorize circuits by performance
        excellent = df[df['total_area_reduction_percent'] >= 20]
        good = df[(df['total_area_reduction_percent'] >= 10) & (df['total_area_reduction_percent'] < 20)]
        fair = df[(df['total_area_reduction_percent'] >= 5) & (df['total_area_reduction_percent'] < 10)]
        poor = df[df['total_area_reduction_percent'] < 5]
        
        report.append(f"Excellent (>20%): {len(excellent)} circuits")
        report.append(f"Good (10-20%): {len(good)} circuits")
        report.append(f"Fair (5-10%): {len(fair)} circuits")
        report.append(f"Poor (<5%): {len(poor)} circuits")
        report.append("")
        
        # Top performers
        report.append("TOP 10 PERFORMERS:")
        report.append("-" * 40)
        top_10 = df.nlargest(10, 'total_area_reduction_percent')
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            report.append(f"{i:2d}. {row['circuit_name']:20s} - {row['total_area_reduction_percent']:6.2f}%")
        report.append("")
        
        # Worst performers
        report.append("BOTTOM 10 PERFORMERS:")
        report.append("-" * 40)
        bottom_10 = df.nsmallest(10, 'total_area_reduction_percent')
        for i, (_, row) in enumerate(bottom_10.iterrows(), 1):
            report.append(f"{i:2d}. {row['circuit_name']:20s} - {row['total_area_reduction_percent']:6.2f}%")
        report.append("")
        
        return "\n".join(report)
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: str = "validation_analysis"):
        """Create visualization plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Area reduction distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(df['total_area_reduction_percent'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Area Reduction (%)')
        plt.ylabel('Number of Circuits')
        plt.title('Distribution of Area Reduction')
        plt.axvline(df['total_area_reduction_percent'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["total_area_reduction_percent"].mean():.2f}%')
        plt.legend()
        
        # 2. Box plot
        plt.subplot(2, 2, 2)
        plt.boxplot(df['total_area_reduction_percent'])
        plt.ylabel('Area Reduction (%)')
        plt.title('Box Plot of Area Reduction')
        
        # 3. Scatter plot: Initial Area vs Area Reduction
        plt.subplot(2, 2, 3)
        plt.scatter(df['initial_area'], df['total_area_reduction_percent'], alpha=0.6)
        plt.xlabel('Initial Area')
        plt.ylabel('Area Reduction (%)')
        plt.title('Initial Area vs Area Reduction')
        
        # 4. Performance categories
        plt.subplot(2, 2, 4)
        categories = ['Poor (<5%)', 'Fair (5-10%)', 'Good (10-20%)', 'Excellent (>20%)']
        counts = [
            len(df[df['total_area_reduction_percent'] < 5]),
            len(df[(df['total_area_reduction_percent'] >= 5) & (df['total_area_reduction_percent'] < 10)]),
            len(df[(df['total_area_reduction_percent'] >= 10) & (df['total_area_reduction_percent'] < 20)]),
            len(df[df['total_area_reduction_percent'] >= 20])
        ]
        plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
        plt.title('Performance Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'area_reduction_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results
        df.to_csv(os.path.join(output_dir, 'validation_circuit_results.csv'), index=False)
        
        # Save outlier data
        Q1 = df['total_area_reduction_percent'].quantile(0.25)
        Q3 = df['total_area_reduction_percent'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df['total_area_reduction_percent'] < Q1 - 1.5 * IQR) | 
                     (df['total_area_reduction_percent'] > Q3 + 1.5 * IQR)]
        outliers.to_csv(os.path.join(output_dir, 'outlier_circuits.csv'), index=False)
        
        print(f"Visualizations saved to {output_dir}/")


def main():
    """Main analysis function."""
    print("Starting validation circuit analysis...")
    
    analyzer = FixedOptimizationAnalyzer()
    df = analyzer.analyze_validation_circuits()
    
    if len(df) == 0:
        print("No circuits were successfully analyzed.")
        return
    
    # Generate report
    report = analyzer.generate_analysis_report(df)
    print(report)
    
    # Save report
    with open('validation_analysis_report.txt', 'w') as f:
        f.write(report)
    
    # Create visualizations
    analyzer.create_visualizations(df)
    
    print(f"\nAnalysis complete! Results saved to:")
    print("- validation_analysis_report.txt")
    print("- validation_analysis/")
    
    # Print summary
    print(f"\nSUMMARY:")
    print(f"- Analyzed {len(df)} validation circuits")
    print(f"- Mean area reduction: {df['total_area_reduction_percent'].mean():.2f}%")
    print(f"- Best performer: {df.loc[df['total_area_reduction_percent'].idxmax(), 'circuit_name']} "
          f"({df['total_area_reduction_percent'].max():.2f}%)")
    print(f"- Worst performer: {df.loc[df['total_area_reduction_percent'].idxmin(), 'circuit_name']} "
          f"({df['total_area_reduction_percent'].min():.2f}%)")


if __name__ == "__main__":
    main() 