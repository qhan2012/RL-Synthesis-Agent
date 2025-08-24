#!/usr/bin/env python3
"""
Check which circuits have both AAG and AIG formats in the four benchmark folders.
"""

import os
import glob
from pathlib import Path
from collections import defaultdict

def check_circuit_formats():
    """Check which circuits have both AAG and AIG formats."""
    print("🔍 Checking circuit formats in four benchmark folders...")
    print("=" * 60)
    
    # Benchmark suites to check
    benchmark_suites = ["MCNC", "IWLS", "Synthetic", "EPFL"]
    
    total_stats = {
        'aag_only': 0,
        'aig_only': 0,
        'both_formats': 0,
        'total_circuits': 0
    }
    
    for suite in benchmark_suites:
        print(f"\n🏆 Checking {suite} suite...")
        
        suite_dir = f"testcase/{suite}"
        if not os.path.exists(suite_dir):
            print(f"   ❌ Suite directory not found: {suite_dir}")
            continue
        
        # Find all AAG and AIG files
        aag_files = set()
        aig_files = set()
        
        # Find AAG files
        for aag_file in glob.glob(f"{suite_dir}/**/*.aag", recursive=True):
            circuit_name = Path(aag_file).stem  # Remove extension
            aag_files.add(circuit_name)
        
        # Find AIG files
        for aig_file in glob.glob(f"{suite_dir}/**/*.aig", recursive=True):
            circuit_name = Path(aig_file).stem  # Remove extension
            aig_files.add(circuit_name)
        
        # Analyze format availability
        aag_only = aag_files - aig_files
        aig_only = aig_files - aag_files
        both_formats = aag_files & aig_files
        
        print(f"   📊 {suite} Results:")
        print(f"      - AAG files found: {len(aag_files)}")
        print(f"      - AIG files found: {len(aig_files)}")
        print(f"      - AAG only: {len(aag_only)}")
        print(f"      - AIG only: {len(aig_only)}")
        print(f"      - Both formats: {len(both_formats)}")
        
        # Show examples of each category
        if aag_only:
            print(f"      - AAG only examples: {list(aag_only)[:5]}")
        if aig_only:
            print(f"      - AIG only examples: {list(aig_only)[:5]}")
        if both_formats:
            print(f"      - Both formats examples: {list(both_formats)[:5]}")
        
        # Update total stats
        total_stats['aag_only'] += len(aag_only)
        total_stats['aig_only'] += len(aig_only)
        total_stats['both_formats'] += len(both_formats)
        total_stats['total_circuits'] += len(aag_files | aig_files)
    
    # Overall summary
    print(f"\n🎯 OVERALL SUMMARY")
    print("=" * 60)
    print(f"Total unique circuits: {total_stats['total_circuits']}")
    print(f"AAG only: {total_stats['aag_only']}")
    print(f"AIG only: {total_stats['aig_only']}")
    print(f"Both formats: {total_stats['both_formats']}")
    
    # Calculate percentages
    if total_stats['total_circuits'] > 0:
        aag_only_pct = (total_stats['aag_only'] / total_stats['total_circuits']) * 100
        aig_only_pct = (total_stats['aig_only'] / total_stats['total_circuits']) * 100
        both_pct = (total_stats['both_formats'] / total_stats['total_circuits']) * 100
        
        print(f"\n📊 PERCENTAGES:")
        print(f"AAG only: {aag_only_pct:.1f}%")
        print(f"AIG only: {aig_only_pct:.1f}%")
        print(f"Both formats: {both_pct:.1f}%")
    
    return total_stats

def check_specific_circuits():
    """Check specific circuits to see their format availability."""
    print(f"\n🔍 DETAILED CIRCUIT FORMAT CHECK")
    print("=" * 60)
    
    # Sample circuits to check in detail
    sample_circuits = [
        "testcase/MCNC/alu1",
        "testcase/MCNC/C3540", 
        "testcase/IWLS/ex01",
        "testcase/IWLS/ex100",
        "testcase/Synthetic/mult_4x4_array_ml1",
        "testcase/Synthetic/adder_16b_csel_m5",
        "testcase/EPFL/ctrl",
        "testcase/EPFL/adder"
    ]
    
    for circuit_base in sample_circuits:
        aag_path = f"{circuit_base}.aag"
        aig_path = f"{circuit_base}.aig"
        
        aag_exists = os.path.exists(aag_path)
        aig_exists = os.path.exists(aig_path)
        
        circuit_name = Path(circuit_base).name
        print(f"{circuit_name}:")
        print(f"  - AAG: {'✅' if aag_exists else '❌'} ({aag_path})")
        print(f"  - AIG: {'✅' if aig_exists else '❌'} ({aig_path})")
        
        if aag_exists and aig_exists:
            print(f"  - Status: ✅ Both formats available")
        elif aag_exists:
            print(f"  - Status: 📄 AAG only")
        elif aig_exists:
            print(f"  - Status: 📄 AIG only")
        else:
            print(f"  - Status: ❌ Neither format found")

def check_format_distribution():
    """Check the distribution of formats across all circuits."""
    print(f"\n📊 FORMAT DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    benchmark_suites = ["MCNC", "IWLS", "Synthetic", "EPFL"]
    
    for suite in benchmark_suites:
        print(f"\n🏆 {suite} Format Distribution:")
        
        suite_dir = f"testcase/{suite}"
        if not os.path.exists(suite_dir):
            continue
        
        # Count files by format
        aag_count = len(glob.glob(f"{suite_dir}/**/*.aag", recursive=True))
        aig_count = len(glob.glob(f"{suite_dir}/**/*.aig", recursive=True))
        
        print(f"  - AAG files: {aag_count}")
        print(f"  - AIG files: {aig_count}")
        print(f"  - Total files: {aag_count + aig_count}")
        
        if aag_count + aig_count > 0:
            aag_pct = (aag_count / (aag_count + aig_count)) * 100
            aig_pct = (aig_count / (aag_count + aig_count)) * 100
            print(f"  - AAG percentage: {aag_pct:.1f}%")
            print(f"  - AIG percentage: {aig_pct:.1f}%")

if __name__ == "__main__":
    # Run comprehensive format check
    stats = check_circuit_formats()
    
    # Check specific circuits
    check_specific_circuits()
    
    # Check format distribution
    check_format_distribution()
    
    print(f"\n🎯 FINAL SUMMARY")
    print("=" * 60)
    print(f"Total circuits with any format: {stats['total_circuits']}")
    print(f"Circuits with both AAG and AIG: {stats['both_formats']}")
    print(f"Circuits with AAG only: {stats['aag_only']}")
    print(f"Circuits with AIG only: {stats['aig_only']}")
    
    if stats['total_circuits'] > 0:
        both_pct = (stats['both_formats'] / stats['total_circuits']) * 100
        print(f"\nPercentage of circuits with both formats: {both_pct:.1f}%")
        
        if both_pct == 100:
            print("🎉 ALL circuits have both AAG and AIG formats!")
        elif both_pct > 50:
            print("✅ Most circuits have both formats")
        else:
            print("⚠️  Many circuits are missing one format") 