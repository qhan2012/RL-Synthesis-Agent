#!/usr/bin/env python3
"""
Analyze circuit complexities and prepare for partitioning into training/validation/test sets.
"""

import os
import glob
from pathlib import Path
import json
from collections import defaultdict

def analyze_circuit_complexity():
    """Analyze all circuits and their complexities."""
    print("🔍 Analyzing circuit complexities...")
    print("=" * 60)
    
    # Benchmark suites to analyze
    benchmark_suites = ["MCNC", "IWLS", "Synthetic", "EPFL"]
    
    all_circuits = {}
    complexity_stats = defaultdict(list)
    
    for suite in benchmark_suites:
        print(f"\n🏆 Analyzing {suite} suite...")
        
        suite_dir = f"testcase/{suite}"
        if not os.path.exists(suite_dir):
            print(f"   ❌ Suite directory not found: {suite_dir}")
            continue
        
        # Find all AAG files (we'll use AAG for complexity analysis)
        aag_files = glob.glob(f"{suite_dir}/**/*.aag", recursive=True)
        
        print(f"   Found {len(aag_files)} AAG files")
        
        for aag_file in aag_files:
            circuit_name = Path(aag_file).stem
            circuit_path = str(aag_file)
            
            try:
                # Parse AAG file to get complexity metrics
                complexity = parse_aag_complexity(aag_file)
                
                circuit_info = {
                    'name': circuit_name,
                    'suite': suite,
                    'path': circuit_path,
                    'aag_path': circuit_path,
                    'aig_path': circuit_path.replace('.aag', '.aig'),
                    'complexity': complexity,
                    'gate_count': complexity['total_gates'],
                    'input_count': complexity['inputs'],
                    'output_count': complexity['outputs'],
                    'and_gates': complexity['and_gates']
                }
                
                all_circuits[f"{suite}/{circuit_name}"] = circuit_info
                complexity_stats[suite].append(circuit_info)
                
            except Exception as e:
                print(f"   ❌ Error analyzing {circuit_name}: {e}")
    
    # Print complexity statistics
    print(f"\n📊 COMPLEXITY STATISTICS")
    print("=" * 60)
    
    for suite in benchmark_suites:
        if suite in complexity_stats:
            circuits = complexity_stats[suite]
            gate_counts = [c['gate_count'] for c in circuits]
            
            print(f"\n🏆 {suite}:")
            print(f"  - Total circuits: {len(circuits)}")
            print(f"  - Min gates: {min(gate_counts)}")
            print(f"  - Max gates: {max(gate_counts)}")
            print(f"  - Avg gates: {sum(gate_counts) / len(gate_counts):.1f}")
            print(f"  - Median gates: {sorted(gate_counts)[len(gate_counts)//2]}")
    
    return all_circuits, complexity_stats

def parse_aag_complexity(file_path):
    """Parse AAG file to extract complexity metrics."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the header line (starts with 'aag')
    header_line = None
    for line in lines:
        line = line.strip()
        if line.startswith('aag'):
            header_line = line
            break
    
    if not header_line:
        raise ValueError("No AAG header found")
    
    # Parse header: aag max_var inputs outputs and_gates num_latches
    parts = header_line.split()
    if len(parts) < 5:
        raise ValueError("Invalid AAG header format")
    
    max_var = int(parts[1])
    inputs = int(parts[2])
    outputs = int(parts[3])
    and_gates = int(parts[4])
    num_latches = int(parts[5]) if len(parts) > 5 else 0
    
    # Calculate total gates (inputs + outputs + AND gates)
    total_gates = inputs + outputs + and_gates
    
    return {
        'max_var': max_var,
        'inputs': inputs,
        'outputs': outputs,
        'and_gates': and_gates,
        'num_latches': num_latches,
        'total_gates': total_gates
    }

def create_partitioning_strategy():
    """Create the partitioning strategy based on specifications."""
    print(f"\n🎯 CREATING PARTITIONING STRATEGY")
    print("=" * 60)
    
    # Load circuit data
    all_circuits, complexity_stats = analyze_circuit_complexity()
    
    # Define partitioning strategy
    strategy = {
        'training': {
            'EPFL': 32,
            'Synthetic': 160,
            'MCNC': 200,
            'IWLS': 0  # Not included in training per specification
        },
        'validation': {
            'EPFL': 4,
            'Synthetic': 20,
            'MCNC': 25,
            'IWLS': 0  # Not included in validation per specification
        },
        'test': {
            'EPFL': 4,
            'Synthetic': 20,
            'MCNC': 25,
            'IWLS': 0  # Not included in test per specification
        }
    }
    
    # Complexity targets
    complexity_targets = {
        'training': {'min': 50, 'max': 2000},
        'validation': {'min': 100, 'max': 5000},
        'test': {'min': 200, 'max': 10000}
    }
    
    # Partition circuits
    partitions = partition_circuits(all_circuits, strategy, complexity_targets)
    
    # Save partitioning information
    save_partitioning_info(partitions, all_circuits)
    
    return partitions

def partition_circuits(all_circuits, strategy, complexity_targets):
    """Partition circuits according to the strategy."""
    print(f"\n🔧 PARTITIONING CIRCUITS")
    print("=" * 60)
    
    partitions = {
        'training': [],
        'validation': [],
        'test': []
    }
    
    # Group circuits by suite
    suite_circuits = defaultdict(list)
    for circuit_id, circuit_info in all_circuits.items():
        suite = circuit_info['suite']
        suite_circuits[suite].append(circuit_info)
    
    # Sort circuits by gate count within each suite
    for suite in suite_circuits:
        suite_circuits[suite].sort(key=lambda x: x['gate_count'])
    
    # Partition each suite
    for suite in ['EPFL', 'Synthetic', 'MCNC']:
        if suite not in suite_circuits:
            continue
            
        circuits = suite_circuits[suite]
        print(f"\n🏆 Partitioning {suite} ({len(circuits)} circuits):")
        
        # Training set (80%)
        train_count = strategy['training'][suite]
        train_circuits = circuits[:train_count]
        partitions['training'].extend(train_circuits)
        print(f"  - Training: {len(train_circuits)} circuits")
        
        # Validation set (10%)
        val_count = strategy['validation'][suite]
        val_start = train_count
        val_end = val_start + val_count
        val_circuits = circuits[val_start:val_end]
        partitions['validation'].extend(val_circuits)
        print(f"  - Validation: {len(val_circuits)} circuits")
        
        # Test set (10%)
        test_count = strategy['test'][suite]
        test_start = val_end
        test_end = test_start + test_count
        test_circuits = circuits[test_start:test_end]
        partitions['test'].extend(test_circuits)
        print(f"  - Test: {len(test_circuits)} circuits")
        
        # Remaining circuits (if any)
        remaining = len(circuits) - (train_count + val_count + test_count)
        if remaining > 0:
            print(f"  - Remaining: {remaining} circuits (not used)")
    
    # Print summary
    print(f"\n📊 PARTITIONING SUMMARY")
    print("=" * 60)
    for split_name, circuits in partitions.items():
        gate_counts = [c['gate_count'] for c in circuits]
        if gate_counts:
            print(f"{split_name.capitalize()}: {len(circuits)} circuits")
            print(f"  - Gate range: {min(gate_counts)} - {max(gate_counts)}")
            print(f"  - Avg gates: {sum(gate_counts) / len(gate_counts):.1f}")
        else:
            print(f"{split_name.capitalize()}: 0 circuits")
    
    return partitions

def save_partitioning_info(partitions, all_circuits):
    """Save partitioning information to files."""
    print(f"\n💾 SAVING PARTITIONING INFORMATION")
    print("=" * 60)
    
    # Create detailed partitioning file
    partitioning_data = {
        'metadata': {
            'total_circuits': len(all_circuits),
            'training_circuits': len(partitions['training']),
            'validation_circuits': len(partitions['validation']),
            'test_circuits': len(partitions['test']),
            'strategy': {
                'training_target': '50-2,000 gates (manageable complexity)',
                'validation_target': '100-5,000 gates (moderate challenge)',
                'test_target': '200-10,000 gates (real-world complexity)'
            }
        },
        'partitions': {}
    }
    
    for split_name, circuits in partitions.items():
        partitioning_data['partitions'][split_name] = []
        
        for circuit in circuits:
            partitioning_data['partitions'][split_name].append({
                'name': circuit['name'],
                'suite': circuit['suite'],
                'path': circuit['path'],
                'aag_path': circuit['aag_path'],
                'aig_path': circuit['aig_path'],
                'complexity': circuit['complexity']
            })
    
    # Save JSON file
    with open('circuit_partitioning.json', 'w') as f:
        json.dump(partitioning_data, f, indent=2)
    
    # Save simple text files for easy access
    for split_name, circuits in partitions.items():
        filename = f'circuits_{split_name}.txt'
        with open(filename, 'w') as f:
            f.write(f"# {split_name.upper()} CIRCUITS\n")
            f.write(f"# Total: {len(circuits)}\n")
            f.write(f"# Format: suite/circuit_name\n\n")
            
            for circuit in circuits:
                f.write(f"{circuit['suite']}/{circuit['name']}\n")
    
    # Save summary file
    with open('partitioning_summary.txt', 'w') as f:
        f.write("CIRCUIT PARTITIONING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("STRATEGY:\n")
        f.write("- Training Set (80%): 50-2,000 gates (manageable complexity)\n")
        f.write("- Validation Set (10%): 100-5,000 gates (moderate challenge)\n")
        f.write("- Test Set (10%): 200-10,000 gates (real-world complexity)\n\n")
        
        f.write("DISTRIBUTION:\n")
        for split_name, circuits in partitions.items():
            gate_counts = [c['gate_count'] for c in circuits]
            if gate_counts:
                f.write(f"- {split_name.upper()}: {len(circuits)} circuits\n")
                f.write(f"  Gate range: {min(gate_counts)} - {max(gate_counts)}\n")
                f.write(f"  Avg gates: {sum(gate_counts) / len(gate_counts):.1f}\n\n")
        
        f.write("FILES CREATED:\n")
        f.write("- circuit_partitioning.json: Detailed partitioning data\n")
        f.write("- circuits_training.txt: Training circuit list\n")
        f.write("- circuits_validation.txt: Validation circuit list\n")
        f.write("- circuits_test.txt: Test circuit list\n")
        f.write("- partitioning_summary.txt: This summary file\n")
    
    print("✅ Partitioning files created:")
    print("  - circuit_partitioning.json")
    print("  - circuits_training.txt")
    print("  - circuits_validation.txt")
    print("  - circuits_test.txt")
    print("  - partitioning_summary.txt")

if __name__ == "__main__":
    # Create partitioning strategy
    partitions = create_partitioning_strategy()
    
    print(f"\n🎉 PARTITIONING COMPLETE!")
    print("=" * 60)
    print("All circuit partitioning files have been created.")
    print("Use these files for consistent training/validation/test splits.") 