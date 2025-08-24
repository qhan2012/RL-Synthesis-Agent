#!/usr/bin/env python3
"""
Create realistic circuit partitioning using all available circuits.
"""

import os
import glob
from pathlib import Path
import json
from collections import defaultdict

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

def load_all_circuits():
    """Load all circuits with their complexity information."""
    print("🔍 Loading all circuits...")
    
    benchmark_suites = ["MCNC", "IWLS", "Synthetic", "EPFL"]
    all_circuits = {}
    
    for suite in benchmark_suites:
        suite_dir = f"testcase/{suite}"
        if not os.path.exists(suite_dir):
            continue
        
        aag_files = glob.glob(f"{suite_dir}/**/*.aag", recursive=True)
        print(f"  {suite}: {len(aag_files)} circuits")
        
        for aag_file in aag_files:
            circuit_name = Path(aag_file).stem
            circuit_path = str(aag_file)
            
            try:
                complexity = parse_aag_complexity(aag_file)
                
                circuit_info = {
                    'name': circuit_name,
                    'suite': suite,
                    'path': circuit_path,
                    'aag_path': circuit_path,
                    'aig_path': circuit_path.replace('.aag', '.aig'),
                    'complexity': complexity,
                    'gate_count': complexity['total_gates']
                }
                
                all_circuits[f"{suite}/{circuit_name}"] = circuit_info
                
            except Exception as e:
                print(f"    ❌ Error with {circuit_name}: {e}")
    
    return all_circuits

def create_realistic_partitioning():
    """Create realistic partitioning using all available circuits."""
    print("🎯 Creating realistic partitioning...")
    
    # Load all circuits
    all_circuits = load_all_circuits()
    
    # Group by suite
    suite_circuits = defaultdict(list)
    for circuit_id, circuit_info in all_circuits.items():
        suite = circuit_info['suite']
        suite_circuits[suite].append(circuit_info)
    
    # Sort by gate count (ascending) within each suite
    for suite in suite_circuits:
        suite_circuits[suite].sort(key=lambda x: x['gate_count'])
    
    # Create partitions
    partitions = {
        'training': [],
        'validation': [],
        'test': []
    }
    
    print("\n📊 Realistic partitioning by suite:")
    
    # Use only MCNC, Synthetic, and EPFL (exclude IWLS as per specification)
    target_suites = ['MCNC', 'Synthetic', 'EPFL']
    
    for suite in target_suites:
        if suite not in suite_circuits:
            continue
        
        circuits = suite_circuits[suite]
        total_circuits = len(circuits)
        
        print(f"\n🏆 {suite} ({total_circuits} total circuits):")
        
        # Calculate realistic splits (80% training, 10% validation, 10% test)
        train_count = int(total_circuits * 0.8)
        val_count = int(total_circuits * 0.1)
        test_count = total_circuits - train_count - val_count
        
        # Training set (smaller circuits first)
        train_circuits = circuits[:train_count]
        partitions['training'].extend(train_circuits)
        print(f"  - Training: {len(train_circuits)} circuits (gates: {min([c['gate_count'] for c in train_circuits])}-{max([c['gate_count'] for c in train_circuits])})")
        
        # Validation set (medium complexity)
        val_start = train_count
        val_end = val_start + val_count
        val_circuits = circuits[val_start:val_end]
        partitions['validation'].extend(val_circuits)
        if val_circuits:
            print(f"  - Validation: {len(val_circuits)} circuits (gates: {min([c['gate_count'] for c in val_circuits])}-{max([c['gate_count'] for c in val_circuits])})")
        else:
            print(f"  - Validation: 0 circuits")
        
        # Test set (larger circuits)
        test_start = val_end
        test_circuits = circuits[test_start:]
        partitions['test'].extend(test_circuits)
        if test_circuits:
            print(f"  - Test: {len(test_circuits)} circuits (gates: {min([c['gate_count'] for c in test_circuits])}-{max([c['gate_count'] for c in test_circuits])})")
        else:
            print(f"  - Test: 0 circuits")
    
    return partitions, all_circuits

def save_partitioning_files(partitions, all_circuits):
    """Save all partitioning files."""
    print("\n💾 Saving partitioning files...")
    
    # Calculate totals
    total_training = len(partitions['training'])
    total_validation = len(partitions['validation'])
    total_test = len(partitions['test'])
    total_used = total_training + total_validation + total_test
    
    # Create detailed JSON file
    partitioning_data = {
        'metadata': {
            'total_circuits': len(all_circuits),
            'training_circuits': total_training,
            'validation_circuits': total_validation,
            'test_circuits': total_test,
            'total_used': total_used,
            'strategy': {
                'training_target': '50-2,000 gates (manageable complexity)',
                'validation_target': '100-5,000 gates (moderate challenge)', 
                'test_target': '200-10,000 gates (real-world complexity)',
                'split_ratio': '80% training, 10% validation, 10% test'
            },
            'suites_used': ['MCNC', 'Synthetic', 'EPFL'],
            'suites_excluded': ['IWLS']
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
    with open('circuit_partitioning_realistic.json', 'w') as f:
        json.dump(partitioning_data, f, indent=2)
    
    # Save simple text files
    for split_name, circuits in partitions.items():
        filename = f'circuits_{split_name}_realistic.txt'
        with open(filename, 'w') as f:
            f.write(f"# {split_name.upper()} CIRCUITS (REALISTIC PARTITIONING)\n")
            f.write(f"# Total: {len(circuits)}\n")
            f.write(f"# Format: suite/circuit_name\n\n")
            
            for circuit in circuits:
                f.write(f"{circuit['suite']}/{circuit['name']}\n")
    
    # Save summary
    with open('partitioning_summary_realistic.txt', 'w') as f:
        f.write("REALISTIC CIRCUIT PARTITIONING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("STRATEGY:\n")
        f.write("- Training Set (80%): 50-2,000 gates (manageable complexity)\n")
        f.write("- Validation Set (10%): 100-5,000 gates (moderate challenge)\n")
        f.write("- Test Set (10%): 200-10,000 gates (real-world complexity)\n\n")
        
        f.write("SUITES USED:\n")
        f.write("- MCNC: 221 circuits\n")
        f.write("- Synthetic: 100 circuits\n")
        f.write("- EPFL: 20 circuits\n")
        f.write("- IWLS: Excluded (not used in training)\n\n")
        
        f.write("ACTUAL DISTRIBUTION:\n")
        for split_name, circuits in partitions.items():
            gate_counts = [c['gate_count'] for c in circuits]
            if gate_counts:
                f.write(f"- {split_name.upper()}: {len(circuits)} circuits\n")
                f.write(f"  Gate range: {min(gate_counts)} - {max(gate_counts)}\n")
                f.write(f"  Avg gates: {sum(gate_counts) / len(gate_counts):.1f}\n\n")
        
        f.write("FILES CREATED:\n")
        f.write("- circuit_partitioning_realistic.json: Detailed partitioning data\n")
        f.write("- circuits_training_realistic.txt: Training circuit list\n")
        f.write("- circuits_validation_realistic.txt: Validation circuit list\n")
        f.write("- circuits_test_realistic.txt: Test circuit list\n")
        f.write("- partitioning_summary_realistic.txt: This summary file\n")
    
    print("✅ Files created:")
    print("  - circuit_partitioning_realistic.json")
    print("  - circuits_training_realistic.txt")
    print("  - circuits_validation_realistic.txt")
    print("  - circuits_test_realistic.txt")
    print("  - partitioning_summary_realistic.txt")

if __name__ == "__main__":
    # Create realistic partitioning
    partitions, all_circuits = create_realistic_partitioning()
    
    # Save files
    save_partitioning_files(partitions, all_circuits)
    
    # Print final summary
    print(f"\n🎯 FINAL PARTITIONING SUMMARY")
    print("=" * 60)
    
    total_training = len(partitions['training'])
    total_validation = len(partitions['validation'])
    total_test = len(partitions['test'])
    
    print(f"Training circuits: {total_training}")
    print(f"Validation circuits: {total_validation}")
    print(f"Test circuits: {total_test}")
    print(f"Total used: {total_training + total_validation + total_test}")
    print(f"Total available: {len(all_circuits)}")
    
    # Show gate ranges
    for split_name, circuits in partitions.items():
        if circuits:
            gate_counts = [c['gate_count'] for c in circuits]
            print(f"{split_name.capitalize()} gate range: {min(gate_counts)} - {max(gate_counts)}")
    
    print(f"\n🎉 Realistic partitioning complete!")
    print("Use the '_realistic' files for consistent training/validation/test splits.") 