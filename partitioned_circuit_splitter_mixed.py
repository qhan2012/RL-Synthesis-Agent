#!/usr/bin/env python3
"""
Partitioned Circuit Data Splitter for Mixed Complexity Training.

This module provides a data splitter that loads circuits from pre-partitioned
mixed-complexity files for training, validation, and testing.
"""

import os
import random
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

class PartitionedCircuitDataSplitterMixed:
    """
    Data splitter for mixed-complexity partitioned circuits.
    
    Loads circuits from pre-partitioned files that include mixed complexity
    levels in training for better learning and generalization.
    """
    
    def __init__(self, base_path: str = "testcase"):
        """
        Initialize the mixed-complexity circuit data splitter.
        
        Args:
            base_path: Base path to the testcase directory
        """
        self.base_path = base_path
        self.training_circuits = []
        self.validation_circuits = []
        self.test_circuits = []
        
        # Load circuits from mixed-complexity partitioning files
        self._load_mixed_complexity_circuits()
    
    def _load_mixed_complexity_circuits(self):
        """Load circuits from mixed-complexity partitioning files."""
        print("📁 Loading mixed-complexity partitioned circuits...")
        
        # Load training circuits
        training_file = "circuits_training_mixed_complexity.txt"
        if os.path.exists(training_file):
            with open(training_file, 'r') as f:
                lines = f.readlines()
                self.training_circuits = [line.strip() for line in lines 
                                        if not line.startswith('#') and line.strip()]
            print(f"  ✅ Training: {len(self.training_circuits)} circuits")
        else:
            print(f"  ❌ Training file not found: {training_file}")
        
        # Load validation circuits
        validation_file = "circuits_validation_mixed_complexity.txt"
        if os.path.exists(validation_file):
            with open(validation_file, 'r') as f:
                lines = f.readlines()
                self.validation_circuits = [line.strip() for line in lines 
                                          if not line.startswith('#') and line.strip()]
            print(f"  ✅ Validation: {len(self.validation_circuits)} circuits")
        else:
            print(f"  ❌ Validation file not found: {validation_file}")
        
        # Load test circuits
        test_file = "circuits_test_mixed_complexity.txt"
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                lines = f.readlines()
                self.test_circuits = [line.strip() for line in lines 
                                    if not line.startswith('#') and line.strip()]
            print(f"  ✅ Test: {len(self.test_circuits)} circuits")
        else:
            print(f"  ❌ Test file not found: {test_file}")
    
    def _create_metadata(self, circuits: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
        """Create metadata for circuits."""
        metadata = []
        for circuit in circuits:
            # Extract suite and name
            if '/' in circuit:
                suite, name = circuit.split('/', 1)
            else:
                suite = 'unknown'
                name = circuit
            
            # Create circuit path - circuits are in subdirectories
            circuit_path = os.path.join(self.base_path, f"{circuit}/{circuit.split('/')[-1]}.aag")
            
            # Create metadata
            circuit_metadata = {
                'name': name,
                'suite': suite,
                'path': circuit_path,
                'aag_path': circuit_path,
                'aig_path': circuit_path.replace('.aag', '.aig'),
                'circuit_id': circuit
            }
            
            metadata.append((circuit_path, circuit_metadata))
        return metadata
    
    def get_training_circuits(self, num_circuits: Optional[int] = None) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get training circuits with mixed complexity.
        
        Args:
            num_circuits: Number of circuits to return (None for all)
            
        Returns:
            List of (circuit_path, metadata) tuples
        """
        if num_circuits is None:
            circuits = self.training_circuits
        else:
            circuits = random.sample(self.training_circuits, min(num_circuits, len(self.training_circuits)))
        
        return self._create_metadata(circuits)
    
    def get_validation_circuits(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get validation circuits (medium + large complexity).
        
        Returns:
            List of (circuit_path, metadata) tuples
        """
        return self._create_metadata(self.validation_circuits)
    
    def get_test_circuits(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get test circuits (large + very large complexity).
        
        Returns:
            List of (circuit_path, metadata) tuples
        """
        return self._create_metadata(self.test_circuits)
    
    def get_split_info(self) -> Dict[str, Any]:
        """Get information about the data splits."""
        total_circuits = len(self.training_circuits) + len(self.validation_circuits) + len(self.test_circuits)
        
        return {
            'total_circuits': total_circuits,
            'train_circuits': len(self.training_circuits),
            'val_circuits': len(self.validation_circuits),
            'eval_circuits': len(self.test_circuits),
            'train_ratio': len(self.training_circuits) / total_circuits if total_circuits > 0 else 0,
            'val_ratio': len(self.validation_circuits) / total_circuits if total_circuits > 0 else 0,
            'eval_ratio': len(self.test_circuits) / total_circuits if total_circuits > 0 else 0,
            'strategy': 'Mixed Complexity Partitioning'
        }
    
    def get_circuit_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed statistics about circuit distribution."""
        def analyze_suite_distribution(circuits):
            suite_counts = defaultdict(int)
            for circuit in circuits:
                if '/' in circuit:
                    suite = circuit.split('/', 1)[0]
                    suite_counts[suite] += 1
            return dict(suite_counts)
        
        return {
            'training': {
                'count': len(self.training_circuits),
                'suites': analyze_suite_distribution(self.training_circuits),
                'complexity': 'Mixed (Small + Medium + Large + Very Large)'
            },
            'validation': {
                'count': len(self.validation_circuits),
                'suites': analyze_suite_distribution(self.validation_circuits),
                'complexity': 'Medium + Large + Very Large'
            },
            'test': {
                'count': len(self.test_circuits),
                'suites': analyze_suite_distribution(self.test_circuits),
                'complexity': 'Large + Very Large'
            }
        }
    
    def sample_training_circuits(self, num_samples: int) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Sample circuits from training set.
        
        Args:
            num_samples: Number of circuits to sample
            
        Returns:
            List of (circuit_path, metadata) tuples
        """
        if num_samples >= len(self.training_circuits):
            return self.get_training_circuits()
        
        sampled_circuits = random.sample(self.training_circuits, num_samples)
        return self._create_metadata(sampled_circuits)
    
    def get_all_circuits(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get all circuits from all splits."""
        all_circuits = self.training_circuits + self.validation_circuits + self.test_circuits
        return self._create_metadata(all_circuits)

# Example usage
if __name__ == "__main__":
    # Create splitter
    splitter = PartitionedCircuitDataSplitterMixed()
    
    # Get split information
    split_info = splitter.get_split_info()
    print(f"\n📊 Split Information:")
    print(f"  Total circuits: {split_info['total_circuits']}")
    print(f"  Training: {split_info['train_circuits']} ({split_info['train_ratio']:.1%})")
    print(f"  Validation: {split_info['val_circuits']} ({split_info['val_ratio']:.1%})")
    print(f"  Test: {split_info['eval_circuits']} ({split_info['eval_ratio']:.1%})")
    
    # Get circuit statistics
    stats = splitter.get_circuit_stats()
    print(f"\n📈 Circuit Statistics:")
    for split_name, split_stats in stats.items():
        print(f"  {split_name.capitalize()}: {split_stats['count']} circuits")
        print(f"    Complexity: {split_stats['complexity']}")
        for suite, count in split_stats['suites'].items():
            print(f"    {suite}: {count} circuits") 