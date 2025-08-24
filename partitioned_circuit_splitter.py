#!/usr/bin/env python3
"""
Partitioned Circuit Data Splitter

This module provides a data splitter that uses the pre-partitioned circuit files
for training, validation, and testing.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any


class PartitionedCircuitDataSplitter:
    """Data splitter that uses pre-partitioned circuit files."""
    
    def __init__(self, training_file: str = 'circuits_training_realistic.txt',
                 validation_file: str = 'circuits_validation_realistic.txt',
                 test_file: str = 'circuits_test_realistic.txt',
                 base_path: str = 'testcase'):
        """
        Initialize the partitioned circuit data splitter.
        
        Args:
            training_file: Path to training circuit list file
            validation_file: Path to validation circuit list file
            test_file: Path to test circuit list file
            base_path: Base path for circuit files
        """
        self.base_path = base_path
        self.training_file = training_file
        self.validation_file = validation_file
        self.test_file = test_file
        
        # Load circuit lists
        self.train_circuits = self._load_circuit_list(training_file)
        self.val_circuits = self._load_circuit_list(validation_file)
        self.test_circuits = self._load_circuit_list(test_file)
        
        # Create metadata for each circuit
        self.train_metadata = self._create_metadata(self.train_circuits)
        self.val_metadata = self._create_metadata(self.val_circuits)
        self.test_metadata = self._create_metadata(self.test_circuits)
        
        print(f"Loaded partitioned circuits:")
        print(f"  Training: {len(self.train_circuits)} circuits")
        print(f"  Validation: {len(self.val_circuits)} circuits")
        print(f"  Test: {len(self.test_circuits)} circuits")
    
    def _load_circuit_list(self, filename: str) -> List[str]:
        """Load circuit list from file."""
        circuits = []
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        circuits.append(line)
        else:
            print(f"Warning: Circuit file {filename} not found")
        return circuits
    
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
    
    def get_train_circuits(self):
        """Get training circuits with metadata."""
        return self.train_metadata
    
    def get_val_circuits(self):
        """Get validation circuits with metadata."""
        return self.val_metadata
    
    def get_eval_circuits(self):
        """Get test circuits with metadata."""
        return self.test_metadata
    
    def get_split_info(self):
        """Get information about the data splits."""
        total_circuits = len(self.train_circuits) + len(self.val_circuits) + len(self.test_circuits)
        
        return {
            'total_circuits': total_circuits,
            'train_circuits': len(self.train_circuits),
            'val_circuits': len(self.val_circuits),
            'eval_circuits': len(self.test_circuits),
            'train_ratio': len(self.train_circuits) / total_circuits if total_circuits > 0 else 0,
            'val_ratio': len(self.val_circuits) / total_circuits if total_circuits > 0 else 0,
            'eval_ratio': len(self.test_circuits) / total_circuits if total_circuits > 0 else 0,
            'split_type': 'partitioned'
        }
    
    def sample_training_circuits(self, num_circuits: int) -> List[Tuple[str, Dict[str, Any]]]:
        """Sample circuits from training set."""
        if num_circuits >= len(self.train_metadata):
            # If requesting more circuits than available, return all with repetition
            sampled = []
            for _ in range(num_circuits):
                sampled.append(random.choice(self.train_metadata))
            return sampled
        else:
            # Sample without replacement
            return random.sample(self.train_metadata, num_circuits)
    
    def get_circuit_stats(self):
        """Get statistics about the circuits."""
        stats = {
            'training': {
                'count': len(self.train_circuits),
                'suites': self._get_suite_distribution(self.train_circuits)
            },
            'validation': {
                'count': len(self.val_circuits),
                'suites': self._get_suite_distribution(self.val_circuits)
            },
            'test': {
                'count': len(self.test_circuits),
                'suites': self._get_suite_distribution(self.test_circuits)
            }
        }
        return stats
    
    def _get_suite_distribution(self, circuits: List[str]) -> Dict[str, int]:
        """Get distribution of circuits by suite."""
        distribution = {}
        for circuit in circuits:
            if '/' in circuit:
                suite = circuit.split('/', 1)[0]
            else:
                suite = 'unknown'
            distribution[suite] = distribution.get(suite, 0) + 1
        return distribution


def create_partitioned_splitter():
    """Create a partitioned circuit data splitter."""
    return PartitionedCircuitDataSplitter() 