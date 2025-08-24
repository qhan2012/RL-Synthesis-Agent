#!/usr/bin/env python3
"""
Example script for RL Synthesis Agent

This script demonstrates how to use the RL synthesis agent
for training and evaluation.
"""

import sys
import yaml
import logging
from pathlib import Path
import torch

# Add project root to path
sys.path.append('.')

from models.ppo_agent import PPOSynthesisAgent
from env.synthesis_env import SynthesisEnvironment
from data.dataset import CircuitDataset
from utils.logger import setup_logger
from utils.metrics import MetricsTracker


def load_config():
    """Load configuration from YAML file."""
    config_path = 'config/ppo_config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print("Using default configuration...")
        return get_default_config()


def get_default_config():
    """Get default configuration."""
    return {
        'gnn_encoder': {
            'type': 'GIN',
            'hidden_dim': 256,       # INCREASED: from 128 to 256
            'num_layers': 5,         # INCREASED: from 3 to 5 layers
            'pooling': 'mean'
        },
        'actor_head': {
            'input_dim': 257,  # 256 (GNN) + 1 (timestep) - UPDATED
            'layers': [256, 128, 64],  # ENHANCED: deeper network
            'output': 'softmax'
        },
        'critic_head': {
            'input_dim': 257,  # 256 (GNN) + 1 (timestep) - UPDATED
            'layers': [512, 256, 128, 64],  # ENHANCED: much deeper network
            'output': 'scalar'
        },
        'learning_rate': 3e-4,
        'clip_range': 0.1,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'ppo_epochs': 4,
        'env': {
            'max_steps': 10,
            'action_space': ['b', 'rw', 'rf', 'rwz', 'rfz'],
            'reward_shaping': True,
            'reward_normalization': True,
            'final_bonus': True
        },
        'dataset': {
            'data_root': '../testcase',
            'sources': ['EPFL', 'MCNC', 'ISCAS85']
        }
    }


def example_agent_creation():
    """Example: Create and test the agent."""
    print("="*50)
    print("Example: Agent Creation")
    print("="*50)
    
    # Load configuration
    config = load_config()
    
    # Create agent
    agent = PPOSynthesisAgent(config)
    
    print(f"Agent created successfully!")
    print(f"Device: {agent.device}")
    print(f"Total parameters: {sum(p.numel() for p in agent.parameters()):,}")
    
    # Test with dummy data
    import torch_geometric.data as pyg_data
    
    # Create a simple test graph
    x = torch.randn(10, 6)  # 10 nodes, 6 features each
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9],
                              [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7, 9, 8]], dtype=torch.long)
    timestep = torch.tensor([0.5])
    
    data = pyg_data.Data(x=x, edge_index=edge_index, timestep=timestep)
    
    # Test forward pass
    agent.eval()
    with torch.no_grad():
        action_logits, value = agent.forward([data])
        action, log_prob, value_est = agent.get_action(data)
        
        print(f"Action logits shape: {action_logits.shape}")
        print(f"Value estimate: {value_est:.4f}")
        print(f"Sampled action: {action}")
        print(f"Log probability: {log_prob:.4f}")
    
    return agent


def example_dataset_loading():
    """Example: Load and explore the dataset."""
    print("\n" + "="*50)
    print("Example: Dataset Loading")
    print("="*50)
    
    try:
        # Create dataset
        dataset = CircuitDataset(
            data_root='../testcase',
            sources=['EPFL', 'MCNC', 'ISCAS85']
        )
        
        print(f"Dataset loaded successfully!")
        print(f"Total circuits: {len(dataset)}")
        
        # Get statistics
        stats = dataset.get_statistics()
        print(f"Dataset statistics:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value}")
        
        # Sample some circuits
        print(f"\nSampling circuits...")
        for i in range(3):
            circuit_path, metadata = dataset.sample_circuit()
            print(f"  Circuit {i+1}: {metadata['name']} ({metadata['source']})")
            print(f"    Gates: {metadata.get('gate_count', 'N/A')}")
            print(f"    Difficulty: {metadata.get('difficulty', 'N/A')}")
        
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("This is expected if testcase directory is not available.")
        return None


def example_environment_usage():
    """Example: Use the synthesis environment."""
    print("\n" + "="*50)
    print("Example: Environment Usage")
    print("="*50)
    
    # Create environment
    env = SynthesisEnvironment(
        max_steps=5,
        action_space=['b', 'rw', 'rf', 'rwz', 'rfz'],
        reward_shaping=True,
        reward_normalization=True,
        final_bonus=True
    )
    
    print(f"Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Max steps: {env.max_steps}")
    
    # Note: This would require actual circuit files and ABC
    print("Note: Full environment testing requires circuit files and ABC synthesis tool.")
    print("This example shows the environment interface.")
    
    return env


def example_training_loop():
    """Example: Demonstrate training loop structure."""
    print("\n" + "="*50)
    print("Example: Training Loop Structure")
    print("="*50)
    
    # Load configuration
    config = load_config()
    
    # Create components
    agent = PPOSynthesisAgent(config)
    metrics_tracker = MetricsTracker()
    logger = setup_logger('example_training')
    
    print("Training components created:")
    print(f"  - Agent: {type(agent).__name__}")
    print(f"  - Metrics tracker: {type(metrics_tracker).__name__}")
    print(f"  - Logger: {type(logger).__name__}")
    
    # Simulate training statistics
    print("\nSimulating training statistics...")
    
    for episode in range(5):
        # Simulate episode data
        episode_info = {
            'total_reward': 10.0 + episode * 2.0,
            'num_steps': 8,
            'area_reduction': 15 + episode * 3,
            'area_reduction_percent': 12.5 + episode * 2.5,
            'initial_area': 120,
            'final_area': 105 - episode * 2,
            'best_area': 100 - episode * 3,
            'actions': ['b', 'rw', 'rf', 'rwz', 'rfz']
        }
        
        metrics_tracker.update_episode(episode_info)
        
        if episode % 2 == 0:
            stats = metrics_tracker.get_average_stats()
            logger.info(f"Episode {episode}: Avg reward = {stats['avg_reward']:.2f}, "
                       f"Avg area reduction = {stats['avg_area_reduction_percent']:.2f}%")
    
    # Get final statistics
    final_stats = metrics_tracker.get_average_stats()
    print(f"\nFinal training statistics:")
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def example_evaluation():
    """Example: Demonstrate evaluation process."""
    print("\n" + "="*50)
    print("Example: Evaluation Process")
    print("="*50)
    
    # Load configuration
    config = load_config()
    
    # Create agent (would load trained model in practice)
    agent = PPOSynthesisAgent(config)
    
    print("Evaluation components created:")
    print(f"  - Agent: {type(agent).__name__}")
    
    # Simulate evaluation results
    print("\nSimulating evaluation results...")
    
    eval_results = {
        'avg_reward': 15.5,
        'avg_area_reduction_percent': 18.2,
        'std_area_reduction_percent': 5.3,
        'min_area_reduction_percent': 8.1,
        'max_area_reduction_percent': 28.7,
        'avg_episode_length': 8.5,
        'num_episodes': 50
    }
    
    print("Evaluation Results:")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Simulate action statistics
    action_stats = {
        'action_percentages': {
            'b': 25.0,
            'rw': 30.0,
            'rf': 20.0,
            'rwz': 15.0,
            'rfz': 10.0
        }
    }
    
    print("\nAction Usage:")
    for action, percentage in action_stats['action_percentages'].items():
        print(f"  {action}: {percentage:.1f}%")


def main():
    """Run all examples."""
    print("RL Synthesis Agent - Examples")
    print("="*60)
    
    try:
        # Run examples
        example_agent_creation()
        example_dataset_loading()
        example_environment_usage()
        example_training_loop()
        example_evaluation()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
        print("\nNext steps:")
        print("1. Install ABC synthesis tool")
        print("2. Ensure testcase directory contains circuit files")
        print("3. Run: python train.py")
        print("4. Run: python eval.py --model outputs/models/best_model.pth")
        
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 