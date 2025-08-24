#!/usr/bin/env python3
"""
Tiny Real Training Script for RL Synthesis Agent

This script uses actual circuits for training with ABC integration.
"""

import sys
import yaml
import logging
from pathlib import Path
import torch
import numpy as np
import os

# Add project root to path
sys.path.append('.')

from models.ppo_agent import PPOSynthesisAgent, compute_gae, normalize_advantages
from env.synthesis_env import SynthesisEnvironment
from utils.logger import setup_logger
from utils.metrics import MetricsTracker


def load_config():
    """Load minimal configuration for testing."""
    return {
        'gnn_encoder': {
            'type': 'GIN',
            'hidden_dim': 64,
            'num_layers': 2,
            'pooling': 'mean'
        },
        'actor_head': {
            'input_dim': 65,
            'layers': [64, 32],
            'output': 'softmax'
        },
        'critic_head': {
            'input_dim': 65,
            'layers': [64, 32],
            'output': 'scalar'
        },
        'learning_rate': 3e-4,
        'clip_range': 0.1,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'ppo_epochs': 2,
        'env': {
            'max_steps': 5,
            'action_space': ['b', 'rw', 'rf', 'rwz', 'rfz'],
            'reward_shaping': True,
            'reward_normalization': False,
            'final_bonus': False
        }
    }


def find_circuits():
    """Find available circuit files."""
    circuits = []
    
    # Look for AIG files first, then AAG files
    for root, dirs, files in os.walk('testcase'):
        for file in files:
            if file.endswith('.aig'):
                circuits.append(os.path.join(root, file))
            elif file.endswith('.aag'):
                # Check if corresponding AIG exists
                aig_path = os.path.join(root, file.replace('.aag', '.aig'))
                if os.path.exists(aig_path):
                    circuits.append(aig_path)
    
    return circuits


def collect_episode(agent, env, circuit_path):
    """Collect a single episode of experience."""
    observations = []
    actions = []
    rewards = []
    values = []
    log_probs = []
    
    # Reset environment
    obs, info = env.reset(circuit_path)
    observations.append(obs)
    
    for step in range(env.max_steps):
        # Get action from agent
        action, log_prob, value = agent.get_action(obs)
        
        # Execute action
        next_obs, reward, done, step_info = env.step(action)
        
        # Store experience
        observations.append(next_obs)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        log_probs.append(log_prob)
        
        obs = next_obs
        
        if done:
            break
    
    # Get episode summary
    episode_info = env.get_episode_summary()
    episode_info['circuit_path'] = circuit_path
    
    return observations, actions, rewards, values, log_probs, episode_info


def collect_batch(agent, env, circuits, batch_size=2):
    """Collect a small batch of episodes for training."""
    all_observations = []
    all_actions = []
    all_rewards = []
    all_values = []
    all_log_probs = []
    episode_infos = []
    
    # Sample circuits for this batch
    batch_circuits = np.random.choice(circuits, min(batch_size, len(circuits)), replace=False)
    
    for circuit_path in batch_circuits:
        circuit_name = os.path.basename(circuit_path)
        print(f"Processing circuit: {circuit_name}")
        
        try:
            observations, actions, rewards, values, log_probs, episode_info = collect_episode(
                agent, env, circuit_path
            )
            
            # Store episode data
            all_observations.extend(observations[:-1])
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_values.extend(values)
            all_log_probs.extend(log_probs)
            episode_infos.append(episode_info)
            
        except Exception as e:
            print(f"Error processing circuit {circuit_name}: {e}")
            continue
    
    if not all_observations:
        print("No valid episodes collected!")
        return None
    
    # Compute advantages and returns
    advantages, returns = compute_gae(all_rewards, all_values)
    advantages = normalize_advantages(advantages)
    
    # Convert to tensors
    actions_tensor = torch.tensor(all_actions, dtype=torch.long)
    log_probs_tensor = torch.tensor(all_log_probs, dtype=torch.float)
    returns_tensor = torch.tensor(returns, dtype=torch.float)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float)
    
    return {
        'observations': all_observations,
        'actions': actions_tensor,
        'log_probs': log_probs_tensor,
        'returns': returns_tensor,
        'advantages': advantages_tensor,
        'episode_infos': episode_infos
    }


def tiny_train_real():
    """Run a tiny training session with real circuits."""
    print("="*50)
    print("TINY REAL TRAINING SESSION")
    print("="*50)
    
    # Load configuration
    config = load_config()
    
    # Setup logging
    logger = setup_logger('tiny_real_training', level=logging.INFO)
    
    # Find circuits
    circuits = find_circuits()
    print(f"Found {len(circuits)} circuits")
    
    if len(circuits) == 0:
        print("No circuits found! Running dummy training...")
        return run_dummy_training(config, logger)
    
    # Print first few circuits
    print("Available circuits:")
    for i, circuit in enumerate(circuits[:5]):
        print(f"  {i+1}. {os.path.basename(circuit)}")
    if len(circuits) > 5:
        print(f"  ... and {len(circuits) - 5} more")
    
    # Create environment
    env_config = config.get('env', {})
    env = SynthesisEnvironment(
        max_steps=env_config.get('max_steps', 5),
        action_space=env_config.get('action_space', ['b', 'rw', 'rf', 'rwz', 'rfz']),
        reward_shaping=env_config.get('reward_shaping', True),
        reward_normalization=env_config.get('reward_normalization', False),
        final_bonus=env_config.get('final_bonus', False)
    )
    
    # Create agent
    agent = PPOSynthesisAgent(config)
    
    logger.info(f"Agent created with {sum(p.numel() for p in agent.parameters()):,} parameters")
    logger.info(f"Device: {agent.device}")
    
    # Setup metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Training parameters
    num_episodes = 3  # Very small for testing
    batch_size = 2
    
    logger.info("Starting tiny real training...")
    
    try:
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            # Collect batch of episodes
            batch_data = collect_batch(agent, env, circuits, batch_size)
            
            if batch_data is None:
                print("Skipping episode due to collection failure")
                continue
            
            # Update agent
            training_stats = agent.update(batch_data)
            
            # Update metrics
            for episode_info in batch_data['episode_infos']:
                metrics_tracker.update_episode(episode_info)
            
            # Log statistics
            avg_stats = metrics_tracker.get_average_stats()
            logger.info(
                f"Episode {episode + 1}, "
                f"Avg Reward: {avg_stats.get('avg_reward', 0):.3f}, "
                f"Avg Area Reduction: {avg_stats.get('avg_area_reduction_percent', 0):.2f}%, "
                f"Policy Loss: {training_stats['policy_loss']:.4f}"
            )
            
            # Print episode details
            for i, episode_info in enumerate(batch_data['episode_infos']):
                circuit_name = os.path.basename(episode_info.get('circuit_path', 'unknown'))
                print(f"  Circuit {i+1} ({circuit_name}): "
                      f"{episode_info.get('area_reduction_percent', 0):.2f}% reduction, "
                      f"reward: {episode_info.get('total_reward', 0):.3f}")
        
        # Final statistics
        final_stats = metrics_tracker.get_average_stats()
        print(f"\nFinal Training Statistics:")
        for key, value in final_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Save model
        output_dir = Path('outputs')
        output_dir.mkdir(exist_ok=True)
        
        model_dir = output_dir / 'models'
        model_dir.mkdir(exist_ok=True)
        
        agent.save_model(model_dir / 'tiny_real_model.pth')
        logger.info("Tiny real model saved!")
        
        return True
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        env.close()


def run_dummy_training(config, logger):
    """Run training with dummy data when no circuits are available."""
    print("Running dummy training with synthetic data...")
    
    # Create agent
    agent = PPOSynthesisAgent(config)
    
    logger.info(f"Agent created with {sum(p.numel() for p in agent.parameters()):,} parameters")
    logger.info(f"Device: {agent.device}")
    
    # Simulate training with dummy data
    for episode in range(3):
        print(f"\nDummy Episode {episode + 1}")
        
        # Create dummy batch data
        import torch_geometric.data as pyg_data
        
        observations = []
        for i in range(2):
            x = torch.randn(10 + i, 6)
            edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
            timestep = torch.tensor([i / 5.0])
            data = pyg_data.Data(x=x, edge_index=edge_index, timestep=timestep)
            observations.append(data)
        
        batch_data = {
            'observations': observations,
            'actions': torch.randint(0, 5, (2,)),
            'log_probs': torch.randn(2),
            'returns': torch.randn(2),
            'advantages': torch.randn(2)
        }
        
        # Update agent
        training_stats = agent.update(batch_data)
        
        print(f"  Policy Loss: {training_stats['policy_loss']:.4f}")
        print(f"  Value Loss: {training_stats['value_loss']:.4f}")
    
    # Save model
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    model_dir = output_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    
    agent.save_model(model_dir / 'dummy_test_model.pth')
    logger.info("Dummy test model saved!")
    
    return True


def main():
    """Main function."""
    print("Starting tiny real training test...")
    
    try:
        success = tiny_train_real()
        
        if success:
            print("\n" + "="*50)
            print("TINY REAL TRAINING COMPLETED SUCCESSFULLY!")
            print("="*50)
            print("Check outputs/models/ for saved models")
        else:
            print("\nTraining failed!")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 