#!/usr/bin/env python3
"""
Very Tiny Training Script to Verify Flow on MacBook

This script runs a minimal training to verify everything works.
"""

import sys
import logging
from pathlib import Path
import torch

# Add project root to path
sys.path.append('.')

from models.ppo_agent import PPOSynthesisAgent, compute_gae, normalize_advantages
from env.synthesis_env import SynthesisEnvironment
from data.dataset import CircuitDataset
from utils.logger import setup_logger
from utils.metrics import MetricsTracker


def load_config():
    """Load minimal configuration for verification."""
    return {
        'gnn_encoder': {
            'type': 'GIN',
            'hidden_dim': 32,  # Very small
            'num_layers': 1,   # Minimal
            'pooling': 'mean'
        },
        'actor_head': {
            'input_dim': 33,  # 32 (GNN) + 1 (timestep)
            'layers': [32, 16],
            'output': 'softmax'
        },
        'critic_head': {
            'input_dim': 33,
            'layers': [32, 16],
            'output': 'scalar'
        },
        'learning_rate': 3e-4,
        'clip_range': 0.1,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'ppo_epochs': 1,  # Minimal
        'env': {
            'max_steps': 3,  # Very short episodes
            'action_space': ['b', 'rw', 'rf', 'rwz', 'rfz'],  # Full action space
            'reward_shaping': True,
            'reward_normalization': False,
            'final_bonus': False
        },
        'dataset': {
            'data_root': 'testcase',
            'sources': ['MCNC']
        }
    }


def collect_episode(agent, env, circuit_path):
    """Collect a single episode of experience."""
    observations = []
    actions = []
    rewards = []
    values = []
    log_probs = []
    
    # Reset environment
    obs, info = env.reset(circuit_path)
    
    # Check if circuit was skipped (area = 0)
    if info.get('skipped', False):
        print(f"[DEBUG] Skipping episode for circuit with area 0: {circuit_path}")
        episode_info = {
            'circuit_path': circuit_path,
            'total_reward': 0.0,
            'area_reduction': 0,
            'area_reduction_percent': 0.0,
            'best_area': 0,
            'num_steps': 0,
            'skipped': True
        }
        return observations, actions, rewards, values, log_probs, episode_info
    
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


def collect_batch(agent, env, dataset, batch_size=1):
    """Collect a minimal batch of episodes."""
    all_observations = []
    all_actions = []
    all_rewards = []
    all_values = []
    all_log_probs = []
    episode_infos = []
    
    # Sample circuits for this batch
    circuits = dataset.sample_circuits(batch_size)
    
    for circuit_path, metadata in circuits:
        print(f"Processing circuit: {metadata['name']}")
        observations, actions, rewards, values, log_probs, episode_info = collect_episode(
            agent, env, circuit_path
        )
        
        # Skip episodes that were skipped due to area = 0
        if episode_info.get('skipped', False):
            continue
        
        # Store episode data
        all_observations.extend(observations[:-1])
        all_actions.extend(actions)
        all_rewards.extend(rewards)
        all_values.extend(values)
        all_log_probs.extend(log_probs)
        episode_infos.append(episode_info)
    
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


def verify_flow():
    """Verify the complete training flow."""
    print("="*50)
    print("VERY TINY TRAINING VERIFICATION")
    print("="*50)
    
    # Load configuration
    config = load_config()
    
    # Setup logging
    logger = setup_logger('tiny_verify', level=logging.INFO)
    
    # Create dataset
    try:
        dataset = CircuitDataset(
            data_root=config['dataset'].get('data_root', 'testcase'),
            sources=config['dataset'].get('sources', ['MCNC'])
        )
        
        logger.info(f"Loaded {len(dataset)} circuits from dataset")
        
        if len(dataset) == 0:
            print("No circuits found. Exiting...")
            return
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create environment
    env_config = config.get('env', {})
    env = SynthesisEnvironment(
        max_steps=env_config.get('max_steps', 3),
        action_space=env_config.get('action_space', ['b', 'rw', 'rf']),
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
    
    # Training parameters (very minimal)
    num_episodes = 2  # Just 2 episodes
    batch_size = 1    # Batch size of 1
    
    # Create output directories
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    model_dir = output_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    
    print(f"Training for {num_episodes} episodes with batch size {batch_size}")
    
    try:
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            # Collect batch
            batch_data = collect_batch(agent, env, dataset, batch_size)
            
            if len(batch_data['observations']) == 0:
                print("No valid episodes in batch, continuing...")
                continue
            
            # Update agent
            training_stats = agent.update(batch_data)
            
            # Log episode statistics
            for episode_info in batch_data['episode_infos']:
                metrics_tracker.update_episode(episode_info)
            
            # Print training stats
            print(f"  Policy Loss: {training_stats['policy_loss']:.4f}")
            print(f"  Value Loss: {training_stats['value_loss']:.4f}")
            
            # Print episode stats
            try:
                avg_stats = metrics_tracker.get_average_stats()
                print(f"  Avg Reward: {avg_stats.get('avg_reward', 0.0):.3f}")
                print(f"  Avg Area Reduction: {avg_stats.get('avg_area_reduction_percent', 0.0):.2f}%")
            except Exception as e:
                print(f"  Could not get episode statistics: {e}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {e}")
        raise
    
    finally:
        # Save final model
        agent.save_model(model_dir / 'tiny_verify_model.pth')
        print("\nVerification completed. Model saved.")
        
        # Print final statistics
        try:
            final_stats = metrics_tracker.get_average_stats()
            print(f"Final Statistics:")
            print(f"  Avg Reward: {final_stats.get('avg_reward', 0.0):.3f}")
            print(f"  Avg Area Reduction: {final_stats.get('avg_area_reduction_percent', 0.0):.2f}%")
        except Exception as e:
            print(f"Could not get final statistics: {e}")
        
        print("\n" + "="*50)
        print("FLOW VERIFICATION COMPLETED SUCCESSFULLY!")
        print("="*50)


if __name__ == "__main__":
    verify_flow() 