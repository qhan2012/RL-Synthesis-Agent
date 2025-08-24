#!/usr/bin/env python3
"""
Training Script for RL Synthesis Agent

This script implements the main training loop for the PPO-based
logic synthesis optimization agent.
"""

import os
import sys
import yaml
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append('.')

from models.ppo_agent import PPOSynthesisAgent, compute_gae, normalize_advantages
from env.synthesis_env import SynthesisEnvironment
from data.dataset import CircuitDataset
from utils.logger import setup_logger
from utils.metrics import MetricsTracker


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def collect_episode(
    agent: PPOSynthesisAgent,
    env: SynthesisEnvironment,
    circuit_path: str
) -> Tuple[List, List, List, List, List, Dict]:
    """
    Collect a single episode of experience.
    
    Returns:
        tuple: (observations, actions, rewards, values, log_probs, episode_info)
    """
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
        # Return empty episode data
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
    episode_reward = 0.0
    
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
        
        episode_reward += reward
        obs = next_obs
        
        if done:
            break
    
    # Get episode summary
    episode_info = env.get_episode_summary()
    episode_info['circuit_path'] = circuit_path
    
    return observations, actions, rewards, values, log_probs, episode_info


def collect_batch(
    agent: PPOSynthesisAgent,
    env: SynthesisEnvironment,
    dataset: CircuitDataset,
    batch_size: int
) -> Dict:
    """
    Collect a batch of episodes for training.
    
    Args:
        agent: PPO agent
        env: Synthesis environment
        dataset: Circuit dataset
        batch_size: Number of episodes to collect
        
    Returns:
        Dict: Batch data for training
    """
    all_observations = []
    all_actions = []
    all_rewards = []
    all_values = []
    all_log_probs = []
    episode_infos = []
    
    # Sample circuits for this batch
    circuits = dataset.sample_circuits(batch_size)
    
    for circuit_path, metadata in circuits:
        observations, actions, rewards, values, log_probs, episode_info = collect_episode(
            agent, env, circuit_path
        )
        
        # Skip episodes that were skipped due to area = 0
        if episode_info.get('skipped', False):
            continue
        
        # Store episode data
        all_observations.extend(observations[:-1])  # Exclude final observation
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


def train(
    config: Dict,
    agent: PPOSynthesisAgent,
    env: SynthesisEnvironment,
    dataset: CircuitDataset,
    logger: logging.Logger,
    metrics_tracker: MetricsTracker
):
    """
    Main training loop with enhanced real-time validation.
    
    Args:
        config: Training configuration
        agent: PPO agent
        env: Synthesis environment
        dataset: Circuit dataset
        logger: Logger instance
        metrics_tracker: Metrics tracker
    """
    total_timesteps = config.get('total_timesteps', 10000000)
    n_steps = config.get('n_steps', 256)
    batch_size = config.get('batch_size', 64)
    
    # Enhanced validation configuration
    val_config = config.get('validation', {})
    val_interval_early = val_config.get('val_interval_early', 500)     # Frequent validation early
    val_interval_late = val_config.get('val_interval_late', 2000)      # Less frequent later
    val_transition_timesteps = val_config.get('val_transition_timesteps', 100000)  # When to transition
    val_episodes = val_config.get('val_episodes', 5)                   # Episodes per validation
    
    # Early stopping configuration
    early_stop_patience = val_config.get('early_stop_patience', 10)    # Stop after 10 bad validations
    early_stop_min_delta = val_config.get('early_stop_min_delta', 0.5) # Min improvement required
    
    # Other intervals
    eval_interval = config.get('eval_interval', 5000)      # Reduced from 10000
    log_interval = config.get('log_interval', 500)
    save_interval = config.get('save_interval', 50000)
    
    # Create output directories
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    model_dir = output_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    
    # Training loop variables
    timesteps = 0
    episode_count = 0
    
    # Validation tracking
    best_val_performance = 0.0
    val_patience_counter = 0
    val_history = []
    
    # Create validation subset (20% of dataset)
    val_size = max(1, len(dataset) // 5)
    val_circuits = dataset.sample_circuits(val_size)
    
    logger.info("Starting enhanced training with real-time validation...")
    logger.info(f"Validation: {val_episodes} episodes every {val_interval_early}->{val_interval_late} timesteps")
    logger.info(f"Early stopping: patience={early_stop_patience}, min_delta={early_stop_min_delta}")
    
    try:
        while timesteps < total_timesteps:
            # Collect batch of episodes
            batch_data = collect_batch(agent, env, dataset, batch_size)
            
            # Update agent
            training_stats = agent.update(batch_data)
            
            # Update metrics
            timesteps += len(batch_data['observations'])
            episode_count += len(batch_data['episode_infos'])
            
            # Log episode statistics
            for episode_info in batch_data['episode_infos']:
                metrics_tracker.update_episode(episode_info)
            
            # Log training statistics
            if episode_count % log_interval == 0:
                avg_stats = metrics_tracker.get_average_stats()
                training_stats.update(avg_stats)
                
                logger.info(
                    f"Episode {episode_count}, Timesteps {timesteps}, "
                    f"Avg Reward: {avg_stats['avg_reward']:.3f}, "
                    f"Avg Area Reduction: {avg_stats['avg_area_reduction_percent']:.2f}%, "
                    f"Policy Loss: {training_stats['policy_loss']:.4f}, "
                    f"Value Loss: {training_stats['value_loss']:.4f}"
                )
                
                # Reset metrics for next interval
                metrics_tracker.reset()
            
            # Adaptive validation frequency
            current_val_interval = val_interval_early if timesteps < val_transition_timesteps else val_interval_late
            
            # Real-time validation
            if timesteps % current_val_interval == 0 and timesteps > 0:
                logger.info(f"🔍 Running validation at timestep {timesteps}...")
                
                val_results = validate_agent(agent, env, val_circuits, val_episodes)
                val_performance = val_results['avg_area_reduction_percent']
                val_history.append(val_performance)
                
                logger.info(f"Validation Results: "
                          f"Avg Reward: {val_results['avg_reward']:.3f}, "
                          f"Avg Area Reduction: {val_performance:.2f}%, "
                          f"Success Rate: {val_results.get('success_rate', 0):.2f}")
                
                # Save best model based on validation performance
                if val_performance > best_val_performance + early_stop_min_delta:
                    best_val_performance = val_performance
                    val_patience_counter = 0
                    agent.save_model(str(model_dir / 'best_val_model.pth'))
                    logger.info(f"✅ New best validation model saved! Area reduction: {val_performance:.2f}%")
                else:
                    val_patience_counter += 1
                    logger.info(f"⏳ Validation patience: {val_patience_counter}/{early_stop_patience}")
                
                # Early stopping check
                if val_patience_counter >= early_stop_patience:
                    logger.info(f"🛑 Early stopping triggered after {val_patience_counter} validations without improvement")
                    break
            
            # Full evaluation (less frequent)
            if timesteps % eval_interval == 0 and timesteps > 0:
                logger.info(f"🧪 Running full evaluation at timestep {timesteps}...")
                eval_results = evaluate_agent(agent, env, dataset, num_episodes=10)
                logger.info(f"Evaluation Results: {eval_results}")
                
                # Save checkpoint based on evaluation
                if eval_results['avg_area_reduction_percent'] > metrics_tracker.best_area_reduction:
                    metrics_tracker.best_area_reduction = eval_results['avg_area_reduction_percent']
                    agent.save_model(str(model_dir / 'best_eval_model.pth'))
                    logger.info(f"New best evaluation model saved! Area reduction: {eval_results['avg_area_reduction_percent']:.2f}%")
            
            # Save checkpoint
            if timesteps % save_interval == 0 and timesteps > 0:
                agent.save_model(str(model_dir / f'checkpoint_timestep_{timesteps}.pth'))
                logger.info(f"Checkpoint saved at timestep {timesteps}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    
    finally:
        # Save final model
        agent.save_model(str(model_dir / 'final_model.pth'))
        logger.info("Training completed. Final model saved.")
        
        # Print validation summary
        if val_history:
            logger.info(f"Validation Summary:")
            logger.info(f"  Best validation performance: {best_val_performance:.2f}%")
            logger.info(f"  Final validation performance: {val_history[-1]:.2f}%")
            logger.info(f"  Validation improvement: {val_history[-1] - val_history[0]:.2f}%")


def evaluate_agent(
    agent: PPOSynthesisAgent,
    env: SynthesisEnvironment,
    dataset: CircuitDataset,
    num_episodes: int = 10
) -> Dict:
    """
    Evaluate the agent on a set of test circuits.
    
    Args:
        agent: PPO agent
        env: Synthesis environment
        dataset: Circuit dataset
        num_episodes: Number of evaluation episodes
        
    Returns:
        Dict: Evaluation results
    """
    agent.eval()
    
    eval_results = {
        'episode_rewards': [],
        'area_reductions': [],
        'area_reduction_percents': [],
        'best_areas': [],
        'episode_lengths': []
    }
    
    # Sample test circuits
    test_circuits = dataset.sample_circuits(num_episodes)
    
    for circuit_path, metadata in test_circuits:
        observations, actions, rewards, values, log_probs, episode_info = collect_episode(
            agent, env, circuit_path
        )
        
        eval_results['episode_rewards'].append(episode_info['total_reward'])
        eval_results['area_reductions'].append(episode_info['area_reduction'])
        eval_results['area_reduction_percents'].append(episode_info['area_reduction_percent'])
        eval_results['best_areas'].append(episode_info['best_area'])
        eval_results['episode_lengths'].append(episode_info['num_steps'])
    
    # Compute averages
    results = {
        'avg_reward': np.mean(eval_results['episode_rewards']),
        'avg_area_reduction': np.mean(eval_results['area_reductions']),
        'avg_area_reduction_percent': np.mean(eval_results['area_reduction_percents']),
        'avg_best_area': np.mean(eval_results['best_areas']),
        'avg_episode_length': np.mean(eval_results['episode_lengths']),
        'std_reward': np.std(eval_results['episode_rewards']),
        'std_area_reduction_percent': np.std(eval_results['area_reduction_percents'])
    }
    
    return results


def validate_agent(
    agent: PPOSynthesisAgent,
    env: SynthesisEnvironment,
    val_circuits: List,
    num_episodes: int = 5
) -> Dict:
    """
    Validate the agent on a fixed set of validation circuits.
    
    Args:
        agent: PPO agent
        env: Synthesis environment
        val_circuits: List of validation circuits
        num_episodes: Number of episodes to validate
        
    Returns:
        Dict: Validation results
    """
    agent.eval()
    
    val_results = {
        'episode_rewards': [],
        'area_reductions': [],
        'area_reduction_percents': [],
        'best_areas': [],
        'episode_lengths': []
    }
    
    # Use the same validation circuits for consistency
    for i, (circuit_path, metadata) in enumerate(val_circuits[:num_episodes]):
        observations, actions, rewards, values, log_probs, episode_info = collect_episode(
            agent, env, circuit_path
        )
        
        if not episode_info.get('skipped', False):
            val_results['episode_rewards'].append(episode_info['total_reward'])
            val_results['area_reductions'].append(episode_info['area_reduction'])
            val_results['area_reduction_percents'].append(episode_info['area_reduction_percent'])
            val_results['best_areas'].append(episode_info['best_area'])
            val_results['episode_lengths'].append(episode_info['num_steps'])
    
    # Compute averages
    if val_results['episode_rewards']:
        results = {
            'avg_reward': np.mean(val_results['episode_rewards']),
            'avg_area_reduction': np.mean(val_results['area_reductions']),
            'avg_area_reduction_percent': np.mean(val_results['area_reduction_percents']),
            'avg_best_area': np.mean(val_results['best_areas']),
            'avg_episode_length': np.mean(val_results['episode_lengths']),
            'std_reward': np.std(val_results['episode_rewards']),
            'std_area_reduction_percent': np.std(val_results['area_reduction_percents']),
            'success_rate': np.mean([r > 0 for r in val_results['area_reduction_percents']]),
            'num_episodes': len(val_results['episode_rewards'])
        }
    else:
        results = {
            'avg_reward': 0.0,
            'avg_area_reduction': 0.0,
            'avg_area_reduction_percent': 0.0,
            'avg_best_area': 0.0,
            'avg_episode_length': 0.0,
            'std_reward': 0.0,
            'std_area_reduction_percent': 0.0,
            'success_rate': 0.0,
            'num_episodes': 0
        }
    
    agent.train()  # Return to training mode
    return results


def main():
    """Main training function."""
    # Load configuration
    config_path = 'config/ppo_config.yaml'
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logger('rl_synthesis_training', level=logging.INFO)
    
    # Setup metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Create dataset
    dataset = CircuitDataset(
        data_root=config['dataset'].get('data_root', 'testcase'),
        sources=config['dataset'].get('sources', ['EPFL', 'MCNC', 'ISCAS85'])
    )
    
    logger.info(f"Loaded {len(dataset)} circuits from dataset")
    logger.info(f"Dataset statistics: {dataset.get_statistics()}")
    
    # Create environment
    env_config = config.get('env', {})
    env = SynthesisEnvironment(
        max_steps=env_config.get('max_steps', 10),
        action_space=env_config.get('action_space', ['b', 'rw', 'rf', 'rwz', 'rfz']),
        reward_shaping=env_config.get('reward_shaping', True),
        reward_normalization=env_config.get('reward_normalization', True),
        final_bonus=env_config.get('final_bonus', True),
        cleanup_logs=True  # Clean up logs during large training
    )
    
    # Create agent
    agent = PPOSynthesisAgent(config)
    
    logger.info(f"Agent created with {sum(p.numel() for p in agent.parameters())} parameters")
    logger.info(f"Device: {agent.device}")
    
    # Start training
    train(config, agent, env, dataset, logger, metrics_tracker)


if __name__ == "__main__":
    main() 