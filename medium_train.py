#!/usr/bin/env python3
"""
Medium Training Script for RL Synthesis Agent

This is a medium-scale training optimized for 2-3 hours on 3080 GPU.
Uses the same GNN and RL architecture as normal training but with enhanced monitoring.
"""

import sys
import os
import yaml
import logging
import time
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.append('.')

from models.ppo_agent import PPOSynthesisAgent, compute_gae, normalize_advantages
from env.synthesis_env import SynthesisEnvironment
from data.dataset import CircuitDataset
from utils.logger import setup_logger
from utils.metrics import MetricsTracker
from utils.tensorboard_monitor import TensorBoardMonitor


def load_medium_config():
    """Load medium-scale configuration for 2-3 hours training on 3080."""
    return {
        'algorithm': 'PPO',
        'total_timesteps': 250000,  # Target for 2-3 hours
        'n_steps': 256,  # Increased from small training
        'batch_size': 128,  # INCREASED: Reduce gradient noise (was 64)
        'ppo_epochs': 4,
        'gamma': 0.95,
        'gae_lambda': 0.90,  # REDUCED: Lower variance in GAE (was 0.95)
        'clip_range': 0.1,
        'value_loss_coef': 0.3,  # REDUCED: Less dominance of value loss (was 0.5)
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'learning_rate': 3e-4,
        'lr_schedule': 'linear',
        
        # NEW: Return normalization
        'normalize_returns': True,
        'return_normalization_epsilon': 1e-8,
        
        # Same GNN architecture as normal training
        'gnn_encoder': {
            'type': 'GIN',
            'num_layers': 5,        # INCREASED: from 3 to 5 layers
            'hidden_dim': 256,       # INCREASED: from 128 to 256
            'activation': 'ReLU',
            'pooling': 'mean',
            'use_global_features': True
        },
        
        # Enhanced actor-critic heads with updated dimensions
        'actor_head': {
            'input_dim': 257,  # 256 (GNN) + 1 (timestep) - UPDATED
            'layers': [256, 128, 64],  # ENHANCED: deeper network
            'output': 'softmax'
        },
        'critic_head': {
            'input_dim': 257,  # 256 (GNN) + 1 (timestep) - UPDATED
            'layers': [512, 256, 128, 64],  # ENHANCED: much deeper network
            'dropout': 0.1,
            'output': 'scalar'
        },
        
        # Same timestep handling
        'timestep': {
            'use': True,
            'input': 'current_step',
            'normalization': True,
            'inject_location': 'after_gnn_pooling'
        },
        
        # Same environment config
        'env': {
            'max_steps': 10,
            'action_space': ['b', 'rw', 'rf', 'rwz', 'rfz'],
            'reward_shaping': True,
            'reward_normalization': True,
            'final_bonus': True
        },
        
        # Expanded dataset for medium training
        'dataset': {
            'type': 'offline_circuit_library',
            'sources': ['IWLS', 'MCNC'],  # Same sources as normal training
            'sampling': 'random_circuit_per_episode',
            'augmentation': True,
            'caching': True,
            'max_circuits': 150  # Increased from 50 for more diversity
        },
        
        # Enhanced validation for medium training
        'validation': {
            'val_interval_early': 1000,         # More frequent validation early
            'val_interval_late': 2500,          # Less frequent later
            'val_transition_timesteps': 50000,  # Transition at 50k timesteps
            'val_episodes': 5,                  # More episodes per validation
            'early_stop_patience': 8,           # Higher patience for longer training
            'early_stop_min_delta': 0.2,       # Lower threshold for medium training
        },
        
        # Enhanced evaluation for medium training
        'eval': {
            'eval_interval': 5000,              # Evaluate every 5k timesteps
            'eval_episodes': 10,                # More episodes for evaluation
            'metrics': ['gate_count', 'gate_reduction_percent', 'best_area', 'avg_episode_reward'],
            'test_set': 'held_out_circuits',
            'save_best_model': True,
            'online_eval': True,                # Enable online evaluation
            'eval_patience': 5,                 # Early stopping based on evaluation
            'eval_min_delta': 0.1,              # Minimum improvement for evaluation
        },
        
        # Enhanced logging for monitoring
        'logging': {
            'log_interval': 500,  # Log every 500 episodes
            'log_items': ['reward', 'advantage', 'entropy', 'policy_loss', 'value_loss', 'learning_rate'],
            'tensorboard_log_interval': 100,  # TensorBoard logging frequency
        },
        
        # Model checkpointing
        'checkpoint': {
            'save_interval': 10000,     # Save checkpoint every 10k timesteps
            'keep_best_n': 3,           # Keep best 3 models
            'save_optimizer': True,     # Save optimizer state
        }
    }


def collect_episode(agent, env, circuit_path):
    """Collect a single episode of experience."""
    observations = []
    actions = []
    rewards = []
    values = []
    log_probs = []
    action_probs_list = []
    
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
            'skipped': True,
            'step_rewards': [],
            'actions_taken': []
        }
        return observations, actions, rewards, values, log_probs, episode_info
    
    observations.append(obs)
    episode_reward = 0.0
    actions_taken = []
    
    for step in range(env.max_steps):
        # Get action from agent with action probabilities
        action, log_prob, value, action_probs = agent.get_action(obs, return_probs=True)
        
        # Execute action
        next_obs, reward, done, step_info = env.step(action)
        
        # Store experience
        observations.append(next_obs)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        log_probs.append(log_prob)
        if action_probs is not None:
            action_probs_list.append(action_probs)
        actions_taken.append(action)
        
        episode_reward += reward
        obs = next_obs
        
        if done:
            break
    
    # Get episode summary
    episode_info = env.get_episode_summary()
    episode_info['circuit_path'] = circuit_path
    episode_info['step_rewards'] = rewards
    episode_info['actions_taken'] = actions_taken
    episode_info['action_probs'] = action_probs_list
    
    return observations, actions, rewards, values, log_probs, episode_info


def collect_batch(agent, env, dataset, batch_size):
    """Collect a batch of episodes for training."""
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
        all_observations.extend(observations[:-1])  # Exclude final observation
        all_actions.extend(actions)
        all_rewards.extend(rewards)
        all_values.extend(values)
        all_log_probs.extend(log_probs)
        episode_infos.append(episode_info)
    
    # Compute advantages and returns
    advantages, returns = compute_gae(all_rewards, all_values)
    advantages = normalize_advantages(advantages)
    
    # NEW: Normalize returns to reduce value loss variance
    if len(returns) > 1:  # Only normalize if we have multiple returns
        returns_array = np.array(returns)
        returns_mean = returns_array.mean()
        returns_std = returns_array.std()
        if returns_std > 1e-8:  # Avoid division by zero
            returns = ((returns_array - returns_mean) / (returns_std + 1e-8)).tolist()
            print(f"[DEBUG] Normalized returns: mean={returns_mean:.3f}, std={returns_std:.3f}")
    
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


def evaluate_agent_medium(agent, env, dataset, num_episodes=10):
    """Evaluate agent on a set of circuits."""
    print(f"🧪 Starting evaluation with {num_episodes} episodes...")
    
    # Set agent to evaluation mode
    agent.eval()
    
    # Sample circuits for evaluation
    eval_circuits = dataset.sample_circuits(num_episodes)
    
    results = []
    total_reward = 0.0
    total_area_reduction = 0.0
    successful_episodes = 0
    
    for circuit_path, metadata in eval_circuits:
        print(f"  Evaluating circuit: {metadata['name']}")
        
        # Run episode
        observations, actions, rewards, values, log_probs, episode_info = collect_episode(
            agent, env, circuit_path
        )
        
        # Skip if circuit was skipped
        if episode_info.get('skipped', False):
            continue
        
        # Collect results
        episode_reward = episode_info['total_reward']
        area_reduction_percent = episode_info.get('area_reduction_percent', 0.0)
        
        results.append({
            'circuit': metadata['name'],
            'reward': episode_reward,
            'area_reduction_percent': area_reduction_percent,
            'num_steps': episode_info.get('num_steps', 0),
            'initial_area': episode_info.get('initial_area', 0),
            'final_area': episode_info.get('final_area', 0),
            'best_area': episode_info.get('best_area', 0)
        })
        
        total_reward += episode_reward
        total_area_reduction += area_reduction_percent
        successful_episodes += 1
        
        print(f"    Result: Reward={episode_reward:.3f}, Area Reduction={area_reduction_percent:.2f}%")
    
    if successful_episodes == 0:
        print("⚠️  No successful episodes in evaluation!")
        return {
            'avg_reward': 0.0,
            'avg_area_reduction_percent': 0.0,
            'success_rate': 0.0,
            'results': results
        }
    
    # Calculate averages
    avg_reward = total_reward / successful_episodes
    avg_area_reduction_percent = total_area_reduction / successful_episodes
    success_rate = successful_episodes / len(eval_circuits)
    
    print(f"📊 Evaluation Results:")
    print(f"  Average Reward: {avg_reward:.3f}")
    print(f"  Average Area Reduction: {avg_area_reduction_percent:.2f}%")
    print(f"  Success Rate: {success_rate:.2f}")
    print(f"  Successful Episodes: {successful_episodes}/{len(eval_circuits)}")
    
    # Return agent to training mode
    agent.train()
    
    return {
        'avg_reward': avg_reward,
        'avg_area_reduction_percent': avg_area_reduction_percent,
        'success_rate': success_rate,
        'results': results
    }


def validate_agent_medium(agent, env, val_circuits, num_episodes=5):
    """Validate agent on validation circuits."""
    print(f"🔍 Starting validation with {num_episodes} episodes...")
    
    # Set agent to evaluation mode
    agent.eval()
    
    # Use a subset of validation circuits
    val_subset = val_circuits[:num_episodes]
    
    results = []
    total_reward = 0.0
    total_area_reduction = 0.0
    successful_episodes = 0
    
    for circuit_path, metadata in val_subset:
        print(f"  Validating circuit: {metadata['name']}")
        
        # Run episode
        observations, actions, rewards, values, log_probs, episode_info = collect_episode(
            agent, env, circuit_path
        )
        
        # Skip if circuit was skipped
        if episode_info.get('skipped', False):
            continue
        
        # Collect results
        episode_reward = episode_info['total_reward']
        area_reduction_percent = episode_info.get('area_reduction_percent', 0.0)
        
        results.append({
            'circuit': metadata['name'],
            'reward': episode_reward,
            'area_reduction_percent': area_reduction_percent
        })
        
        total_reward += episode_reward
        total_area_reduction += area_reduction_percent
        successful_episodes += 1
    
    if successful_episodes == 0:
        return {
            'avg_reward': 0.0,
            'avg_area_reduction_percent': 0.0,
            'success_rate': 0.0,
            'results': results
        }
    
    # Calculate averages
    avg_reward = total_reward / successful_episodes
    avg_area_reduction_percent = total_area_reduction / successful_episodes
    success_rate = successful_episodes / len(val_subset)
    
    # Return agent to training mode
    agent.train()
    
    return {
        'avg_reward': avg_reward,
        'avg_area_reduction_percent': avg_area_reduction_percent,
        'success_rate': success_rate,
        'results': results
    }


def medium_train():
    """Run medium training session optimized for 2-3 hours on 3080."""
    print("="*60)
    print("MEDIUM TRAINING SESSION (2-3 hours on 3080)")
    print("="*60)
    
    start_time = time.time()
    
    # Load configuration
    config = load_medium_config()
    
    # Setup logging
    logger = setup_logger('medium_training', level=logging.INFO)
    
    # Create dataset with expanded circuits
    try:
        dataset = CircuitDataset(
            data_root=config['dataset'].get('data_root', 'testcase'),
            sources=config['dataset'].get('sources', ['IWLS', 'MCNC'])
        )
        
        # Limit the number of circuits for medium training
        max_circuits = config['dataset'].get('max_circuits', 150)
        if len(dataset) > max_circuits:
            # Take a subset of circuits
            dataset.circuits = dataset.circuits[:max_circuits]
            logger.info(f"Limited dataset to {len(dataset)} circuits for medium training")
        
        logger.info(f"Loaded {len(dataset)} circuits from dataset (IWLS, MCNC)")
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return False
    
    # Create environment
    env_config = config.get('env', {})
    env = SynthesisEnvironment(
        max_steps=env_config.get('max_steps', 10),
        action_space=env_config.get('action_space', ['b', 'rw', 'rf', 'rwz', 'rfz']),
        reward_shaping=env_config.get('reward_shaping', True),
        reward_normalization=env_config.get('reward_normalization', True),
        final_bonus=env_config.get('final_bonus', True),
        cleanup_logs=False  # Keep logs for debugging
    )
    
    # Create agent
    agent = PPOSynthesisAgent(config)
    
    logger.info(f"Agent created with {sum(p.numel() for p in agent.parameters()):,} parameters")
    logger.info(f"Device: {agent.device}")
    
    # Setup metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Training parameters
    total_timesteps = config.get('total_timesteps', 250000)
    n_steps = config.get('n_steps', 256)
    batch_size = config.get('batch_size', 64)
    
    # Enhanced validation configuration
    val_config = config.get('validation', {})
    val_interval_early = val_config.get('val_interval_early', 1000)
    val_interval_late = val_config.get('val_interval_late', 2500)
    val_transition_timesteps = val_config.get('val_transition_timesteps', 50000)
    val_episodes = val_config.get('val_episodes', 5)
    
    # Early stopping configuration
    early_stop_patience = val_config.get('early_stop_patience', 8)
    early_stop_min_delta = val_config.get('early_stop_min_delta', 0.2)
    
    # Enhanced evaluation configuration
    eval_config = config.get('eval', {})
    eval_interval = eval_config.get('eval_interval', 5000)
    eval_episodes = eval_config.get('eval_episodes', 10)
    eval_patience = eval_config.get('eval_patience', 5)
    eval_min_delta = eval_config.get('eval_min_delta', 0.1)
    
    # Logging configuration
    log_config = config.get('logging', {})
    log_interval = log_config.get('log_interval', 500)
    tb_log_interval = log_config.get('tensorboard_log_interval', 100)
    
    # Checkpoint configuration
    checkpoint_config = config.get('checkpoint', {})
    save_interval = checkpoint_config.get('save_interval', 10000)
    
    # Create output directories
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    model_dir = output_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    
    # Setup comprehensive TensorBoard monitor
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = output_dir / 'tensorboard_logs'
    tb_dir.mkdir(exist_ok=True)
    tb_log_dir = tb_dir / f'medium_train_{current_time}'
    monitor = TensorBoardMonitor(log_dir=str(tb_log_dir))
    
    logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
    print(f"📊 TensorBoard logs: {tb_log_dir}")
    
    # Log hyperparameters
    hparams = {
        'total_timesteps': total_timesteps,
        'batch_size': batch_size,
        'learning_rate': config.get('learning_rate', 3e-4),
        'gamma': config.get('gamma', 0.95),
        'gae_lambda': config.get('gae_lambda', 0.95),
        'clip_range': config.get('clip_range', 0.1),
        'value_loss_coef': config.get('value_loss_coef', 0.5),
        'entropy_coef': config.get('entropy_coef', 0.01),
        'val_episodes': val_episodes,
        'eval_episodes': eval_episodes,
        'val_patience': early_stop_patience,
        'eval_patience': eval_patience
    }
    monitor.log_hyperparameters(hparams, {'training_started': 1.0})
    
    logger.info("Starting medium training with real-time validation and evaluation...")
    logger.info(f"Target timesteps: {total_timesteps}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Validation: {val_episodes} episodes every {val_interval_early}->{val_interval_late} timesteps")
    logger.info(f"Evaluation: {eval_episodes} episodes every {eval_interval} timesteps")
    logger.info(f"Early stopping: val_patience={early_stop_patience}, eval_patience={eval_patience}")
    
    # Training loop variables
    timesteps = 0
    episode_count = 0
    
    # Validation tracking
    best_val_performance = 0.0
    val_patience_counter = 0
    val_history = []
    last_validation_timestep = 0
    
    # Evaluation tracking
    best_eval_performance = 0.0
    eval_patience_counter = 0
    eval_history = []
    last_evaluation_timestep = 0
    
    # Create validation and evaluation subsets
    val_size = max(1, len(dataset) // 5)  # 20% for validation
    eval_size = max(1, len(dataset) // 10)  # 10% for evaluation
    
    val_circuits = dataset.sample_circuits(val_size)
    eval_circuits = dataset.sample_circuits(eval_size)
    
    logger.info(f"Validation set: {len(val_circuits)} circuits")
    logger.info(f"Evaluation set: {len(eval_circuits)} circuits")
    
    try:
        with tqdm(total=total_timesteps, desc="Training Progress") as pbar:
            while timesteps < total_timesteps:
                batch_start_time = time.time()
                
                # Collect batch of episodes
                batch_data = collect_batch(agent, env, dataset, batch_size)
                
                # Update agent
                training_stats = agent.update(batch_data)
                
                # Update metrics
                batch_timesteps = len(batch_data['observations'])
                timesteps += batch_timesteps
                episode_count += len(batch_data['episode_infos'])
                
                # Update progress bar
                pbar.update(batch_timesteps)
                
                # Log episode statistics
                for episode_info in batch_data['episode_infos']:
                    metrics_tracker.update_episode(episode_info)
                
                batch_time = time.time() - batch_start_time
                
                # Log training statistics
                if episode_count % log_interval == 0:
                    avg_stats = metrics_tracker.get_average_stats()
                    training_stats.update(avg_stats)
                    
                    elapsed_time = time.time() - start_time
                    logger.info(
                        f"Episode {episode_count}, Timesteps {timesteps}, "
                        f"Time: {elapsed_time:.1f}s, "
                        f"Avg Reward: {avg_stats['avg_reward']:.3f}, "
                        f"Avg Area Reduction: {avg_stats['avg_area_reduction_percent']:.2f}%, "
                        f"Policy Loss: {training_stats['policy_loss']:.4f}, "
                        f"Value Loss: {training_stats['value_loss']:.4f}"
                    )
                    
                    # Reset metrics for next interval
                    metrics_tracker.reset()
                
                # Log comprehensive metrics using TensorBoard monitor
                monitor.log_training_metrics(timesteps, training_stats)
                
                # Log GNN metrics if available
                if hasattr(agent, 'get_gnn_metrics'):
                    try:
                        gnn_stats = agent.get_gnn_metrics()
                        if gnn_stats:
                            monitor.log_gnn_metrics(timesteps, gnn_stats)
                    except Exception as e:
                        # Don't fail training if GNN metrics collection fails
                        logger.debug(f"Failed to collect GNN metrics: {e}")
                        pass
                
                # Log batch-level metrics
                batch_info = {
                    'batch_size': len(batch_data['episode_infos']),
                    'batch_time': batch_time,
                    'episodes_per_second': len(batch_data['episode_infos']) / batch_time,
                    'timesteps_per_second': batch_timesteps / batch_time
                }
                monitor.log_batch_metrics(timesteps, batch_info)
                
                # Log episode-level metrics with enhanced monitoring
                for episode_info in batch_data['episode_infos']:
                    monitor.log_agent_performance(timesteps, episode_info)
                    monitor.log_qor_metrics(timesteps, episode_info)
                    
                    # Log environment metrics with action information
                    if 'actions_taken' in episode_info and len(episode_info['actions_taken']) > 0:
                        # Use the last action taken as representative
                        last_action = episode_info['actions_taken'][-1]
                        action_probs = episode_info.get('action_probs', [None])[-1] if episode_info.get('action_probs') else None
                        monitor.log_environment_metrics(timesteps, episode_info, 
                                                       action_probs=action_probs, 
                                                       action_taken=last_action)
                    else:
                        monitor.log_environment_metrics(timesteps, episode_info)
                
                # Log summary statistics periodically
                if timesteps % 5000 == 0:
                    monitor.log_summary_statistics(timesteps)
                
                # Adaptive validation frequency
                current_val_interval = val_interval_early if timesteps < val_transition_timesteps else val_interval_late
                
                # Real-time validation - trigger when we cross validation thresholds
                next_validation_timestep = last_validation_timestep + current_val_interval
                if timesteps >= next_validation_timestep and timesteps > 0:
                    logger.info(f"🔍 Running validation at timestep {timesteps}...")
                    
                    val_results = validate_agent_medium(agent, env, val_circuits, val_episodes)
                    val_performance = val_results['avg_area_reduction_percent']
                    val_history.append(val_performance)
                    
                    logger.info(f"Validation Results: "
                              f"Avg Reward: {val_results['avg_reward']:.3f}, "
                              f"Avg Area Reduction: {val_performance:.2f}%, "
                              f"Success Rate: {val_results.get('success_rate', 0):.2f}")
                    
                    # Update last validation timestep
                    last_validation_timestep = timesteps
                    
                    # Log validation metrics to TensorBoard
                    monitor.log_validation_metrics(timesteps, val_results)
                    
                    # Save best model based on validation performance
                    if val_performance > best_val_performance + early_stop_min_delta:
                        best_val_performance = val_performance
                        val_patience_counter = 0
                        agent.save_model(str(model_dir / 'medium_best_val_model.pth'))
                        logger.info(f"✅ New best validation model saved! Area reduction: {val_performance:.2f}%")
                    else:
                        val_patience_counter += 1
                        logger.info(f"⏳ Validation patience: {val_patience_counter}/{early_stop_patience}")
                    
                    # Early stopping check based on validation
                    if val_patience_counter >= early_stop_patience:
                        logger.info(f"🛑 Early stopping triggered after {val_patience_counter} validations without improvement")
                        break
                
                # Online evaluation - trigger when we cross evaluation thresholds
                next_evaluation_timestep = last_evaluation_timestep + eval_interval
                if timesteps >= next_evaluation_timestep and timesteps > 0:
                    logger.info(f"🧪 Running online evaluation at timestep {timesteps}...")
                    
                    eval_results = evaluate_agent_medium(agent, env, dataset, eval_episodes)
                    eval_performance = eval_results['avg_area_reduction_percent']
                    eval_history.append(eval_performance)
                    
                    logger.info(f"Evaluation Results: "
                              f"Avg Reward: {eval_results['avg_reward']:.3f}, "
                              f"Avg Area Reduction: {eval_performance:.2f}%, "
                              f"Success Rate: {eval_results.get('success_rate', 0):.2f}")
                    
                    # Update last evaluation timestep
                    last_evaluation_timestep = timesteps
                    
                    # Log evaluation metrics to TensorBoard
                    monitor.log_evaluation_metrics(timesteps, eval_results)
                    
                    # Save best model based on evaluation performance
                    if eval_performance > best_eval_performance + eval_min_delta:
                        best_eval_performance = eval_performance
                        eval_patience_counter = 0
                        agent.save_model(str(model_dir / 'medium_best_eval_model.pth'))
                        logger.info(f"✅ New best evaluation model saved! Area reduction: {eval_performance:.2f}%")
                    else:
                        eval_patience_counter += 1
                        logger.info(f"⏳ Evaluation patience: {eval_patience_counter}/{eval_patience}")
                    
                    # Early stopping check based on evaluation
                    if eval_patience_counter >= eval_patience:
                        logger.info(f"🛑 Early stopping triggered after {eval_patience_counter} evaluations without improvement")
                        break
                
                # Save checkpoint
                if timesteps % save_interval == 0 and timesteps > 0:
                    checkpoint_path = model_dir / f'medium_checkpoint_{timesteps}.pth'
                    agent.save_model(str(checkpoint_path))
                    logger.info(f"💾 Checkpoint saved at timestep {timesteps}")
                
                # Check if we've exceeded time limit (3.5 hours max)
                elapsed_time = time.time() - start_time
                if elapsed_time > 12600:  # 3.5 hours
                    logger.info("Reached time limit, stopping training")
                    break
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Log final summary statistics
        monitor.log_summary_statistics(timesteps)
        
        # Log final hyperparameters with results
        if val_history and eval_history:
            final_metrics = {
                'final_validation_performance': val_history[-1] if val_history else 0.0,
                'best_validation_performance': best_val_performance,
                'final_evaluation_performance': eval_history[-1] if eval_history else 0.0,
                'best_evaluation_performance': best_eval_performance,
                'total_timesteps_completed': timesteps,
                'total_episodes_completed': episode_count,
                'training_time_hours': (time.time() - start_time) / 3600
            }
            monitor.log_hyperparameters(hparams, final_metrics)
        
        # Close TensorBoard monitor
        monitor.close()
        
        # Save final model
        agent.save_model(str(model_dir / 'medium_final_model.pth'))
        elapsed_time = time.time() - start_time
        logger.info(f"Training completed in {elapsed_time:.1f} seconds ({elapsed_time/3600:.2f} hours). Final model saved.")
        
        # Print validation summary
        if val_history:
            logger.info(f"Validation Summary:")
            logger.info(f"  Best validation performance: {best_val_performance:.2f}%")
            logger.info(f"  Final validation performance: {val_history[-1]:.2f}%")
            logger.info(f"  Total validations: {len(val_history)}")
            logger.info(f"  Validation improvement: {val_history[-1] - val_history[0]:.2f}%")
        
        # Print evaluation summary
        if eval_history:
            logger.info(f"Evaluation Summary:")
            logger.info(f"  Best evaluation performance: {best_eval_performance:.2f}%")
            logger.info(f"  Final evaluation performance: {eval_history[-1]:.2f}%")
            logger.info(f"  Total evaluations: {len(eval_history)}")
            logger.info(f"  Evaluation improvement: {eval_history[-1] - eval_history[0]:.2f}%")
        
        # Final statistics
        final_stats = metrics_tracker.get_average_stats()
        print(f"\nFinal Training Statistics:")
        for key, value in final_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\nTraining Summary:")
        print(f"  Total timesteps: {timesteps:,}")
        print(f"  Total episodes: {episode_count:,}")
        print(f"  Training time: {elapsed_time:.1f}s ({elapsed_time/3600:.2f} hours)")
        print(f"  Best validation: {best_val_performance:.2f}%")
        print(f"  Best evaluation: {best_eval_performance:.2f}%")
        
        env.close()
        return True


def main():
    """Main function."""
    print("Starting medium training test...")
    
    try:
        success = medium_train()
        
        if success:
            print("\n" + "="*60)
            print("MEDIUM TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("Check outputs/models/ for saved models")
            print("Check outputs/tensorboard_logs/ for TensorBoard logs")
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