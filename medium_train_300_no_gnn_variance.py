#!/usr/bin/env python3
"""
Medium Training Script with 300 Circuits and Balanced Test Classification (No GNN Variance Loss)

This script uses the medium training approach from medium_train_300_with_balanced_splits.py but
removes the GNN variance loss component to test its impact on training performance.
"""

import sys
import os
import yaml
import logging
import time
import random
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("TensorBoard not available, using fallback logging")
    SummaryWriter = None

# Add project root to path
sys.path.append('.')

from models.ppo_agent import PPOSynthesisAgent, compute_gae, normalize_advantages
from env.synthesis_env import SynthesisEnvironment
from data.dataset import CircuitDataset
from utils.logger import setup_logger
from utils.metrics import MetricsTracker
from utils.tensorboard_monitor import TensorBoardMonitor
from partitioned_circuit_splitter_balanced import PartitionedCircuitDataSplitterBalanced


def load_medium_config_300_no_gnn_variance():
    """Load medium-scale configuration for 300 circuits with balanced test classification, NO GNN variance loss."""
    return {
        'algorithm': 'PPO',
        'total_timesteps': 1000000,  # Extended to 1M timesteps for much longer training
        'n_steps': 256,
        'batch_size': 64,  # Increased from 20 to 64 for faster learning
        'ppo_epochs': 4,
        'gamma': 0.99,  # Increased from 0.95 to 0.99 for more consistent long-term plans
        'gae_lambda': 0.97,  # Decreased from 0.95 to 0.90 for faster reward response and better exploration
        'clip_range': 0.2,
        'value_loss_coef': 0.9,  # Reduced from 0.9 to 0.7 for better critic stability
        'entropy_coef': 0.01,
        'gnn_variance_loss_coef': 0.0,  # DISABLED: GNN variance loss completely disabled
        'max_grad_norm': 0.5,
        'learning_rate': 3e-4,  # Decreased from 1e-3 to 5e-4 for more stable training
        'lr_schedule': 'linear',
        
        # Return normalization
        'normalize_returns': True,
        'return_normalization_epsilon': 1e-8,
        
        # GNN Encoder Configuration
        'gnn_encoder': {
            'type': 'GIN',
            'num_layers': 5,        # Increased from 3 to 5 layers
            'hidden_dim': 256,       # Increased from 128 to 256
            'activation': 'ReLU',
            'pooling': 'attention',  # Changed from 'mean' to 'attention' for attention-based pooling
            'use_global_features': True,  # Enable global features
            'use_edge_features': True,    # Enable edge features (CRUCIAL FIX)
        },
        
        # Enhanced actor-critic heads with updated dimensions
        'actor_head': {
            'input_dim': 257,  # 256 (GNN) + 1 (timestep)
            'layers': [256, 128, 64],  # Enhanced deeper network
            'dropout': 0.1,
            'weight_decay': 0.0,  # L2 regularization for actor network (0.0 = no weight decay)
            'output': 'softmax'
        },
        'critic_head': {
            'input_dim': 257,  # 256 (GNN) + 1 (timestep)
            'layers': [512, 256, 128, 64],  # Enhanced much deeper network
            'dropout': 0.1,
            'weight_decay': 1e-4,  # L2 regularization for critic network
            'output': 'scalar'
        },
        
        # Timestep handling
        'timestep': {
            'use': True,
            'input': 'current_step',
            'normalization': True,
            'inject_location': 'after_gnn_pooling'
        },
        
        # Environment config
        'env': {
            'max_steps': 10,
            'action_space': ['b', 'rw', 'rf', 'rwz', 'rfz'],
            'reward_shaping': True,
            'reward_normalization': True,
            'final_bonus': True
        },
        
        # Balanced dataset with test classification
        'dataset': {
            'type': 'balanced_partitioned',
            'sources': ['MCNC', 'IWLS', 'Synthetic', 'EPFL'],
            'sampling': 'random_circuit_per_episode',
            'augmentation': True,
            'caching': True,
            'train_ratio': 0.695,   # 376/541 circuits from balanced system
            'val_ratio': 0.148,     # 80/541 circuits from balanced system
            'eval_ratio': 0.157,    # 85/541 circuits from balanced system
            'random_seed': 42
        },
        
        # Adjusted validation for balanced splits
        'validation': {
            'val_interval_early': 1600,         # From medium training config
            'val_interval_late': 4000,          # From medium training config
            'val_transition_timesteps': 40000,  # Earlier transition
            'val_episodes': 80,                 # Use all validation circuits from balanced system
            'early_stop_patience': 999999,      # Disabled - set to very high value
            'early_stop_min_delta': 0.005,     # Decreased for less strict criteria
            'early_stop_min_timesteps': 40000, # Increased minimum timesteps before early stopping
        },
        
        # Adjusted evaluation for balanced splits
        'eval': {
            'eval_interval': 4000,              # From medium training config
            'eval_episodes': 85,                # Use all evaluation circuits from balanced system
            'eval_all_circuits': True,          # Use complete evaluation set
            'metrics': ['gate_count', 'gate_reduction_percent', 'best_area', 'avg_episode_reward'],
            'test_set': 'held_out_circuits',
            'save_best_model': True,
            'online_eval': True,
            'eval_patience': 8,                 # Increased tolerance
            'eval_min_delta': 0.05,            # Decreased for less strict criteria
        },
        
        # Enhanced logging
        'logging': {
            'log_interval': 400,                # Log every 400 timesteps
            'tensorboard_log_interval': 100,    # TensorBoard every 100 timesteps
            'log_items': ['reward', 'advantage', 'entropy', 'policy_loss', 'value_loss', 'learning_rate'],
            'log_validation': True,
            'log_evaluation': True,
            'log_data_splits': True
        },
        
        # Checkpoint configuration
        'checkpoint': {
            'save_interval': 8000,              # Save every 8k timesteps
            'keep_best_n': 3,
            'save_optimizer': True
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
    
    return observations, actions, rewards, values, log_probs, episode_info


def collect_batch_with_balanced_splits(agent, env, data_splitter, batch_size, config):
    """Collect a batch of episodes using the balanced data splitter."""
    all_observations = []
    all_actions = []
    all_rewards = []
    all_values = []
    all_log_probs = []
    episode_infos = []
    
    # Sample circuits from training set using balanced splitter
    train_circuits = data_splitter.get_training_circuits()
    circuits = random.choices(train_circuits, k=batch_size)
    
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
    
    # Compute advantages and returns using config parameters
    gamma = config.get('gamma', 0.95)
    gae_lambda = config.get('gae_lambda', 0.95)
    advantages, returns = compute_gae(all_rewards, all_values, gamma, gae_lambda)
    advantages = normalize_advantages(advantages)
    
    # Normalize returns to reduce value loss variance
    if len(returns) > 1:
        returns_array = np.array(returns)
        returns_mean = returns_array.mean()
        returns_std = returns_array.std()
        if returns_std > 1e-8:
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


def validate_agent_with_balanced_splits(agent, env, val_circuits, num_episodes=80):
    """Validate agent performance using balanced validation set with comprehensive metrics."""
    
    # Use the configured number of validation episodes for better stability
    actual_episodes = min(num_episodes, len(val_circuits))
    print(f"🔍 Starting validation with {actual_episodes} episodes from {len(val_circuits)} validation circuits...")
    
    # Set agent to evaluation mode
    agent.eval()
    
    # Sample from validation circuits
    val_sample = random.choices(val_circuits, k=actual_episodes)
    
    # Track circuit coverage for better monitoring
    circuit_names = [metadata['name'] for _, metadata in val_sample]
    print(f"  Validation circuits: {', '.join(circuit_names)}")
    
    results = []
    total_reward = 0.0
    total_area_reduction = 0.0
    successful_episodes = 0
    
    for circuit_path, metadata in val_sample:
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
        print("⚠️  No successful episodes in validation!")
        # Return agent to training mode
        agent.train()
        return {
            'avg_reward': 0.0,
            'avg_area_reduction_percent': 0.0,
            'success_rate': 0.0,
            'results': results
        }
    
    # Calculate averages
    avg_reward = total_reward / successful_episodes
    avg_area_reduction_percent = total_area_reduction / successful_episodes
    success_rate = successful_episodes / len(val_sample)
    
    print(f"📊 Validation Results:")
    print(f"  Average Reward: {avg_reward:.3f}")
    print(f"  Average Area Reduction: {avg_area_reduction_percent:.2f}%")
    print(f"  Success Rate: {success_rate:.2f}")
    print(f"  Successful Episodes: {successful_episodes}/{len(val_sample)}")
    
    # Return agent to training mode
    agent.train()
    
    return {
        'avg_reward': avg_reward,
        'avg_area_reduction_percent': avg_area_reduction_percent,
        'success_rate': success_rate,
        'results': results
    }


def evaluate_agent_with_balanced_splits(agent, env, eval_circuits, num_episodes=85):
    """Evaluate agent performance using balanced evaluation set with comprehensive metrics."""
    
    # Use all evaluation circuits for complete coverage
    use_all_circuits = True
    
    if use_all_circuits:
        eval_sample = eval_circuits  # Use ALL evaluation circuits
        print(f"🧪 Starting evaluation with ALL {len(eval_circuits)} evaluation circuits...")
    else:
        # Fallback to sampling (original behavior)
        eval_sample = random.choices(eval_circuits, k=min(num_episodes, len(eval_circuits)))
        print(f"🧪 Starting evaluation with {len(eval_sample)} episodes...")
    
    # Set agent to evaluation mode
    agent.eval()
    
    results = []
    total_reward = 0.0
    total_area_reduction = 0.0
    successful_episodes = 0
    
    for circuit_path, metadata in eval_sample:
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
        # Return agent to training mode
        agent.train()
        return {
            'avg_reward': 0.0,
            'avg_area_reduction_percent': 0.0,
            'success_rate': 0.0,
            'results': results
        }
    
    # Calculate averages
    avg_reward = total_reward / successful_episodes
    avg_area_reduction_percent = total_area_reduction / successful_episodes
    success_rate = successful_episodes / len(eval_sample)
    
    print(f"📊 Evaluation Results:")
    print(f"  Average Reward: {avg_reward:.3f}")
    print(f"  Average Area Reduction: {avg_area_reduction_percent:.2f}%")
    print(f"  Success Rate: {success_rate:.2f}")
    print(f"  Successful Episodes: {successful_episodes}/{len(eval_sample)}")
    
    # Return agent to training mode
    agent.train()
    
    return {
        'avg_reward': avg_reward,
        'avg_area_reduction_percent': avg_area_reduction_percent,
        'success_rate': success_rate,
        'results': results
    }


def load_checkpoint(agent, checkpoint_path: str, logger):
    """Load agent state from checkpoint."""
    try:
        agent.load_model(checkpoint_path)
        logger.info(f"✅ Successfully loaded checkpoint from: {checkpoint_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load checkpoint from {checkpoint_path}: {e}")
        return False 


def medium_train_300_no_gnn_variance(checkpoint_path: str = None):
    """Run medium training session with 300 circuits using balanced test classification, NO GNN variance loss."""
    print("="*80)
    print("MEDIUM TRAINING WITH BALANCED TEST CLASSIFICATION - NO GNN VARIANCE LOSS (~4 hours)")
    print("="*80)
    
    start_time = time.time()
    
    # Load configuration
    config = load_medium_config_300_no_gnn_variance()
    
    # Setup logging
    logger = setup_logger('medium_training_300_no_gnn_variance', level=logging.INFO)
    
    # Create balanced dataset using pre-partitioned circuit files
    try:
        data_splitter = PartitionedCircuitDataSplitterBalanced()
        logger.info("Loaded balanced partitioned circuit dataset")
        
        # Get split information from balanced dataset
        split_info = data_splitter.get_split_info()
        logger.info(f"Balanced data splits - Total: {split_info['total_circuits']}, "
                   f"Train: {split_info['train_circuits']} ({split_info['train_ratio']:.1%}), "
                   f"Val: {split_info['val_circuits']} ({split_info['val_ratio']:.1%}), "
                   f"Eval: {split_info['eval_circuits']} ({split_info['eval_ratio']:.1%})")
        
        # Log circuit statistics
        stats = data_splitter.get_circuit_stats()
        logger.info("Circuit distribution by suite:")
        for split_name, split_stats in stats.items():
            logger.info(f"  {split_name.capitalize()}: {split_stats['count']} circuits")
            logger.info(f"    Complexity: {split_stats['complexity']}")
            for suite, count in split_stats['suites'].items():
                logger.info(f"    {suite}: {count} circuits")
        
    except Exception as e:
        logger.error(f"Error loading balanced dataset: {e}")
        return False
    
    # Create environment
    env_config = config.get('env', {})
    env = SynthesisEnvironment(
        max_steps=env_config.get('max_steps', 10),
        action_space=env_config.get('action_space', ['b', 'rw', 'rf', 'rwz', 'rfz']),
        reward_shaping=env_config.get('reward_shaping', True),
        reward_normalization=env_config.get('reward_normalization', True),
        final_bonus=env_config.get('final_bonus', True),
        cleanup_logs=False
    )
    
    # Create agent
    agent = PPOSynthesisAgent(config)
    
    # Load checkpoint if provided
    if checkpoint_path:
        if not load_checkpoint(agent, checkpoint_path, logger):
            logger.error("Failed to load checkpoint, starting from scratch")
        else:
            logger.info(f"Resuming training from checkpoint: {checkpoint_path}")
    
    logger.info(f"Agent created with {sum(p.numel() for p in agent.parameters()):,} parameters")
    logger.info(f"Device: {agent.device}")
    logger.info("⚠️  GNN VARIANCE LOSS DISABLED - Training without GNN variance regularization")
    
    # Setup metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Training parameters
    total_timesteps = config.get('total_timesteps', 1000000)
    batch_size = config.get('batch_size', 64)
    
    # Validation configuration
    val_config = config.get('validation', {})
    val_interval_early = val_config.get('val_interval_early', 1600)
    val_interval_late = val_config.get('val_interval_late', 4000)
    val_transition_timesteps = val_config.get('val_transition_timesteps', 40000)
    val_episodes = val_config.get('val_episodes', 80)

    # Early stopping configuration
    early_stop_patience = val_config.get('early_stop_patience', 999999)
    early_stop_min_delta = val_config.get('early_stop_min_delta', 0.005)
    early_stop_min_timesteps = val_config.get('early_stop_min_timesteps', 40000)

    # Evaluation configuration
    eval_config = config.get('eval', {})
    eval_interval = eval_config.get('eval_interval', 4000)
    eval_episodes = eval_config.get('eval_episodes', 85)
    eval_all_circuits = eval_config.get('eval_all_circuits', True)
    eval_patience = eval_config.get('eval_patience', 8)
    eval_min_delta = eval_config.get('eval_min_delta', 0.05)
    
    # Logging configuration
    log_config = config.get('logging', {})
    log_interval = log_config.get('log_interval', 400)
    tb_log_interval = log_config.get('tensorboard_log_interval', 100)
    
    # Checkpoint configuration
    checkpoint_config = config.get('checkpoint', {})
    save_interval = checkpoint_config.get('save_interval', 8000)
    
    # Create output directories
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    model_dir = output_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    
    # Setup comprehensive TensorBoard monitor
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_dir = output_dir / 'tensorboard_logs'
    tb_dir.mkdir(exist_ok=True)
    tb_log_dir = tb_dir / f'medium_train_300_no_gnn_variance_{current_time}'
    monitor = TensorBoardMonitor(log_dir=str(tb_log_dir))
    
    logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")
    print(f"📊 TensorBoard logs: {tb_log_dir}")
    
    # Log hyperparameters and data splits
    hparams = {
        'total_timesteps': total_timesteps,
        'batch_size': batch_size,
        'learning_rate': config.get('learning_rate', 5e-4),
        'total_circuits': split_info['total_circuits'],
        'train_circuits': split_info['train_circuits'],
        'val_circuits': split_info['val_circuits'],
        'eval_circuits': split_info['eval_circuits'],
        'val_episodes': val_episodes,
        'eval_episodes': eval_episodes,
        'val_patience': early_stop_patience,
        'eval_patience': eval_patience,
        'gnn_variance_loss': False,  # Mark that GNN variance loss is disabled
        'gnn_variance_loss_coef': 0.0  # Explicitly set to 0.0 to disable
    }
    monitor.log_hyperparameters(hparams, {'training_started': 1.0})
    
    # Log data split information to TensorBoard
    monitor.writer.add_text('data_splits/overview', 
                           f"Total: {split_info['total_circuits']}, "
                           f"Train: {split_info['train_circuits']} ({split_info['train_ratio']:.1%}), "
                           f"Val: {split_info['val_circuits']} ({split_info['val_ratio']:.1%}), "
                           f"Eval: {split_info['eval_circuits']} ({split_info['eval_ratio']:.1%})")
    
    logger.info("Starting medium training with balanced test classification (NO GNN VARIANCE LOSS)...")
    logger.info(f"Target timesteps: {total_timesteps}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Validation: {val_episodes} episodes every {val_interval_early}->{val_interval_late} timesteps (transition at {val_transition_timesteps})")
    logger.info(f"Evaluation: {eval_episodes} episodes every {eval_interval} timesteps")
    logger.info(f"Early stopping: patience={early_stop_patience}, min_delta={early_stop_min_delta}%, min_timesteps={early_stop_min_timesteps}")
    logger.info(f"⚠️  EARLY STOPPING DISABLED: Training will run for full {total_timesteps:,} timesteps")
    logger.info(f"⚠️  GNN VARIANCE LOSS DISABLED: Training without GNN variance regularization")
    
    # Training loop variables
    timesteps = 0
    episode_count = 0
    
    # Validation tracking
    best_val_performance = 0.0
    val_patience_counter = 0
    val_history = []
    last_validation_timestep = 0
    validation_count = 0
    
    # Evaluation tracking
    best_eval_performance = 0.0
    eval_patience_counter = 0
    eval_history = []
    last_evaluation_timestep = 0
    evaluation_count = 0
    
    # Get balanced circuit sets
    train_circuits = data_splitter.get_training_circuits()
    val_circuits = data_splitter.get_validation_circuits()
    eval_circuits = data_splitter.get_test_circuits()
    
    logger.info(f"Using balanced splits: {len(train_circuits)} train, {len(val_circuits)} val, {len(eval_circuits)} eval")
    
    if eval_all_circuits:
        logger.info(f"Using complete evaluation set: ALL {len(eval_circuits)} circuits per evaluation")
    else:
        logger.info(f"Using sampled evaluation: {eval_episodes} episodes per evaluation")
    
    try:
        with tqdm(total=total_timesteps, desc="Training Progress (No GNN Variance)") as pbar:
            while timesteps < total_timesteps:
                batch_start_time = time.time()
                
                # Collect batch of episodes using balanced training set
                batch_data = collect_batch_with_balanced_splits(agent, env, data_splitter, batch_size, config)
                
                # Update agent
                training_stats = agent.update(batch_data)
                
                # Update metrics
                batch_timesteps = len(batch_data['observations'])
                timesteps += batch_timesteps
                episode_count += len(batch_data['episode_infos'])
                
                # Update progress bar
                pbar.update(batch_timesteps)
                
                # Log episode statistics with comprehensive metrics tracking
                batch_time = time.time() - batch_start_time
                
                # Update metrics tracker
                for episode_info in batch_data['episode_infos']:
                    metrics_tracker.update_episode(episode_info)
                
                if batch_data['episode_infos']:
                    avg_reward = np.mean([info['total_reward'] for info in batch_data['episode_infos']])
                    avg_area_reduction = np.mean([info.get('area_reduction_percent', 0) for info in batch_data['episode_infos']])
                    
                    # Log comprehensive training metrics using TensorBoard monitor
                    monitor.log_training_metrics(timesteps, training_stats)
                    
                    # Log GNN metrics if available (but note that variance loss is disabled)
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
                    if timesteps % 2000 == 0:
                        monitor.log_summary_statistics(timesteps)
                
                # Validation
                current_val_interval = val_interval_early if timesteps < val_transition_timesteps else val_interval_late
                if timesteps - last_validation_timestep >= current_val_interval:
                    validation_count += 1
                    logger.info(f"🔍 Starting validation #{validation_count} at timestep {timesteps} (interval: {current_val_interval})")
                    
                    val_start_time = time.time()
                    val_results = validate_agent_with_balanced_splits(
                        agent, env, val_circuits, val_episodes
                    )
                    val_time = time.time() - val_start_time
                    
                    val_reward = val_results['avg_reward']
                    val_area_reduction = val_results['avg_area_reduction_percent']
                    val_success_rate = val_results['success_rate']
                    
                    # Log validation results to TensorBoard
                    logger.info(f"📊 Logging validation #{validation_count} results to TensorBoard at step {timesteps}")
                    monitor.log_validation_metrics(timesteps, {
                        'avg_reward': val_reward,
                        'avg_area_reduction_percent': val_area_reduction,
                        'success_rate': val_success_rate,
                        'validation_time': val_time
                    })
                    
                    val_history.append(val_area_reduction)
                    
                    # Early stopping check
                    if val_area_reduction > best_val_performance + early_stop_min_delta:
                        best_val_performance = val_area_reduction
                        val_patience_counter = 0
                        
                        # Save best model
                        model_path = model_dir / f'best_val_model_no_gnn_variance_{timesteps}.pth'
                        agent.save_model(str(model_path))
                        logger.info(f"✅ New best validation performance: {val_area_reduction:.2f}% - saved model")
                    else:
                        val_patience_counter += 1
                    
                    logger.info(f"📈 Validation #{validation_count} [{timesteps:,}] - Reward: {val_reward:.3f}, "
                               f"Area Reduction: {val_area_reduction:.2f}%, "
                               f"Success Rate: {val_success_rate:.2f}, "
                               f"Patience: {val_patience_counter}/{early_stop_patience}")
                    
                    last_validation_timestep = timesteps
                    
                    # Check early stopping (with minimum timesteps requirement) - DISABLED
                    if val_patience_counter >= early_stop_patience and timesteps >= early_stop_min_timesteps:
                        logger.info(f"⏹️  Early stopping triggered after {val_patience_counter} validations without improvement (timesteps: {timesteps:,})")
                        break
                    elif val_patience_counter >= early_stop_patience and timesteps < early_stop_min_timesteps:
                        logger.info(f"⏳ Early stopping criteria met ({val_patience_counter}/{early_stop_patience}) but minimum timesteps not reached ({timesteps:,}/{early_stop_min_timesteps:,}). Continuing training...")
                        # Reset patience counter to give model more time
                        val_patience_counter = early_stop_patience - 1
                
                # Evaluation
                if timesteps - last_evaluation_timestep >= eval_interval:
                    evaluation_count += 1
                    logger.info(f"🧪 Starting evaluation #{evaluation_count} at timestep {timesteps} (interval: {eval_interval})")
                    
                    eval_start_time = time.time()
                    eval_results = evaluate_agent_with_balanced_splits(
                        agent, env, eval_circuits, eval_episodes
                    )
                    eval_time = time.time() - eval_start_time
                    
                    eval_reward = eval_results['avg_reward']
                    eval_area_reduction = eval_results['avg_area_reduction_percent']
                    eval_success_rate = eval_results['success_rate']
                    
                    # Log evaluation results to TensorBoard
                    logger.info(f"📊 Logging evaluation #{evaluation_count} results to TensorBoard at step {timesteps}")
                    monitor.log_evaluation_metrics(timesteps, {
                        'avg_reward': eval_reward,
                        'avg_area_reduction_percent': eval_area_reduction,
                        'success_rate': eval_success_rate,
                        'evaluation_time': eval_time
                    })
                    
                    eval_history.append(eval_area_reduction)
                    
                    # Track best evaluation performance
                    if eval_area_reduction > best_eval_performance + eval_min_delta:
                        best_eval_performance = eval_area_reduction
                        eval_patience_counter = 0
                        
                        # Save best evaluation model
                        model_path = model_dir / f'best_eval_model_no_gnn_variance_{timesteps}.pth'
                        agent.save_model(str(model_path))
                        logger.info(f"✅ New best evaluation performance: {eval_area_reduction:.2f}% - saved model")
                    else:
                        eval_patience_counter += 1
                    
                    logger.info(f"📈 Evaluation #{evaluation_count} [{timesteps:,}] - Reward: {eval_reward:.3f}, "
                               f"Area Reduction: {eval_area_reduction:.2f}%, "
                               f"Success Rate: {eval_success_rate:.2f}, "
                               f"Patience: {eval_patience_counter}/{eval_patience}")
                    
                    last_evaluation_timestep = timesteps
                
                # Checkpoint saving
                if timesteps % save_interval == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_path = model_dir / f'checkpoint_no_gnn_variance_{timesteps}_{timestamp}.pth'
                    agent.save_model(str(model_path))
                    logger.info(f"Saved checkpoint at timestep {timesteps}")
                
                # Periodic logging
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
                    remaining_time = (elapsed_time / timesteps) * (total_timesteps - timesteps)
                    logger.info(f"Progress [{timesteps:,}/{total_timesteps:,}] "
                               f"({timesteps/total_timesteps*100:.1f}%) - "
                               f"Episodes: {episode_count}, "
                               f"Elapsed: {elapsed_time/3600:.1f}h, "
                               f"Remaining: {remaining_time/3600:.1f}h")
        
        # Final evaluation
        logger.info("Running final evaluation...")
        final_eval_results = evaluate_agent_with_balanced_splits(
            agent, env, eval_circuits, eval_episodes * 2  # More episodes for final evaluation
        )
        
        final_eval_reward = final_eval_results['avg_reward']
        final_eval_area_reduction = final_eval_results['avg_area_reduction_percent']
        final_eval_success_rate = final_eval_results['success_rate']
        
        # Log final results
        monitor.log_evaluation_metrics(timesteps, {
            'avg_reward': final_eval_reward,
            'avg_area_reduction_percent': final_eval_area_reduction,
            'success_rate': final_eval_success_rate
        })
        
        # Save final model
        final_model_path = model_dir / f'final_model_no_gnn_variance_{timesteps}.pth'
        agent.save_model(str(final_model_path))
        
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
                'training_time_hours': (time.time() - start_time) / 3600,
                'data_split_train_circuits': split_info['train_circuits'],
                'data_split_val_circuits': split_info['val_circuits'],
                'data_split_eval_circuits': split_info['eval_circuits'],
                'gnn_variance_loss_disabled': True,
                'gnn_variance_loss_coef': 0.0
            }
            monitor.log_hyperparameters(hparams, final_metrics)
        
        # Training summary
        training_time = time.time() - start_time
        logger.info("="*80)
        logger.info("TRAINING COMPLETED (NO GNN VARIANCE LOSS)")
        logger.info("="*80)
        logger.info(f"Total training time: {training_time/3600:.2f} hours")
        logger.info(f"Total timesteps: {timesteps:,}")
        logger.info(f"Total episodes: {episode_count:,}")
        logger.info(f"Total validations: {validation_count}")
        logger.info(f"Total evaluations: {evaluation_count}")
        logger.info(f"Best validation performance: {best_val_performance:.2f}%")
        logger.info(f"Best evaluation performance: {best_eval_performance:.2f}%")
        logger.info(f"Final evaluation performance: {final_eval_area_reduction:.2f}%")
        logger.info(f"Balanced data splits used: {split_info['train_circuits']} train, "
                   f"{split_info['val_circuits']} val, {split_info['eval_circuits']} eval")
        logger.info(f"TensorBoard logs: {tb_log_dir}")
        logger.info("⚠️  GNN VARIANCE LOSS WAS DISABLED FOR THIS TRAINING RUN")
        logger.info("="*80)
        
        # Print validation summary
        if val_history:
            logger.info(f"Validation Summary:")
            logger.info(f"  Best validation performance: {best_val_performance:.2f}%")
            logger.info(f"  Final validation performance: {val_history[-1]:.2f}%")
            logger.info(f"  Total validations: {validation_count}")
        
        # Print evaluation summary
        if eval_history:
            logger.info(f"Evaluation Summary:")
            logger.info(f"  Best evaluation performance: {best_eval_performance:.2f}%")
            logger.info(f"  Final evaluation performance: {eval_history[-1]:.2f}%")
            logger.info(f"  Total evaluations: {evaluation_count}")
        
        # Close TensorBoard monitor
        monitor.close()
        
        return True
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Log final summary statistics
        monitor.log_summary_statistics(timesteps)
        monitor.close()
        return False
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        # Log final summary statistics
        monitor.log_summary_statistics(timesteps)
        monitor.close()
        return False


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == '--help':
        print("Medium Training Script with Balanced Test Classification (No GNN Variance Loss)")
        print("Usage: python medium_train_300_no_gnn_variance.py [checkpoint_path]")
        print()
        print("This script combines the medium training approach from medium_train_300_with_balanced_splits.py")
        print("but removes the GNN variance loss component to test its impact on training performance.")
        print()
        print("Arguments:")
        print("  checkpoint_path: Optional path to checkpoint file to resume training")
        print()
        print("Features:")
        print("  - Medium training configuration (1M timesteps, enhanced GNN)")
        print("  - Balanced circuit partitioning across complexity levels")
        print("  - Even distribution across MCNC, IWLS, Synthetic, EPFL suites")
        print("  - Comprehensive validation and evaluation")
        print("  - TensorBoard monitoring")
        print("  - GNN VARIANCE LOSS DISABLED")
        return
    
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else None
    success = medium_train_300_no_gnn_variance(checkpoint_path)
    if success:
        print("✅ Training completed successfully!")
        print("📊 Check TensorBoard logs for detailed metrics")
        print("🔧 Models saved in outputs/models/")
        print("⚠️  This training run had GNN variance loss disabled")
    else:
        print("❌ Training failed or was interrupted")
        sys.exit(1)


if __name__ == "__main__":
    main() 