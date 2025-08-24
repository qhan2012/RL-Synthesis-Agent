#!/usr/bin/env python3
"""
Tiny Training with TensorBoard Dashboard

This script runs tiny training with real-time TensorBoard monitoring.
"""

import sys
import os
import yaml
import time
import logging
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.append('.')

# Import TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from tensorboard import SummaryWriter
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        print("Warning: TensorBoard not available, using dummy logger")
        TENSORBOARD_AVAILABLE = False
        SummaryWriter = None

from models.ppo_agent import PPOSynthesisAgent, compute_gae, normalize_advantages
from env.synthesis_env import SynthesisEnvironment
from data.dataset import CircuitDataset
from utils.logger import setup_logger
from utils.metrics import MetricsTracker


class TensorBoardLogger:
    """Simple TensorBoard logger for training metrics."""
    
    def __init__(self, log_dir="outputs/tensorboard_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / f"tiny_train_{timestamp}"
        
        if TENSORBOARD_AVAILABLE and SummaryWriter is not None:
            self.writer = SummaryWriter(str(self.run_dir))
            print(f"📊 TensorBoard logging to: {self.run_dir}")
            print(f"🌐 View dashboard at: http://localhost:6006")
        else:
            self.writer = None
            print(f"📊 TensorBoard not available, metrics will be logged to console only")
        
        self.step = 0
    
    def log_episode(self, episode_info, episode_num):
        """Log episode metrics."""
        if not self.writer:
            return
            
        if 'area_reduction_percent' in episode_info:
            self.writer.add_scalar('Episode/Area_Reduction_%', 
                                 episode_info['area_reduction_percent'], episode_num)
        
        if 'total_reward' in episode_info:
            self.writer.add_scalar('Episode/Total_Reward', 
                                 episode_info['total_reward'], episode_num)
        
        if 'num_steps' in episode_info:
            self.writer.add_scalar('Episode/Steps_Taken', 
                                 episode_info['num_steps'], episode_num)
        
        if 'final_area' in episode_info:
            self.writer.add_scalar('Episode/Final_Area', 
                                 episode_info['final_area'], episode_num)
        
        if 'initial_area' in episode_info:
            self.writer.add_scalar('Episode/Initial_Area', 
                                 episode_info['initial_area'], episode_num)
    
    def log_training(self, losses, batch_num):
        """Log training losses."""
        if not self.writer:
            return
            
        if 'policy_loss' in losses:
            self.writer.add_scalar('Training/Policy_Loss', 
                                 losses['policy_loss'], batch_num)
        
        if 'value_loss' in losses:
            self.writer.add_scalar('Training/Value_Loss', 
                                 losses['value_loss'], batch_num)
        
        if 'entropy_loss' in losses:
            self.writer.add_scalar('Training/Entropy_Loss', 
                                 losses['entropy_loss'], batch_num)
        
        if 'total_loss' in losses:
            self.writer.add_scalar('Training/Total_Loss', 
                                 losses['total_loss'], batch_num)
    
    def log_batch_summary(self, batch_info, batch_num):
        """Log batch summary metrics."""
        if not self.writer:
            return
            
        if 'avg_area_reduction' in batch_info:
            self.writer.add_scalar('Batch/Avg_Area_Reduction_%', 
                                 batch_info['avg_area_reduction'], batch_num)
        
        if 'avg_reward' in batch_info:
            self.writer.add_scalar('Batch/Avg_Reward', 
                                 batch_info['avg_reward'], batch_num)
        
        if 'best_area_reduction' in batch_info:
            self.writer.add_scalar('Batch/Best_Area_Reduction_%', 
                                 batch_info['best_area_reduction'], batch_num)
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.writer:
            self.writer.close()


def load_config():
    """Load minimal configuration for testing."""
    return {
        'gnn_encoder': {
            'type': 'GIN',
            'hidden_dim': 64,
            'num_layers': 2,
            'pooling': 'mean',
            'use_global_features': True
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
            'max_steps': 10,
            'action_space': ['b', 'rw', 'rf', 'rwz', 'rfz'],
            'reward_shaping': True,
            'reward_normalization': False,
            'final_bonus': False
        },
        'dataset': {
            'data_root': 'testcase',
            'sources': ['ISCAS85', 'IWLS', 'Synthetic']
        }
    }


def collect_episode(agent, env, circuit_path, tb_logger, episode_num):
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
    
    # Log to TensorBoard
    tb_logger.log_episode(episode_info, episode_num)
    
    return observations, actions, rewards, values, log_probs, episode_info


def collect_batch(agent, env, dataset, tb_logger, batch_num, batch_size=2):
    """Collect a small batch of episodes for training."""
    all_observations = []
    all_actions = []
    all_rewards = []
    all_values = []
    all_log_probs = []
    episode_infos = []
    
    # Sample circuits for this batch
    circuits = dataset.sample_circuits(batch_size)
    
    episode_num = batch_num * batch_size
    
    for i, (circuit_path, metadata) in enumerate(circuits):
        print(f"Processing circuit: {metadata['name']}")
        observations, actions, rewards, values, log_probs, episode_info = collect_episode(
            agent, env, circuit_path, tb_logger, episode_num + i
        )
        
        # Store episode data
        all_observations.extend(observations[:-1])
        all_actions.extend(actions)
        all_rewards.extend(rewards)
        all_values.extend(values)
        all_log_probs.extend(log_probs)
        episode_infos.append(episode_info)
    
    # Compute batch summary
    area_reductions = [info.get('area_reduction_percentage', 0) for info in episode_infos]
    total_rewards = [info.get('total_reward', 0) for info in episode_infos]
    
    batch_info = {
        'avg_area_reduction': np.mean(area_reductions),
        'avg_reward': np.mean(total_rewards),
        'best_area_reduction': np.max(area_reductions)
    }
    
    # Log batch summary
    tb_logger.log_batch_summary(batch_info, batch_num)
    
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


def tiny_train_with_dashboard():
    """Run tiny training with TensorBoard dashboard."""
    print("="*60)
    print("TINY TRAINING WITH TENSORBOARD DASHBOARD")
    print("="*60)
    
    # Initialize TensorBoard logger
    tb_logger = TensorBoardLogger()
    
    # Load configuration
    config = load_config()
    
    # Setup logging
    logger = setup_logger('tiny_training_dashboard', level=logging.INFO)
    
    # Create dataset
    try:
        dataset = CircuitDataset(
            data_root=config['dataset'].get('data_root', 'testcase'),
            sources=config['dataset'].get('sources', ['ISCAS85', 'IWLS', 'Synthetic'])
        )
        
        logger.info(f"Loaded {len(dataset)} circuits from dataset")
        
        if len(dataset) == 0:
            print("No circuits found. Cannot proceed with training.")
            tb_logger.close()
            return
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        tb_logger.close()
        return
    
    # Create environment
    env_config = config.get('env', {})
    env = SynthesisEnvironment(
        max_steps=env_config.get('max_steps', 10),
        action_space=env_config.get('action_space', ['b', 'rw', 'rf', 'rwz', 'rfz']),
        reward_shaping=env_config.get('reward_shaping', True),
        reward_normalization=env_config.get('reward_normalization', False),
        final_bonus=env_config.get('final_bonus', False),
        cleanup_logs=False
    )
    
    # Create agent
    agent = PPOSynthesisAgent(config)
    
    # Create metrics tracker
    metrics = MetricsTracker()
    
    # Training parameters
    num_batches = 4
    batch_size = 2
    
    print(f"\n🚀 Starting training with {num_batches} batches, {batch_size} episodes each")
    print(f"📊 TensorBoard dashboard: http://localhost:6006")
    print(f"💡 Run 'tensorboard --logdir {tb_logger.run_dir.parent}' to start dashboard")
    
    try:
        for batch_num in range(num_batches):
            print(f"\n--- Batch {batch_num + 1}/{num_batches} ---")
            
            # Collect batch
            batch_data = collect_batch(agent, env, dataset, tb_logger, batch_num, batch_size)
            
            # Train on batch
            losses = agent.update(batch_data)
            
            # Log training losses
            tb_logger.log_training(losses, batch_num)
            
            # Update metrics
            for episode_info in batch_data['episode_infos']:
                metrics.update_episode(episode_info)
            
            # Print batch summary
            area_reductions = [info.get('area_reduction_percentage', 0) for info in batch_data['episode_infos']]
            avg_area_reduction = np.mean(area_reductions)
            best_area_reduction = np.max(area_reductions)
            
            print(f"  Avg Area Reduction: {avg_area_reduction:.2f}%")
            print(f"  Best Area Reduction: {best_area_reduction:.2f}%")
            print(f"  Policy Loss: {losses.get('policy_loss', 0):.4f}")
            print(f"  Value Loss: {losses.get('value_loss', 0):.4f}")
        
        # Final summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        
        final_stats = metrics.get_average_stats()
        print(f"📈 Final Results:")
        print(f"  Average Area Reduction: {final_stats.get('avg_area_reduction_percent', 0):.2f}%")
        print(f"  Best Area Reduction: {final_stats.get('max_area_reduction_percent', 0):.2f}%")
        print(f"  Total Episodes: {final_stats.get('num_episodes', 0)}")
        
        # Save model
        model_path = "outputs/models/tiny_dashboard_model.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        agent.save_model(model_path)
        print(f"💾 Model saved to: {model_path}")
        
        print(f"\n🌐 TensorBoard Dashboard: http://localhost:6006")
        print(f"📊 Log directory: {tb_logger.run_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logger.error(f"Training failed: {e}")
    finally:
        tb_logger.close()
        print("📊 TensorBoard logger closed")


def main():
    """Main entry point."""
    tiny_train_with_dashboard()


if __name__ == "__main__":
    main() 