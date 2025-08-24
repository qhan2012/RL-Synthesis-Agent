"""
PPO Agent for RL Synthesis

This module implements the complete PPO agent with GNN encoder and
actor-critic networks for logic synthesis optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import deque
import os
import sys

# Add the project root to the path to import the compatibility layer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from aag2gnn_compatibility_fixed import AAG2GNNCompatibilityLayer
    COMPATIBILITY_AVAILABLE = True
except ImportError:
    print("Warning: AAG2GNN compatibility layer not available")
    COMPATIBILITY_AVAILABLE = False

from .gnn_encoder import create_gnn_encoder
from .actor_head import create_actor_head
from .critic_head import create_critic_head


class PPOSynthesisAgent(nn.Module):
    """
    PPO agent for logic synthesis optimization.
    
    Combines GNN encoder with actor-critic networks and implements
    PPO training logic.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize compatibility layer if available
        if COMPATIBILITY_AVAILABLE:
            self.compatibility_layer = AAG2GNNCompatibilityLayer()
        else:
            self.compatibility_layer = None
        
        # Create networks
        self.gnn_encoder = create_gnn_encoder(config['gnn_encoder'])
        self.actor_head = create_actor_head(config['actor_head'])
        self.critic_head = create_critic_head(config['critic_head'])
        
        # Move to device
        self.gnn_encoder.to(self.device)
        self.actor_head.to(self.device)
        self.critic_head.to(self.device)
        
        # Optimizers with separate weight decay for critic
        learning_rate = config.get('learning_rate', 3e-4)
        if isinstance(learning_rate, str):
            learning_rate = float(learning_rate)
        
        # Get weight decay from configs
        critic_config = config.get('critic_head', {})
        actor_config = config.get('actor_head', {})
        critic_weight_decay = critic_config.get('weight_decay', 0.0)
        actor_weight_decay = actor_config.get('weight_decay', 0.0)
        
        # Log weight decay configuration
        if critic_weight_decay > 0:
            print(f"🔧 Critic network L2 weight decay: {critic_weight_decay}")
        else:
            print("🔧 Critic network: No weight decay")
            
        if actor_weight_decay > 0:
            print(f"🔧 Actor network L2 weight decay: {actor_weight_decay}")
        else:
            print("🔧 Actor network: No weight decay")
        
        # Optimizer for GNN encoder and actor head with configurable weight decay
        self.actor_optimizer = optim.Adam(
            list(self.gnn_encoder.parameters()) +
            list(self.actor_head.parameters()),
            lr=learning_rate,
            weight_decay=actor_weight_decay  # Configurable weight decay for actor
        )
        
        # Optimizer for critic head with weight decay
        self.critic_optimizer = optim.Adam(
            list(self.critic_head.parameters()),
            lr=learning_rate,
            weight_decay=critic_weight_decay  # L2 regularization for critic
        )
        
        # Keep original optimizer for backward compatibility
        self.optimizer = self.actor_optimizer
        
        # PPO parameters
        self.clip_range = config.get('clip_range', 0.1)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # GNN variance loss parameters for preventing embedding collapse
        # Set to 0.0 if not provided to disable GNN variance loss
        self.gnn_variance_loss_coef = config.get('gnn_variance_loss_coef', 0.0)
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=config.get('n_steps', 256))
        
        # Training statistics
        self.training_stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'clip_fraction': 0.0,
            'gnn_variance_loss': 0.0
        }
    
    def forward(self, observations: List[Data]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the agent.
        
        Args:
            observations (List[Data]): List of graph observations
            
        Returns:
            tuple: (action_logits, value_estimates, gnn_embeddings)
        """
        # Ensure all observations are on the same device
        observations = [obs.to(self.device) for obs in observations]
        
        # Always use standard PyTorch Geometric batching for proper GNN embeddings
        # (Compatibility layer bypasses GNN encoder causing zero variance loss)
        # Standard PyTorch Geometric batching
        try:
            batch = Batch.from_data_list(observations)
            batch = batch.to(self.device)
            
            # Get GNN embeddings using the actual GNN encoder
            gnn_embeddings = self.gnn_encoder(batch)
        except Exception as e:
            print(f"Warning: Standard batching failed: {e}")
            # Fallback to single observation processing
            if len(observations) == 1:
                gnn_embeddings = self.gnn_encoder(observations[0].to(self.device))
            else:
                raise e
        
        # Add timestep information - handle batch size properly
        num_observations = len(observations)
        if num_observations == 1:
            # Single observation
            timesteps = observations[0].timestep.to(self.device)
            if timesteps.dim() == 0:
                timesteps = timesteps.unsqueeze(0)
            if timesteps.dim() == 1:
                timesteps = timesteps.unsqueeze(0)  # Shape: [1, 1]
        else:
            # Multiple observations - stack timesteps
            timesteps = torch.stack([obs.timestep for obs in observations]).to(self.device)
            if timesteps.dim() == 1:
                timesteps = timesteps.unsqueeze(1)  # Shape: [batch_size, 1]
        
        # Ensure gnn_embeddings has the right batch size
        if gnn_embeddings.size(0) != num_observations:
            # Repeat the embeddings for each observation
            gnn_embeddings = gnn_embeddings.repeat(num_observations, 1)
        
        # Combine features
        combined_features = torch.cat([gnn_embeddings, timesteps], dim=1)  # Shape: [batch_size, 257]
        
        # Get actor and critic outputs
        action_logits = self.actor_head(combined_features)
        value_estimates = self.critic_head(combined_features)
        
        return action_logits, value_estimates, gnn_embeddings
    
    def get_action(self, observation: Data, return_probs: bool = False) -> Tuple[int, float, float, Optional[np.ndarray]]:
        """
        Get action for a single observation.
        
        Args:
            observation (Data): Graph observation
            return_probs (bool): Whether to return action probabilities
            
        Returns:
            tuple: (action, log_prob, value, action_probs) where action_probs is None if return_probs=False
        """
        self.eval()
        with torch.no_grad():
            action_logits, value, _ = self.forward([observation])
            
            # Sample action
            probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            if return_probs:
                action_probs = probs.cpu().numpy().flatten()
                return action.item(), log_prob.item(), value.item(), action_probs
            else:
                return action.item(), log_prob.item(), value.item(), None
    
    def evaluate_actions(self, observations: List[Data], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for computing losses.
        
        Args:
            observations (List[Data]): List of graph observations
            actions (torch.Tensor): Action indices
            
        Returns:
            tuple: (log_probs, values, entropy, gnn_embeddings)
        """
        action_logits, values, gnn_embeddings = self.forward(observations)
        
        # Compute log probabilities
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values, entropy, gnn_embeddings
    
    def update(self, batch_data: Dict) -> Dict:
        """
        Update the agent using PPO.
        
        Args:
            batch_data (Dict): Training batch with observations, actions, rewards, etc.
            
        Returns:
            Dict: Training statistics
        """
        self.train()
        
        observations = batch_data['observations']
        actions = batch_data['actions'].to(self.device)
        old_log_probs = batch_data['log_probs'].to(self.device)
        returns = batch_data['returns'].to(self.device)
        advantages = batch_data['advantages'].to(self.device)
        
        # Convert observations to batch using standard PyTorch Geometric batching
        # (Ensures consistent GNN encoder usage for proper variance calculation)
        batch = Batch.from_data_list(observations).to(self.device)
        
        # Multiple PPO epochs
        ppo_epochs = self.config.get('ppo_epochs', 4)
        batch_size = len(observations)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_gnn_variance_loss = 0.0
        total_clip_fraction = 0.0
        
        for epoch in range(ppo_epochs):
            # Evaluate current policy
            log_probs, values, entropy, gnn_embeddings = self.evaluate_actions(observations, actions)
            
            # Policy loss (PPO clipped objective)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, returns)
            
            # Entropy loss (for exploration)
            entropy_loss = -entropy.mean()
            
            # GNN variance loss (for preventing embedding collapse)
            # Only compute if coefficient is non-zero
            if self.gnn_variance_loss_coef > 0:
                gnn_variance_loss = -torch.var(gnn_embeddings, dim=0).mean()
            else:
                gnn_variance_loss = torch.tensor(0.0, device=self.device)
            
            # Total loss
            loss = (policy_loss + 
                   self.value_loss_coef * value_loss + 
                   self.entropy_coef * entropy_loss +
                   self.gnn_variance_loss_coef * gnn_variance_loss)
            
            # Backward pass
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(
                list(self.gnn_encoder.parameters()) +
                list(self.actor_head.parameters()),
                self.max_grad_norm
            )
            nn.utils.clip_grad_norm_(
                list(self.critic_head.parameters()),
                self.max_grad_norm
            )
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            # Statistics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
            total_gnn_variance_loss += gnn_variance_loss.item()
            
            # Clip fraction
            clip_fraction = (abs(ratio - 1) > self.clip_range).float().mean().item()
            total_clip_fraction += clip_fraction
        
        # Update training statistics with enhanced value function diagnostics
        self.training_stats = {
            'policy_loss': total_policy_loss / ppo_epochs,
            'value_loss': total_value_loss / ppo_epochs,
            'entropy_loss': total_entropy_loss / ppo_epochs,
            'gnn_variance_loss': total_gnn_variance_loss / ppo_epochs,
            'total_loss': (total_policy_loss + self.value_loss_coef * total_value_loss + 
                          self.entropy_coef * total_entropy_loss +
                          self.gnn_variance_loss_coef * total_gnn_variance_loss) / ppo_epochs,
            'clip_fraction': total_clip_fraction / ppo_epochs,
            
            # NEW: Value function diagnostics
            'value_predictions': values.detach().cpu().numpy().tolist(),
            'value_targets': returns.detach().cpu().numpy().tolist(),
            'gradient_norm': 0.0  # Will be set below
        }
        
        # Calculate gradient norm for diagnostics
        total_norm = 0
        for p in list(self.gnn_encoder.parameters()) + list(self.actor_head.parameters()) + list(self.critic_head.parameters()):
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.training_stats['gradient_norm'] = total_norm
        
        return self.training_stats
    
    def save_model(self, path: str):
        """Save model to file."""
        torch.save({
            'gnn_encoder_state_dict': self.gnn_encoder.state_dict(),
            'actor_head_state_dict': self.actor_head.state_dict(),
            'critic_head_state_dict': self.critic_head.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'training_stats': self.training_stats
        }, path)
    
    def load_model(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.gnn_encoder.load_state_dict(checkpoint['gnn_encoder_state_dict'])
        self.actor_head.load_state_dict(checkpoint['actor_head_state_dict'])
        self.critic_head.load_state_dict(checkpoint['critic_head_state_dict'])
        
        # Handle both old and new optimizer formats
        if 'actor_optimizer_state_dict' in checkpoint and 'critic_optimizer_state_dict' in checkpoint:
            # New format with separate optimizers
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        elif 'optimizer_state_dict' in checkpoint:
            # Old format - load into actor optimizer for backward compatibility
            self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Warning: Loading old checkpoint format. Critic optimizer state not restored.")
        
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
    
    def get_training_stats(self) -> Dict:
        """Get current training statistics."""
        return self.training_stats.copy()


def compute_gae(rewards: List[float], values: List[float], gamma: float = 0.99, gae_lambda: float = 0.90) -> Tuple[List[float], List[float]]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards (List[float]): List of rewards
        values (List[float]): List of value estimates
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda parameter
        
    Returns:
        tuple: (advantages, returns)
    """
    advantages = []
    returns = []
    
    # Compute advantages using GAE
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
    
    return advantages, returns


def normalize_advantages(advantages: List[float]) -> List[float]:
    """
    Normalize advantages to have zero mean and unit variance.
    
    Args:
        advantages (List[float]): List of advantages
        
    Returns:
        List[float]: Normalized advantages
    """
    if not advantages:
        return advantages
    
    advantages = np.array(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages.tolist() 