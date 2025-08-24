"""
Actor Head for Policy Network

This module implements the actor network that outputs action probabilities
for the synthesis operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ActorHead(nn.Module):
    """
    Actor network that outputs action probabilities for synthesis operations.
    
    Args:
        input_dim (int): Input dimension (GNN embedding + timestep)
        hidden_dims (list): List of hidden layer dimensions
        num_actions (int): Number of possible actions
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64],
        num_actions: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_actions = num_actions
        self.dropout = dropout
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_actions))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the actor network.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Action logits [batch_size, num_actions]
        """
        return self.mlp(x)
    
    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get action probabilities using softmax.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Action probabilities [batch_size, num_actions]
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def sample_action(self, x: torch.Tensor) -> tuple:
        """
        Sample an action from the policy distribution.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            tuple: (action, log_prob, entropy)
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(probs)
        
        # Sample action
        action = dist.sample()
        
        # Get log probability of sampled action
        log_prob = dist.log_prob(action)
        
        # Calculate entropy
        entropy = dist.entropy()
        
        return action, log_prob, entropy


def create_actor_head(config: Dict) -> ActorHead:
    """
    Factory function to create actor head based on configuration.
    
    Args:
        config (Dict): Configuration dictionary with actor parameters
        
    Returns:
        ActorHead: Actor head instance
    """
    input_dim = config.get('input_dim', 129)  # 128 (GNN) + 1 (timestep)
    layers = config.get('layers', [128, 64])
    output_type = config.get('output', 'softmax')
    dropout = config.get('dropout', 0.1)
    
    # Number of synthesis actions
    num_actions = 5  # [b, rw, rf, rwz, rfz]
    
    return ActorHead(
        input_dim=input_dim,
        hidden_dims=layers,
        num_actions=num_actions,
        dropout=dropout
    ) 