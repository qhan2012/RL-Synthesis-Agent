"""
Critic Head for Value Function Network

This module implements the critic network that estimates the value function
V(s_t, t) for the current state and timestep.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class CriticHead(nn.Module):
    """
    Critic network that estimates the value function V(s_t, t).
    
    Args:
        input_dim (int): Input dimension (GNN embedding + timestep)
        hidden_dims (list): List of hidden layer dimensions
        dropout (float): Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
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
        
        # Output layer (single scalar value)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Value estimates [batch_size, 1]
        """
        return self.mlp(x).squeeze(-1)


def create_critic_head(config: Dict) -> CriticHead:
    """
    Factory function to create critic head based on configuration.
    
    Args:
        config (Dict): Configuration dictionary with critic parameters
        
    Returns:
        CriticHead: Critic head instance
    """
    input_dim = config.get('input_dim', 129)  # 128 (GNN) + 1 (timestep)
    layers = config.get('layers', [128, 64])
    output_type = config.get('output', 'scalar')
    dropout = config.get('dropout', 0.1)
    
    return CriticHead(
        input_dim=input_dim,
        hidden_dims=layers,
        dropout=dropout
    ) 