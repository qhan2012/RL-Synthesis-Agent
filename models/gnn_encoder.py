"""
GNN Encoder for Circuit Graph Representation

This module implements a Graph Neural Network encoder that processes
circuit graphs and produces node embeddings for the actor-critic network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv, global_mean_pool, global_add_pool, global_max_pool, GlobalAttention
from torch_geometric.data import Data, Batch
from typing import Dict, Optional


class GINEncoder(nn.Module):
    """
    Graph Isomorphism Network (GIN) encoder for circuit graphs with edge features.
    
    Args:
        input_dim (int): Input node feature dimension
        edge_dim (int): Input edge feature dimension
        hidden_dim (int): Hidden dimension for GNN layers
        output_dim (int): Output embedding dimension
        num_layers (int): Number of GIN layers
        dropout (float): Dropout rate
        pooling (str): Graph pooling method ('mean', 'sum', 'max')
        use_global_features (bool): Whether to use global features
        use_edge_features (bool): Whether to use edge features
    """
    
    def __init__(
        self,
        input_dim: int,
        edge_dim: int = 2,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = 'mean',
        use_global_features: bool = True,
        use_edge_features: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        self.use_global_features = use_global_features
        self.use_edge_features = use_edge_features
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Edge feature projection (if using edge features)
        if self.use_edge_features:
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        # GIN layers with edge features (GINE) or without (GIN)
        self.gin_layers = nn.ModuleList()
        for _ in range(num_layers):
            if self.use_edge_features:
                # Use GINEConv which supports edge features
                gin_layer = GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    ),
                    edge_dim=hidden_dim,  # Edge features after projection
                    train_eps=True
                )
            else:
                # Use standard GINConv
                gin_layer = GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim)
                    ),
                    train_eps=True
                )
            self.gin_layers.append(gin_layer)
        
        # Global feature processing
        if self.use_global_features:
            self.global_proj = nn.Linear(6, hidden_dim // 4)  # 6 global features
            self.global_combine = nn.Linear(hidden_dim + hidden_dim // 4, output_dim)
        else:
            self.global_proj = None
            self.global_combine = None
        
        # Output projection (only used if not using global features)
        if not self.use_global_features:
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_proj = None
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Attention pooling gate network (2-layer MLP with ReLU)
        # Takes node embeddings [num_nodes, hidden_dim] and outputs attention scores [num_nodes, 1]
        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Global attention pooling
        self.attention_pool = GlobalAttention(gate_nn=self.attention_gate)
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the GNN encoder.
        
        Args:
            data (Data): PyTorch Geometric Data object with node features, edge index, and edge features
            
        Returns:
            torch.Tensor: Graph-level embedding [batch_size, output_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        
        # Process edge features if available and enabled
        edge_attr = None
        if self.use_edge_features and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = self.edge_proj(data.edge_attr)
        elif self.use_edge_features and hasattr(data, 'edge_index'):
            # ROBUST FIX: Create dummy edge features if missing but edge features are expected
            num_edges = data.edge_index.size(1)
            if num_edges > 0:
                # Create dummy edge features: [is_inverted=0, level_diff=1] for all edges
                dummy_edge_attr = torch.zeros((num_edges, 2), device=data.edge_index.device, dtype=torch.float)
                dummy_edge_attr[:, 1] = 1.0  # Set level_diff to 1
                edge_attr = self.edge_proj(dummy_edge_attr)
                print(f"[DEBUG] Created {num_edges} dummy edge features for missing edge_attr")
        
        # GIN layers with residual connections
        for i, (gin_layer, bn) in enumerate(zip(self.gin_layers, self.batch_norms)):
            identity = x
            
            # Forward pass with or without edge features
            if self.use_edge_features and edge_attr is not None:
                x = gin_layer(x, edge_index, edge_attr)
            else:
                x = gin_layer(x, edge_index)
            
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection (for all layers including first)
            x = x + identity
        
        # Graph-level pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)  # FIXED: Use proper sum pooling
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)  # FIXED: Use proper max pooling
        elif self.pooling == 'attention':
            # Use attention pooling: output shape [num_graphs, hidden_dim]
            x = self.attention_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Process global features if available and enabled
        if self.use_global_features and hasattr(data, 'global_x') and data.global_x is not None and self.global_proj is not None:
            # Process global features
            global_features = data.global_x
            if global_features.dim() == 1:
                # If length is a multiple of 6, reshape to [batch_size, 6]
                if global_features.numel() % 6 == 0:
                    global_features = global_features.view(-1, 6)
                else:
                    global_features = global_features.unsqueeze(0)  # Add batch dimension
            # Ensure global features have the right shape for batch processing
            if global_features.size(0) != x.size(0):
                global_features = global_features.repeat(x.size(0), 1)
            # Project global features
            global_proj = self.global_proj(global_features)
            global_proj = F.relu(global_proj)
            # Combine graph embedding with global features
            combined = torch.cat([x, global_proj], dim=1)
            x = self.global_combine(combined)
        else:
            # Use standard output projection
            if self.output_proj is not None:
                x = self.output_proj(x)
            else:
                # Fallback if neither global features nor output projection is available
                raise ValueError("Neither global features nor output projection is available")
        
        return x


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network (GCN) encoder with edge features as an alternative to GIN.
    """
    
    def __init__(
        self,
        input_dim: int,
        edge_dim: int = 2,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        pooling: str = 'mean',
        use_global_features: bool = True,
        use_edge_features: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        self.use_global_features = use_global_features
        self.use_edge_features = use_edge_features
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Edge feature projection (if using edge features)
        if self.use_edge_features:
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        # GCN layers (simplified as linear layers with edge aggregation)
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gcn_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Global feature processing
        if self.use_global_features:
            self.global_proj = nn.Linear(6, hidden_dim // 4)  # 6 global features
            self.global_combine = nn.Linear(hidden_dim + hidden_dim // 4, output_dim)
        else:
            self.global_proj = None
            self.global_combine = None
        
        # Output projection (only used if not using global features)
        if not self.use_global_features:
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_proj = None
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Attention pooling gate network (2-layer MLP with ReLU)
        # Takes node embeddings [num_nodes, hidden_dim] and outputs attention scores [num_nodes, 1]
        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Global attention pooling
        self.attention_pool = GlobalAttention(gate_nn=self.attention_gate)
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the GCN encoder.
        
        Args:
            data (Data): PyTorch Geometric Data object
            
        Returns:
            torch.Tensor: Graph-level embedding
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        
        # Process edge features if available and enabled
        edge_attr = None
        if self.use_edge_features and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = self.edge_proj(data.edge_attr)
        
        # GCN layers with neighbor aggregation
        for i, (gcn_layer, bn) in enumerate(zip(self.gcn_layers, self.batch_norms)):
            # Simple neighbor aggregation (mean of neighbors)
            row, col = edge_index
            neighbor_agg = torch.zeros_like(x)
            
            if edge_attr is not None and self.use_edge_features:
                # Weight neighbor features by edge features
                edge_weighted_features = x[col] * edge_attr
                neighbor_agg.index_add_(0, row, edge_weighted_features)
            else:
                # Standard aggregation without edge weighting
                neighbor_agg.index_add_(0, row, x[col])
            
            # Count neighbors for normalization
            neighbor_count = torch.zeros(x.size(0), 1, device=x.device)
            neighbor_count.index_add_(0, row, torch.ones_like(x[col, :1]))
            neighbor_count = torch.clamp(neighbor_count, min=1)
            
            # Normalized aggregation
            neighbor_agg = neighbor_agg / neighbor_count
            
            # Combine with self features
            x = x + neighbor_agg
            x = gcn_layer(x)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph-level pooling
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'attention':
            # Use attention pooling: output shape [num_graphs, hidden_dim]
            x = self.attention_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Process global features if available and enabled
        if self.use_global_features and hasattr(data, 'global_x') and data.global_x is not None and self.global_proj is not None:
            # Process global features
            global_features = data.global_x
            if global_features.dim() == 1:
                global_features = global_features.unsqueeze(0)  # Add batch dimension
            # Ensure global features have the right shape for batch processing
            if global_features.size(0) != x.size(0):
                global_features = global_features.repeat(x.size(0), 1)
            # Project global features
            global_proj = self.global_proj(global_features)
            global_proj = F.relu(global_proj)
            # Combine graph embedding with global features
            combined = torch.cat([x, global_proj], dim=1)
            x = self.global_combine(combined)
        else:
            # Use standard output projection
            if self.output_proj is not None:
                x = self.output_proj(x)
            else:
                # Fallback if neither global features nor output projection is available
                raise ValueError("Neither global features nor output projection is available")
        
        return x


def create_gnn_encoder(config: Dict) -> nn.Module:
    """
    Factory function to create GNN encoder based on configuration.
    
    Args:
        config (Dict): Configuration dictionary with GNN parameters
        
    Returns:
        nn.Module: GNN encoder instance
    """
    gnn_type = config.get('type', 'GIN')
    hidden_dim = config.get('hidden_dim', 128)
    num_layers = config.get('num_layers', 3)
    activation = config.get('activation', 'ReLU')
    pooling = config.get('pooling', 'mean')
    use_global_features = config.get('use_global_features', True)
    use_edge_features = config.get('use_edge_features', True)
    
    # Default input dimensions for circuit features
    input_dim = 6  # [is_input, is_output, is_and, fanin, fanout, level]
    edge_dim = 2   # [is_inverted, level_diff]
    output_dim = hidden_dim
    
    if gnn_type.upper() == 'GIN':
        return GINEncoder(
            input_dim=input_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            pooling=pooling,
            use_global_features=use_global_features,
            use_edge_features=use_edge_features
        )
    elif gnn_type.upper() == 'GCN':
        return GCNEncoder(
            input_dim=input_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            pooling=pooling,
            use_global_features=use_global_features,
            use_edge_features=use_edge_features
        )
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}") 