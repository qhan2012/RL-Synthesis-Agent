#!/usr/bin/env python3
"""
AAG2GNN Compatibility Layer for RL Synthesis Project

This module provides a compatibility layer that fixes PyTorch Geometric version
issues and integrates the AAG2GNN library with the RL synthesis project.
"""

import os
import sys
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Add AAG2GNN to path
sys.path.append('../run1/AAG2GNN')

try:
    from aag2gnn import load_aag_as_gnn_graph
    AAG2GNN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AAG2GNN not available: {e}")
    AAG2GNN_AVAILABLE = False

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import to_dense_batch
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PyTorch Geometric not available: {e}")
    PYTORCH_GEOMETRIC_AVAILABLE = False


class AAG2GNNCompatibilityLayer:
    """
    Compatibility layer that fixes PyTorch Geometric version issues
    and provides a unified interface for AAG2GNN integration.
    """
    
    def __init__(self):
        self.aag2gnn_available = AAG2GNN_AVAILABLE
        self.pytorch_geometric_available = PYTORCH_GEOMETRIC_AVAILABLE
        
        if not self.aag2gnn_available:
            print("Warning: AAG2GNN library not available")
        if not self.pytorch_geometric_available:
            print("Warning: PyTorch Geometric not available")
    
    def load_circuit_as_graph(self, file_path: str) -> Optional[Data]:
        """
        Load an AAG circuit file and convert it to a PyTorch Geometric graph.
        
        Args:
            file_path (str): Path to the .aag file
            
        Returns:
            torch_geometric.data.Data: Graph representation or None if failed
        """
        if not self.aag2gnn_available:
            print(f"Error: AAG2GNN not available, cannot load {file_path}")
            return None
        
        try:
            # Use AAG2GNN to load the circuit
            graph = load_aag_as_gnn_graph(file_path, include_inverter=True)
            
            # Add timestep feature for RL compatibility
            graph.timestep = torch.tensor([0.0], dtype=torch.float)
            
            return graph
            
        except Exception as e:
            print(f"Error loading circuit {file_path}: {e}")
            return None
    
    def create_batch_compatible(self, graphs: List[Data]) -> Data:
        """
        Create a batch from multiple graphs with version compatibility.
        
        Args:
            graphs (List[Data]): List of PyTorch Geometric Data objects
            
        Returns:
            torch_geometric.data.Data: Batched graph
        """
        if not graphs:
            raise ValueError("Cannot create batch from empty list")
        
        if len(graphs) == 1:
            return graphs[0]
        
        try:
            # Try the standard PyTorch Geometric batching
            batch = Batch.from_data_list(graphs)
            return batch
        except AttributeError as e:
            if "'listBatch' object has no attribute 'stores_as'" in str(e):
                # Use alternative batching method for compatibility
                return self._create_batch_manual(graphs)
            else:
                raise e
        except Exception as e:
            print(f"Warning: Standard batching failed, using manual method: {e}")
            return self._create_batch_manual(graphs)
    
    def _create_batch_manual(self, graphs: List[Data]) -> Data:
        """
        Manual batch creation for PyTorch Geometric compatibility.
        
        Args:
            graphs (List[Data]): List of PyTorch Geometric Data objects
            
        Returns:
            torch_geometric.data.Data: Manually batched graph
        """
        # Collect all node features
        all_x = []
        all_edge_index = []
        all_edge_attr = []
        all_global_x = []
        all_timesteps = []
        
        # Track node offsets for edge indices
        node_offset = 0
        
        for graph in graphs:
            # Node features
            all_x.append(graph.x)
            
            # Edge indices (need to offset by node count)
            edge_index = graph.edge_index.clone()
            edge_index[0] += node_offset
            edge_index[1] += node_offset
            all_edge_index.append(edge_index)
            
            # Edge attributes
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                all_edge_attr.append(graph.edge_attr)
            
            # Global features
            if hasattr(graph, 'global_x') and graph.global_x is not None:
                all_global_x.append(graph.global_x)
            
            # Timestep features
            if hasattr(graph, 'timestep'):
                all_timesteps.append(graph.timestep)
            
            # Update node offset
            node_offset += graph.num_nodes
        
        # Concatenate all features
        batched_x = torch.cat(all_x, dim=0)
        batched_edge_index = torch.cat(all_edge_index, dim=1)
        
        # Handle edge attributes
        batched_edge_attr = None
        if all_edge_attr:
            batched_edge_attr = torch.cat(all_edge_attr, dim=0)
        
        # Handle global features
        batched_global_x = None
        if all_global_x:
            batched_global_x = torch.cat(all_global_x, dim=0)
        
        # Handle timestep features
        batched_timesteps = None
        if all_timesteps:
            batched_timesteps = torch.cat(all_timesteps, dim=0)
        
        # Create batched graph
        batched_graph = Data(
            x=batched_x,
            edge_index=batched_edge_index,
            edge_attr=batched_edge_attr,
            global_x=batched_global_x,
            timestep=batched_timesteps,
            num_nodes=batched_x.size(0)
        )
        
        return batched_graph
    
    def process_circuit_batch(self, circuit_paths: List[str]) -> Optional[Data]:
        """
        Process multiple circuits and create a batch.
        
        Args:
            circuit_paths (List[str]): List of circuit file paths
            
        Returns:
            torch_geometric.data.Data: Batched graph or None if failed
        """
        graphs = []
        
        for path in circuit_paths:
            graph = self.load_circuit_as_graph(path)
            if graph is not None:
                graphs.append(graph)
        
        if not graphs:
            print("Warning: No valid graphs loaded")
            return None
        
        return self.create_batch_compatible(graphs)


class CompatiblePPOAgent:
    """
    PPO Agent with AAG2GNN compatibility layer.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.compatibility_layer = AAG2GNNCompatibilityLayer()
        
        # Initialize networks (simplified for compatibility)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # GNN encoder configuration
        gnn_config = config.get('gnn_encoder', {})
        self.gnn_hidden_dim = gnn_config.get('hidden_dim', 256)
        
        # Actor head configuration
        actor_config = config.get('actor_head', {})
        self.actor_layers = actor_config.get('layers', [256, 128, 64])
        self.actor_input_dim = actor_config.get('input_dim', 257)  # GNN + timestep
        
        # Critic head configuration
        critic_config = config.get('critic_head', {})
        self.critic_layers = critic_config.get('layers', [256, 128, 64])
        self.critic_input_dim = critic_config.get('input_dim', 257)  # GNN + timestep
        
        # PPO parameters
        self.entropy_coef = config.get('entropy_coef', 0.1)
        self.value_loss_coef = config.get('value_loss_coef', 0.3)
        self.gnn_variance_loss_coef = config.get('gnn_variance_loss_coef', 0.01)
        
        # Number of actions
        self.num_actions = 5  # [b, rw, rf, rwz, rfz]
        
        # Create simple MLP networks for compatibility
        self.actor_network = self._create_mlp(self.actor_input_dim, self.actor_layers, self.num_actions)
        self.critic_network = self._create_mlp(self.critic_input_dim, self.critic_layers, 1)
        
        # Move to device
        self.actor_network.to(self.device)
        self.critic_network.to(self.device)
        
        # Optimizer
        learning_rate = config.get('learning_rate', 5e-4)
        self.optimizer = torch.optim.Adam(
            list(self.actor_network.parameters()) + list(self.critic_network.parameters()),
            lr=learning_rate
        )
    
    def _create_mlp(self, input_dim: int, hidden_dims: List[int], output_dim: int) -> torch.nn.Module:
        """Create a simple MLP network."""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        
        return torch.nn.Sequential(*layers)
    
    def forward(self, observations: List[Data]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the agent.
        
        Args:
            observations (List[Data]): List of graph observations
            
        Returns:
            tuple: (action_logits, value_estimates, gnn_embeddings)
        """
        # Process observations through compatibility layer
        batch = self.compatibility_layer.create_batch_compatible(observations)
        batch = batch.to(self.device)
        
        # Extract features (simplified - just use node features)
        # In a real implementation, you'd use a proper GNN encoder
        if hasattr(batch, 'x') and batch.x is not None:
            # Use mean pooling of node features as graph embedding
            gnn_embeddings = torch.mean(batch.x, dim=0, keepdim=True)
            if gnn_embeddings.dim() == 1:
                gnn_embeddings = gnn_embeddings.unsqueeze(0)
        else:
            # Fallback: create dummy embeddings
            gnn_embeddings = torch.zeros((1, self.gnn_hidden_dim), device=self.device)
        
        # Add timestep information
        timesteps = torch.stack([obs.timestep for obs in observations]).to(self.device)
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(1)
        
        # Combine features
        combined_features = torch.cat([gnn_embeddings, timesteps], dim=1)
        
        # Get actor and critic outputs
        action_logits = self.actor_network(combined_features)
        value_estimates = self.critic_network(combined_features)
        
        return action_logits, value_estimates, gnn_embeddings
    
    def get_action(self, observations: List[Data]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action for observations.
        
        Args:
            observations (List[Data]): List of graph observations
            
        Returns:
            tuple: (action, action_logits, value)
        """
        self.eval()
        with torch.no_grad():
            action_logits, value, _ = self.forward(observations)
            
            # Sample action
            probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            return action, action_logits, value
    
    def evaluate_actions(self, observations: List[Data], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given observations.
        
        Args:
            observations (List[Data]): List of graph observations
            actions (torch.Tensor): Actions to evaluate
            
        Returns:
            tuple: (action_logits, values, entropy, gnn_embeddings)
        """
        action_logits, values, gnn_embeddings = self.forward(observations)
        
        # Calculate entropy
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        entropy = dist.entropy()
        
        return action_logits, values, entropy, gnn_embeddings


def test_aag2gnn_compatibility():
    """Test the AAG2GNN compatibility layer."""
    print("🧪 Testing AAG2GNN compatibility layer...")
    
    # Test circuit path
    test_circuit = "testcase/EPFL/ctrl/ctrl.aag"
    
    if not os.path.exists(test_circuit):
        print(f"❌ Test circuit not found: {test_circuit}")
        return False
    
    try:
        # Test compatibility layer
        compatibility_layer = AAG2GNNCompatibilityLayer()
        
        # Test single circuit loading
        print("1. Testing single circuit loading...")
        graph = compatibility_layer.load_circuit_as_graph(test_circuit)
        if graph is not None:
            print(f"   ✅ Circuit loaded successfully")
            print(f"   Nodes: {graph.num_nodes}")
            print(f"   Edges: {graph.edge_index.size(1)}")
            print(f"   Node features: {graph.x.shape}")
            if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                print(f"   Edge features: {graph.edge_attr.shape}")
        else:
            print("   ❌ Circuit loading failed")
            return False
        
        # Test batching
        print("2. Testing batch creation...")
        graphs = [graph, graph]  # Create a batch of 2 identical graphs
        batch = compatibility_layer.create_batch_compatible(graphs)
        if batch is not None:
            print(f"   ✅ Batch created successfully")
            print(f"   Batch nodes: {batch.num_nodes}")
            print(f"   Batch edges: {batch.edge_index.size(1)}")
        else:
            print("   ❌ Batch creation failed")
            return False
        
        # Test PPO agent
        print("3. Testing PPO agent...")
        config = {
            'gnn_encoder': {
                'type': 'GIN',
                'hidden_dim': 256,
                'num_layers': 3,
                'pooling': 'mean',
                'use_global_features': True,
                'use_edge_features': True
            },
            'actor_head': {
                'layers': [256, 128, 64],
                'input_dim': 257,
                'dropout': 0.1
            },
            'critic_head': {
                'layers': [256, 128, 64],
                'input_dim': 257,
                'dropout': 0.1
            },
            'learning_rate': 5e-4,
            'entropy_coef': 0.1,
            'value_loss_coef': 0.3,
            'gnn_variance_loss_coef': 0.01
        }
        
        agent = CompatiblePPOAgent(config)
        print("   ✅ PPO agent created successfully")
        
        # Test agent action
        print("4. Testing agent action...")
        action, action_logits, value = agent.get_action([graph])
        print(f"   ✅ Agent action successful")
        print(f"   Action: {action}")
        print(f"   Action logits shape: {action_logits.shape}")
        print(f"   Value shape: {value.shape}")
        
        print("🎉 All compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_aag2gnn_compatibility() 