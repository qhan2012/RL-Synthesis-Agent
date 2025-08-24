#!/usr/bin/env python3
"""
Analyze GNN Outputs from Saved Models

This script loads saved model checkpoints and extracts GNN embeddings
for visualization and analysis using t-SNE.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

# Add project root to path
sys.path.append('.')

from models.ppo_agent import PPOSynthesisAgent
from data.dataset import CircuitDataset
from env.synthesis_env import SynthesisEnvironment


class GNNOutputAnalyzer:
    """Analyzer for extracting and visualizing GNN outputs from saved models."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the analyzer.
        
        Args:
            config_path: Path to model configuration file
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load configuration
        if config_path:
            self.config = self.load_config(config_path)
        else:
            # Default configuration for medium_train_300
            self.config = {
                'gnn_encoder': {
                    'type': 'gin',
                    'input_dim': 6,
                    'edge_dim': 2,
                    'hidden_dim': 128,
                    'output_dim': 128,
                    'num_layers': 3,
                    'dropout': 0.1,
                    'pooling': 'mean',
                    'use_global_features': True,
                    'use_edge_features': True
                },
                'actor_head': {
                    'input_dim': 129,  # 128 + 1 for timestep
                    'hidden_dim': 64,
                    'output_dim': 5
                },
                'critic_head': {
                    'input_dim': 129,  # 128 + 1 for timestep
                    'hidden_dim': 64,
                    'output_dim': 1
                },
                'learning_rate': 3e-4
            }
        
        # Create environment for generating test data
        self.env = SynthesisEnvironment(
            max_steps=10,
            action_space=['b', 'rw', 'rf', 'rwz', 'rfz'],
            reward_shaping=True,
            reward_normalization=True,
            final_bonus=True,
            cleanup_logs=False
        )
        
        # Load dataset for test circuits
        try:
            self.dataset = CircuitDataset(
                data_root='testcase',
                sources=['IWLS', 'MCNC']
            )
            print(f"Loaded dataset with {len(self.dataset)} circuits")
        except Exception as e:
            print(f"Warning: Could not load dataset: {e}")
            self.dataset = None
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_model(self, model_path: str) -> PPOSynthesisAgent:
        """Load a saved model checkpoint."""
        print(f"Loading model from: {model_path}")
        
        # Load checkpoint first to inspect structure
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Check if we need to adjust config based on saved model
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            print(f"Found saved config, using it instead of default")
            self.config = saved_config
        
        # Create agent with potentially updated config
        agent = PPOSynthesisAgent(self.config)
        
        # Load state dict - handle different checkpoint formats
        if 'gnn_encoder_state_dict' in checkpoint:
            # New format with separate state dicts
            print("Loading separate state dicts...")
            
            # Load GNN encoder with strict=False to handle missing keys
            try:
                agent.gnn_encoder.load_state_dict(checkpoint['gnn_encoder_state_dict'], strict=False)
                print("GNN encoder loaded successfully")
            except Exception as e:
                print(f"Warning: GNN encoder loading had issues: {e}")
                # Try to load what we can
                agent.gnn_encoder.load_state_dict(checkpoint['gnn_encoder_state_dict'], strict=False)
            
            # Load actor and critic heads
            try:
                agent.actor_head.load_state_dict(checkpoint['actor_head_state_dict'])
                agent.critic_head.load_state_dict(checkpoint['critic_head_state_dict'])
                print("Actor and critic heads loaded successfully")
            except Exception as e:
                print(f"Warning: Head loading had issues: {e}")
                # Try to load what we can
                agent.actor_head.load_state_dict(checkpoint['actor_head_state_dict'], strict=False)
                agent.critic_head.load_state_dict(checkpoint['critic_head_state_dict'], strict=False)
                
        elif 'model_state_dict' in checkpoint:
            # Standard format
            agent.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct state dict
            agent.load_state_dict(checkpoint)
        
        agent.eval()
        print(f"Model loaded successfully")
        return agent
    
    def extract_gnn_embeddings(self, agent: PPOSynthesisAgent, num_samples: int = 100) -> Tuple[np.ndarray, List[str]]:
        """
        Extract GNN embeddings from the model using test circuits.
        
        Args:
            agent: Loaded PPO agent
            num_samples: Number of samples to extract
            
        Returns:
            tuple: (embeddings, circuit_names)
        """
        embeddings = []
        circuit_names = []
        
        if self.dataset is None:
            print("No dataset available, cannot extract embeddings")
            return np.array([]), []
        
        # Sample circuits
        sample_circuits = list(self.dataset.circuits)[:min(num_samples, len(self.dataset))]
        
        print(f"Extracting GNN embeddings from {len(sample_circuits)} circuits...")
        
        with torch.no_grad():
            for i, circuit_name in enumerate(sample_circuits):
                try:
                    # Get circuit path
                    circuit_path = self.dataset.circuit_metadata[circuit_name]['path']
                    
                    # Create environment observation
                    obs = self.env.reset(circuit_path)
                    
                    # Handle observation format - it should be a PyTorch Geometric Data object
                    if isinstance(obs, tuple):
                        # Environment returns (observation, info)
                        data = obs[0]
                    else:
                        # Environment returns observation directly
                        data = obs
                    
                    # Ensure we have a PyTorch Geometric Data object
                    if not hasattr(data, 'x') or not hasattr(data, 'edge_index'):
                        print(f"Warning: Invalid observation format for circuit {circuit_name}")
                        continue
                    
                    # Move to device
                    data = data.to(self.device)
                    
                    # Extract GNN embeddings
                    gnn_embeddings = agent.gnn_encoder(data)
                    
                    # Convert to numpy
                    embedding_np = gnn_embeddings.cpu().numpy()
                    embeddings.append(embedding_np.flatten())  # Flatten to 1D
                    circuit_names.append(circuit_name)
                    
                    if (i + 1) % 10 == 0:
                        print(f"Processed {i + 1}/{len(sample_circuits)} circuits")
                        
                except Exception as e:
                    print(f"Error processing circuit {circuit_name}: {e}")
                    continue
        
        if embeddings:
            embeddings_array = np.array(embeddings)
            print(f"Extracted {embeddings_array.shape[0]} embeddings of dimension {embeddings_array.shape[1]}")
            return embeddings_array, circuit_names
        else:
            print("No embeddings extracted")
            return np.array([]), []
    
    def visualize_tsne(self, embeddings: np.ndarray, circuit_names: List[str], 
                      model_name: str, output_dir: str = "gnn_analysis"):
        """
        Create t-SNE visualization of GNN embeddings.
        
        Args:
            embeddings: GNN embeddings array
            circuit_names: List of circuit names
            model_name: Name of the model for the plot title
            output_dir: Output directory for saving plots
        """
        if len(embeddings) == 0:
            print("No embeddings to visualize")
            return
        
        print(f"Creating t-SNE visualization for {len(embeddings)} embeddings...")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Apply t-SNE
        print("Applying t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=50)
        
        # Add some circuit names as labels (avoid overcrowding)
        if len(circuit_names) <= 20:
            for i, name in enumerate(circuit_names):
                plt.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                           fontsize=8, alpha=0.7)
        
        plt.title(f't-SNE Visualization of GNN Embeddings\nModel: {model_name}')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = Path(output_dir) / f"tsne_{model_name.replace('.pth', '')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE plot to: {plot_path}")
        
        # Also create PCA visualization for comparison
        print("Creating PCA visualization...")
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 8))
        plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], alpha=0.6, s=50)
        
        if len(circuit_names) <= 20:
            for i, name in enumerate(circuit_names):
                plt.annotate(name, (embeddings_pca[i, 0], embeddings_pca[i, 1]), 
                           fontsize=8, alpha=0.7)
        
        plt.title(f'PCA Visualization of GNN Embeddings\nModel: {model_name}')
        plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.grid(True, alpha=0.3)
        
        # Save PCA plot
        pca_plot_path = Path(output_dir) / f"pca_{model_name.replace('.pth', '')}.png"
        plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved PCA plot to: {pca_plot_path}")
        
        plt.show()
    
    def analyze_model(self, model_path: str, num_samples: int = 100, output_dir: str = "gnn_analysis"):
        """
        Analyze a single model checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            num_samples: Number of samples to extract
            output_dir: Output directory for results
        """
        print(f"\n{'='*60}")
        print(f"ANALYZING MODEL: {model_path}")
        print(f"{'='*60}")
        
        try:
            # Load model
            agent = self.load_model(model_path)
            
            # Extract embeddings
            embeddings, circuit_names = self.extract_gnn_embeddings(agent, num_samples)
            
            if len(embeddings) > 0:
                # Save embeddings
                Path(output_dir).mkdir(exist_ok=True)
                embeddings_path = Path(output_dir) / f"embeddings_{Path(model_path).stem}.npz"
                np.savez(embeddings_path, embeddings=embeddings, circuit_names=circuit_names)
                print(f"Saved embeddings to: {embeddings_path}")
                
                # Create visualizations
                model_name = Path(model_path).name
                self.visualize_tsne(embeddings, circuit_names, model_name, output_dir)
                
                # Print statistics
                print(f"\nEmbedding Statistics:")
                print(f"  Shape: {embeddings.shape}")
                print(f"  Mean: {np.mean(embeddings):.4f}")
                print(f"  Std: {np.std(embeddings):.4f}")
                print(f"  Min: {np.min(embeddings):.4f}")
                print(f"  Max: {np.max(embeddings):.4f}")
                
            else:
                print("No embeddings extracted from this model")
                
        except Exception as e:
            print(f"Error analyzing model {model_path}: {e}")
    
    def analyze_multiple_models(self, model_dir: str = "outputs/models", 
                              pattern: str = "*medium*", num_samples: int = 100):
        """
        Analyze multiple model checkpoints.
        
        Args:
            model_dir: Directory containing model checkpoints
            pattern: Pattern to match model files
            num_samples: Number of samples to extract per model
        """
        model_dir = Path(model_dir)
        if not model_dir.exists():
            print(f"Model directory {model_dir} does not exist")
            return
        
        # Find model files
        model_files = list(model_dir.glob(pattern))
        print(f"Found {len(model_files)} model files matching pattern '{pattern}'")
        
        for model_path in model_files:
            self.analyze_model(str(model_path), num_samples)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze GNN outputs from saved models")
    parser.add_argument("--model", type=str, help="Path to specific model checkpoint")
    parser.add_argument("--model-dir", type=str, default="outputs/models", 
                       help="Directory containing model checkpoints")
    parser.add_argument("--pattern", type=str, default="*medium*", 
                       help="Pattern to match model files")
    parser.add_argument("--num-samples", type=int, default=100, 
                       help="Number of samples to extract per model")
    parser.add_argument("--output-dir", type=str, default="gnn_analysis", 
                       help="Output directory for results")
    parser.add_argument("--config", type=str, help="Path to model configuration file")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = GNNOutputAnalyzer(args.config)
    
    if args.model:
        # Analyze single model
        analyzer.analyze_model(args.model, args.num_samples, args.output_dir)
    else:
        # Analyze multiple models
        analyzer.analyze_multiple_models(args.model_dir, args.pattern, args.num_samples)


if __name__ == "__main__":
    main() 