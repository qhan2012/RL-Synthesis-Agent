#!/usr/bin/env python3
"""
GNN Analysis Summary

This script analyzes and compares GNN embeddings from different saved models.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

def load_embeddings(embeddings_path):
    """Load embeddings from .npz file."""
    data = np.load(embeddings_path)
    embeddings = data['embeddings']
    circuit_names = data['circuit_names']
    return embeddings, circuit_names

def analyze_embeddings(embeddings, model_name):
    """Analyze embedding statistics."""
    print(f"\n=== {model_name} ===")
    print(f"Shape: {embeddings.shape}")
    print(f"Mean: {np.mean(embeddings):.4f}")
    print(f"Std: {np.std(embeddings):.4f}")
    print(f"Min: {np.min(embeddings):.4f}")
    print(f"Max: {np.max(embeddings):.4f}")
    print(f"L2 norm mean: {np.mean(np.linalg.norm(embeddings, axis=1)):.4f}")
    print(f"L2 norm std: {np.std(np.linalg.norm(embeddings, axis=1)):.4f}")

def compare_models(embeddings_dict):
    """Compare embeddings from different models."""
    print("\n=== MODEL COMPARISON ===")
    
    # Calculate pairwise similarities between models
    model_names = list(embeddings_dict.keys())
    n_models = len(model_names)
    
    similarity_matrix = np.zeros((n_models, n_models))
    
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Calculate cosine similarity between mean embeddings
                mean_emb1 = np.mean(embeddings_dict[name1], axis=0)
                mean_emb2 = np.mean(embeddings_dict[name2], axis=0)
                similarity = cosine_similarity([mean_emb1], [mean_emb2])[0, 0]
                similarity_matrix[i, j] = similarity
    
    # Print similarity matrix
    print("Cosine Similarity Matrix:")
    print("Model names:", model_names)
    print(similarity_matrix)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, cmap='viridis', 
                xticklabels=model_names, yticklabels=model_names)
    plt.title('Model Embedding Similarity Matrix')
    plt.tight_layout()
    plt.savefig('gnn_analysis/model_similarity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_combined_visualization(embeddings_dict, output_dir="gnn_analysis"):
    """Create combined t-SNE visualization of all models."""
    print("\n=== CREATING COMBINED VISUALIZATION ===")
    
    # Combine all embeddings
    all_embeddings = []
    all_labels = []
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for i, (model_name, embeddings) in enumerate(embeddings_dict.items()):
        all_embeddings.append(embeddings)
        all_labels.extend([f"{model_name}_{j}" for j in range(len(embeddings))])
    
    combined_embeddings = np.vstack(all_embeddings)
    
    # Apply t-SNE
    print("Applying t-SNE to combined embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_embeddings)-1))
    embeddings_2d = tsne.fit_transform(combined_embeddings)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    start_idx = 0
    for i, (model_name, embeddings) in enumerate(embeddings_dict.items()):
        end_idx = start_idx + len(embeddings)
        plt.scatter(embeddings_2d[start_idx:end_idx, 0], 
                   embeddings_2d[start_idx:end_idx, 1], 
                   alpha=0.6, s=50, label=model_name, color=colors[i % len(colors)])
        start_idx = end_idx
    
    plt.title('Combined t-SNE Visualization of GNN Embeddings from Different Models')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = Path(output_dir) / "combined_tsne_visualization.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined visualization to: {plot_path}")
    plt.show()

def main():
    """Main analysis function."""
    analysis_dir = Path("gnn_analysis")
    
    if not analysis_dir.exists():
        print("No gnn_analysis directory found. Please run analyze_gnn_outputs.py first.")
        return
    
    # Find all embedding files
    embedding_files = list(analysis_dir.glob("embeddings_*.npz"))
    
    if not embedding_files:
        print("No embedding files found in gnn_analysis directory.")
        return
    
    print(f"Found {len(embedding_files)} embedding files:")
    for f in embedding_files:
        print(f"  - {f.name}")
    
    # Load all embeddings
    embeddings_dict = {}
    for embedding_file in embedding_files:
        model_name = embedding_file.stem.replace('embeddings_', '')
        embeddings, circuit_names = load_embeddings(embedding_file)
        embeddings_dict[model_name] = embeddings
        print(f"Loaded {len(embeddings)} embeddings for {model_name}")
    
    # Analyze each model
    for model_name, embeddings in embeddings_dict.items():
        analyze_embeddings(embeddings, model_name)
    
    # Compare models
    if len(embeddings_dict) > 1:
        compare_models(embeddings_dict)
        create_combined_visualization(embeddings_dict)
    
    # Create summary report
    print("\n=== SUMMARY REPORT ===")
    print(f"Total models analyzed: {len(embeddings_dict)}")
    print(f"Total embeddings: {sum(len(emb) for emb in embeddings_dict.values())}")
    print(f"Embedding dimension: {list(embeddings_dict.values())[0].shape[1]}")
    
    # Check for clustering patterns
    print("\n=== CLUSTERING ANALYSIS ===")
    for model_name, embeddings in embeddings_dict.items():
        # Calculate pairwise distances within model
        distances = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                distances.append(dist)
        
        if distances:
            print(f"{model_name}:")
            print(f"  Mean pairwise distance: {np.mean(distances):.4f}")
            print(f"  Std pairwise distance: {np.std(distances):.4f}")
            print(f"  Min distance: {np.min(distances):.4f}")
            print(f"  Max distance: {np.max(distances):.4f}")

if __name__ == "__main__":
    main() 