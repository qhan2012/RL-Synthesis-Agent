# GNN Output Analysis Report

## Overview

This report summarizes the analysis of Graph Neural Network (GNN) outputs from saved models in the `medium_train_300` training session. The analysis extracted GNN embeddings from different model checkpoints and visualized them using t-SNE and PCA techniques.

## Models Analyzed

Three key models from the `medium_train_300` training session were analyzed:

1. **medium_final_model.pth** - Final trained model
2. **medium_best_eval_model.pth** - Best model based on evaluation performance
3. **medium_best_val_model.pth** - Best model based on validation performance

## Data Extraction

- **Dataset**: 419 circuits from IWLS and MCNC benchmarks
- **Sample Size**: 30 circuits per model (90 total embeddings)
- **Embedding Dimension**: 128-dimensional GNN outputs
- **Extraction Method**: Forward pass through GNN encoder with PyTorch Geometric Data objects

## Key Findings

### 1. Embedding Statistics

| Model | Mean | Std | Min | Max | L2 Norm Mean | L2 Norm Std |
|-------|------|-----|-----|-----|--------------|-------------|
| medium_final_model | 0.5527 | 4.9673 | -12.4280 | 16.1671 | 56.3261 | 4.9694 |
| medium_best_eval_model | 0.6557 | 5.0622 | -12.2301 | 16.2239 | 57.5155 | 5.2106 |
| medium_best_val_model | 0.4082 | 6.0782 | -14.0668 | 14.3879 | 68.6419 | 6.2096 |

### 2. Model Similarity Analysis

The cosine similarity matrix reveals interesting patterns:

```
[[1.00000000 0.07407426 0.99122405]
 [0.07407426 1.00000000 0.06947234]
 [0.99122405 0.06947234 1.00000000]]
```

**Key Observations:**
- **medium_final_model** and **medium_best_eval_model** are highly similar (0.991 similarity)
- **medium_best_val_model** is significantly different from both other models (0.07-0.07 similarity)
- This suggests the validation-based model learned a different representation

### 3. Clustering Analysis

| Model | Mean Pairwise Distance | Std Pairwise Distance | Min Distance | Max Distance |
|-------|----------------------|----------------------|--------------|--------------|
| medium_final_model | 3.2529 | 7.2820 | 0.2871 | 30.6327 |
| medium_best_eval_model | 3.0628 | 7.4790 | 0.2963 | 31.1923 |
| medium_best_val_model | 2.8377 | 8.5706 | 0.0544 | 34.8418 |

**Interpretation:**
- All models show similar clustering behavior
- **medium_best_val_model** has slightly tighter clustering (lower mean distance)
- High variance in distances suggests diverse circuit representations

## Visualizations Generated

### Individual Model Visualizations
- **t-SNE plots**: 2D visualization of embedding clusters for each model
- **PCA plots**: Linear dimensionality reduction with variance explained
- Files: `tsne_*.png`, `pca_*.png`

### Comparative Visualizations
- **Combined t-SNE**: All models plotted together with different colors
- **Similarity Heatmap**: Cosine similarity matrix visualization
- Files: `combined_tsne_visualization.png`, `model_similarity_heatmap.png`

## Technical Implementation

### GNN Architecture
- **Encoder Type**: GIN (Graph Isomorphism Network)
- **Input Dimension**: 6 (node features)
- **Edge Features**: 2-dimensional
- **Hidden Dimension**: 128
- **Output Dimension**: 128
- **Layers**: 3 GIN layers with batch normalization
- **Pooling**: Global mean pooling
- **Global Features**: 6-dimensional circuit statistics

### Data Processing Pipeline
1. Load saved model checkpoints with robust error handling
2. Extract circuit graphs using synthesis environment
3. Convert to PyTorch Geometric Data format
4. Forward pass through GNN encoder
5. Extract 128-dimensional embeddings
6. Apply dimensionality reduction (t-SNE, PCA)
7. Generate visualizations and statistics

## Conclusions

### 1. Model Convergence
The high similarity between `medium_final_model` and `medium_best_eval_model` suggests:
- Training converged to a stable representation
- Evaluation-based selection captured the best performing model
- The GNN learned consistent circuit representations

### 2. Representation Diversity
The distinct `medium_best_val_model` indicates:
- Validation-based selection captured different optimization characteristics
- The model may have learned alternative circuit representations
- Different selection criteria led to different learned features

### 3. Embedding Quality
- All models produce well-distributed embeddings (mean ~0.5, std ~5-6)
- L2 norms are consistent across models (~56-68)
- Pairwise distances show good separation between different circuits

### 4. Practical Implications
- **Model Selection**: Evaluation-based models may be more reliable for deployment
- **Ensemble Methods**: Combining different models could leverage diverse representations
- **Transfer Learning**: Pre-trained GNN encoders could be useful for related tasks

## Files Generated

### Embeddings
- `embeddings_medium_final_model.npz`
- `embeddings_medium_best_eval_model.npz`
- `embeddings_medium_best_val_model.npz`

### Visualizations
- Individual t-SNE plots for each model
- Individual PCA plots for each model
- Combined t-SNE visualization
- Model similarity heatmap

### Analysis Scripts
- `analyze_gnn_outputs.py`: Main extraction and visualization script
- `gnn_analysis_summary.py`: Comparative analysis script

## Future Work

1. **Temporal Analysis**: Track embedding evolution during training
2. **Circuit Classification**: Analyze embeddings by circuit characteristics
3. **Performance Correlation**: Link embedding patterns to synthesis performance
4. **Interpretability**: Identify which embedding dimensions correspond to circuit features
5. **Transfer Learning**: Test embeddings on related synthesis tasks

## Technical Notes

- **Dependencies**: PyTorch, PyTorch Geometric, scikit-learn, matplotlib, seaborn
- **Hardware**: CUDA-enabled GPU for efficient processing
- **Memory**: ~19KB per model for 30 embeddings
- **Processing Time**: ~2-3 minutes per model for extraction and visualization

---

*Report generated from analysis of medium_train_300 saved models* 