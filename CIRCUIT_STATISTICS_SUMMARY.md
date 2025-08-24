# 📊 Circuit Statistics Summary for RL Synthesis Agent

**Based on `medium_train_300_no_gnn_variance.py` Configuration**

## 🔗 **Circuit Benchmarks Source**

This project uses circuits from the [awesome-circuit-benchmarks](https://github.com/qhan2012/awesome-circuit-benchmarks) repository, which provides a comprehensive collection of 402 digital circuit benchmarks across 6 major benchmark suites (~161 MB total):

- **EPFL**: 20 modern circuits (mem_ctrl, voter, adder, div, etc.)
- **MCNC**: 150+ classic benchmarks (DES, industrial circuits, etc.)
- **Synthetic**: 200+ artificially generated circuits  
- **ISCAS85**: 11 combinational benchmarks
- **ISCAS89**: 31 sequential benchmarks
- **ITC99**: 22 test conference benchmarks

The RL agent uses a **subset of 541 circuits** selected for balanced complexity distribution across training, validation, and test sets.

---

## 🎯 **Overall Dataset Statistics**

| **Metric** | **Value** | **Percentage** |
|------------|-----------|----------------|
| **Total Circuits** | **541** | **100.0%** |
| **Training Circuits** | **376** | **69.5%** |
| **Validation Circuits** | **80** | **14.8%** |
| **Test/Evaluation Circuits** | **85** | **15.7%** |

---

## 🏗️ **Complexity Level Analysis**

### **Complexity Definitions**
- **Small**: 4-100 gates
- **Medium**: 100-500 gates  
- **Large**: 500-2000 gates
- **Very Large**: 2000+ gates

### **Training Set (376 circuits)**
| **Complexity** | **Count** | **Percentage** |
|----------------|-----------|----------------|
| Small          | 326       | 86.7%          |
| Medium         | 46        | 12.2%          |
| Large          | 4         | 1.1%           |
| Very Large     | 0         | 0.0%           |

### **Validation Set (80 circuits)**
| **Complexity** | **Count** | **Percentage** |
|----------------|-----------|----------------|
| Small          | 70        | 87.5%          |
| Medium         | 9         | 11.2%          |
| Large          | 1         | 1.2%           |
| Very Large     | 0         | 0.0%           |

### **Test/Evaluation Set (85 circuits)**
| **Complexity** | **Count** | **Percentage** |
|----------------|-----------|----------------|
| Small          | 71        | 83.5%          |
| Medium         | 11        | 12.9%          |
| Large          | 2         | 2.4%           |
| Very Large     | 1         | 1.2%           |

---

## 🏭 **Benchmark Suite Distribution**

### **Training Set (376 circuits)**
| **Suite** | **Count** | **Percentage** | **Description** |
|-----------|-----------|----------------|-----------------|
| MCNC      | 221       | 58.8%          | Microelectronics Center benchmark |
| IWLS      | 136       | 36.2%          | International Workshop on Logic Synthesis |
| Synthetic | 19        | 5.1%           | Artificially generated circuits |

### **Validation Set (80 circuits)**
| **Suite** | **Count** | **Percentage** | **Description** |
|-----------|-----------|----------------|-----------------|
| IWLS      | 64        | 80.0%          | International Workshop on Logic Synthesis |
| Synthetic | 16        | 20.0%          | Artificially generated circuits |

### **Test/Evaluation Set (85 circuits)**
| **Suite** | **Count** | **Percentage** | **Description** |
|-----------|-----------|----------------|-----------------|
| Synthetic | 65        | 76.5%          | Artificially generated circuits |
| EPFL      | 20        | 23.5%          | École Polytechnique Fédérale de Lausanne |

---

## 📈 **Key Insights**

### **1. Complexity Distribution Strategy**
- **Heavily skewed toward small circuits** (~86-87% across all sets)
- **Medium complexity well-represented** (~11-13% across all sets)  
- **Large circuits minimal** (1-2% representation)
- **Very large circuits** only in test set (1 circuit)

### **2. Suite Diversity Strategy**
- **Training**: Dominated by classic benchmarks (MCNC + IWLS = 95%)
- **Validation**: IWLS-heavy for consistent validation
- **Test**: Synthetic-heavy for generalization testing

### **3. Balanced Split Quality**
- **Consistent complexity ratios** across train/val/test
- **Strategic suite distribution** to test different aspects:
  - Training: Classic benchmarks for robust learning
  - Validation: IWLS focus for stable validation
  - Test: Synthetic focus for generalization assessment

### **4. Training Implications**
- **376 training circuits** provide substantial learning data
- **Small circuit dominance** enables fast training episodes
- **Medium/large circuits** prevent overfitting to simple cases
- **80 validation episodes** align with validation set size
- **85 evaluation episodes** align with test set size

---

## 🎛️ **Configuration Alignment**

The `medium_train_300_no_gnn_variance.py` configuration perfectly aligns with this dataset:

```python
'dataset': {
    'train_ratio': 0.695,   # 376/541 circuits (69.5%)
    'val_ratio': 0.148,     # 80/541 circuits (14.8%)
    'eval_ratio': 0.157,    # 85/541 circuits (15.7%)
}

'validation': {
    'val_episodes': 80,     # Uses all validation circuits
}

'eval': {
    'eval_episodes': 85,    # Uses all evaluation circuits
}
```

---

## 🚀 **Training Strategy Benefits**

1. **Scalable Learning**: Small circuits enable fast episode completion
2. **Complexity Progression**: Medium/large circuits prevent overfitting
3. **Suite Diversity**: Multiple benchmark sources ensure robustness
4. **Balanced Validation**: Consistent validation across complexity levels
5. **Generalization Testing**: Synthetic-heavy test set challenges generalization

This circuit distribution provides an optimal balance for training a robust RL agent capable of handling diverse synthesis optimization challenges.
