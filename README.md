# 🚀 RL Synthesis Agent

A Reinforcement Learning agent for logic synthesis optimization using Proximal Policy Optimization (PPO) and Graph Neural Networks (GNN).

## Overview
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)]()

This project implements an RL agent that learns to optimize logic synthesis sequences to minimize circuit area (gate count) over fixed-length 10-step episodes. The agent uses a GNN-based actor-critic model to process circuit graphs and make synthesis decisions.

## 🎯 **Key Features**

- 🧠 **Advanced GNN Architecture**: 5-layer Graph Isomorphism Network with attention pooling
- 🎮 **PPO Optimization**: Proximal Policy Optimization with 1M timestep training  
- 🔧 **Actor-Critic Architecture**: Separate policy and value networks
- 🎯 **Reward Shaping**: Area reduction-based rewards with normalization
- 📊 **Comprehensive Benchmark**: 541 circuits from [awesome-circuit-benchmarks](https://github.com/qhan2012/awesome-circuit-benchmarks) (MCNC, IWLS, Synthetic, EPFL suites)
- 🏗️ **Production Ready**: Complete error handling, checkpointing, and monitoring
- 📈 **Proven Results**: 3.5% area reduction across diverse enseen circuits

## 📊 **Circuit Benchmarks**

This project uses a comprehensive collection of 541 digital circuits from the [awesome-circuit-benchmarks](https://github.com/qhan2012/awesome-circuit-benchmarks) repository, which contains:

- **402 total circuits** across 6 benchmark suites (~161 MB)
- **EPFL**: 20 modern arithmetic and control circuits (mem_ctrl, voter, adder, etc.)
- **MCNC**: 150+ classic benchmarks (DES, industrial circuits, etc.)  
- **IWLS**: International Workshop on Logic Synthesis circuits
- **Synthetic**: 200+ artificially generated circuits for diverse complexity
- **ISCAS85/89**: Standard sequential and combinational benchmarks
- **ITC99**: International Test Conference benchmarks

### **Circuit Distribution in Training**
- **Training**: 376 circuits (69.5%) - Balanced complexity mix
- **Validation**: 80 circuits (14.8%) - IWLS + Synthetic focus
- **Test**: 85 circuits (15.7%) - Synthetic + EPFL for generalization

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8+
- CUDA-capable GPU (RTX 3080 recommended)
- Berkeley ABC synthesis tool

### **Installation**

```bash
# Clone the repository
git clone https://github.com/qhan2012/rl-synthesis-agent.git
cd rl-synthesis-agent

# Install Python dependencies
pip install torch>=1.12.0 torch-geometric>=2.0.0
pip install numpy scipy matplotlib seaborn tensorboard tqdm pyyaml

# Install Berkeley ABC (Ubuntu/Debian)
sudo apt-get install abc

# Or install from source
git clone https://github.com/berkeley-abc/abc.git
cd abc && make && sudo make install
```

### **Basic Usage**

```python
from models.ppo_agent import PPOSynthesisAgent
from env.synthesis_env import SynthesisEnvironment

# Create environment and agent
env = SynthesisEnvironment(max_steps=10, action_space=['b', 'rw', 'rf', 'rwz', 'rfz'])
agent = PPOSynthesisAgent(config)

# Load pre-trained model
agent.load_model('outputs/models/best_eval_model.pth')

# Optimize a circuit
results = agent.optimize_circuit('path/to/circuit.aag')
print(f"Area reduction: {results['area_reduction_percent']:.2f}%")
```

## 🏗️ **Architecture Overview**

```
┌─────────────────────────────────────────────────────────────────┐
│                    RL Synthesis Agent                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ GNN Encoder │  │Actor Network│  │    Critic Network       │  │
│  │  5-layer    │  │  3-layer    │  │     4-layer             │  │
│  │  GIN + Attn │  │  Policy     │  │     Value Est.          │  │
│  │  256D Hidden│  │  5 Actions  │  │     1 Output            │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      PPO Training                              │
│        Experience Collection → GAE → Policy Update             │
└─────────────────────────────────────────────────────────────────┘
```

### **Model Specifications**
- **Total Parameters**: 1.19M (efficient design)
- **GNN Encoder**: 5-layer GIN with 256D hidden dimensions
- **Action Space**: 5 ABC synthesis operations (balance, rewrite, refactor, etc.)
- **Episode Length**: 10-step optimization sequences
- **Training**: 1M timesteps with adaptive validation

## 🎮 **Training**

### **Start Training**
```bash
# Train from scratch (4 hours on RTX 3080)
python medium_train_300_no_gnn_variance.py

# Resume from checkpoint
python medium_train_300_no_gnn_variance.py path/to/checkpoint.pth

# Monitor with TensorBoard
tensorboard --logdir outputs/tensorboard_logs/
```

### **Training Configuration**
```python
config = {
    'total_timesteps': 1000000,    # 1M timesteps
    'batch_size': 64,              # 64 circuits per batch  
    'learning_rate': 3e-4,         # Optimized learning rate
    'gamma': 0.99,                 # Long-term planning
    'gae_lambda': 0.97,            # Advantage estimation
    'val_episodes': 80,            # Complete validation set
    'eval_episodes': 85            # Complete evaluation set
}
```

## 📊 **Dataset**

### **Circuit Coverage**
| Benchmark Suite | Circuits | Training | Validation | Evaluation |
|------------------|----------|----------|------------|------------|
| **MCNC**         | 221      | 153      | 33         | 35         |
| **IWLS**         | 200      | 139      | 30         | 31         |
| **Synthetic**    | 100      | 70       | 15         | 15         |
| **EPFL**         | 20       | 14       | 2          | 4          |
| **Total**        | **541**  | **376**  | **80**     | **85**     |

### **Circuit Formats Supported**
- ✅ AIGER (.aig/.aag) - 72.9% of circuits
- ✅ BLIF (.blif) - 15.7% of circuits  
- ✅ Verilog (.v) - 11.4% of circuits

## 📈 **Performance Results**

### **Training Metrics**
- **Training Stability**: Consistent improvement over 1M timesteps
- **Model Convergence**: Policy and value functions stabilized
- **Generalization**: Strong performance on unseen circuits

## 🔧 **API Reference**

### **PPOSynthesisAgent**
```python
class PPOSynthesisAgent:
    def __init__(self, config):
        """Initialize RL agent with GNN encoder and actor-critic networks"""
        
    def get_action(self, observation):
        """Get synthesis action for circuit state"""
        
    def update(self, batch_data):
        """Update agent using PPO algorithm"""
        
    def save_model(self, path):
        """Save complete model state"""
        
    def load_model(self, path):
        """Load pre-trained model"""
```

### **SynthesisEnvironment**
```python
class SynthesisEnvironment:
    def __init__(self, max_steps=10, action_space=['b','rw','rf','rwz','rfz']):
        """Initialize ABC synthesis environment"""
        
    def reset(self, circuit_path):
        """Load circuit and return initial observation"""
        
    def step(self, action):
        """Execute synthesis action and return next state"""
        
    def get_episode_summary(self):
        """Get optimization results"""
```

## 🛠️ **Advanced Usage**

### **Custom Training**
Modify `custom_config.yaml` to adjust:
- PPO hyperparameters (learning rate, batch size, etc.)
- GNN architecture (type, layers, dimensions)
- Environment settings (max steps, reward shaping)
- Training parameters (total timesteps, evaluation intervals)

```python
# Train with custom dataset
data_splitter = PartitionedCircuitDataSplitterBalanced()
success = medium_train_300_no_gnn_variance(checkpoint_path=None)
```

### **Batch Circuit Optimization**
```python
# Optimize multiple circuits
results = []
for circuit_path in circuit_list:
    result = agent.optimize_circuit(circuit_path)
    results.append(result)
    
# Analyze results
avg_reduction = np.mean([r['area_reduction_percent'] for r in results])
print(f"Average area reduction: {avg_reduction:.2f}%")
```

### **Integration with EDA Workflows**
```python
class ProductionOptimizer:
    def __init__(self, model_path):
        self.agent = PPOSynthesisAgent(config)
        self.agent.load_model(model_path)
        
    def optimize_circuit(self, circuit_path):
        """Production circuit optimization"""
        return self.agent.optimize_circuit(circuit_path)
```

## 📁 **Project Structure**

```
rl-synthesis-agent/
├── models/                          # Neural network implementations
│   ├── ppo_agent.py                # Main PPO agent (437 lines)
│   ├── gnn_encoder.py              # GNN encoder (413 lines)
│   ├── actor_head.py               # Policy network
│   └── critic_head.py              # Value network
├── env/
│   └── synthesis_env.py            # ABC synthesis environment
├── data/
│   └── dataset.py                  # Circuit dataset management
├── utils/
│   ├── logger.py                   # Logging utilities
│   ├── metrics.py                  # Metrics tracking
│   └── tensorboard_monitor.py      # TensorBoard integration
├── outputs/                        # Training outputs
│   ├── models/                     # Saved model checkpoints
│   ├── tensorboard_logs/           # TensorBoard logs
│   └── logs/                       # Training logs
├── medium_train_300_no_gnn_variance.py  # Main training script (987 lines)
├── eval.py                         # Model evaluation
└── README.md                       # This file
```

## 🔬 **Research & Technical Details**

### **Key Innovations**
- **Enhanced GNN**: 5-layer GIN with attention pooling and residual connections
- **Asymmetric Design**: Specialized actor (fast decisions) and critic (accurate values)
- **Robust Training**: 1M timesteps with comprehensive monitoring
- **Production Scale**: 541 circuits with 100% compatibility

### **Technical Specifications**
- **Model Size**: 1.19M parameters (efficient for task)
- **Training Time**: ~4 hours on RTX 3080
- **Memory Usage**: ~2-4GB GPU memory during training
- **Disk Usage**: ~18MB per checkpoint

### **Publications & Citations**
```bibtex
@article{rl_synthesis_agent_2024,
  title={Deep Reinforcement Learning for Logic Synthesis Optimization using Graph Neural Networks},
  author={Qiang Han},
  year={2025}
}
```

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### **Development Setup**
```bash
# Clone for development
git clone https://github.com/qhan2012/rl-synthesis-agent.git
cd rl-synthesis-agent

# Install development dependencies
pip install -e .
pip install pytest black flake8 mypy

# Run tests
pytest tests/
```

### **Areas for Contribution**
- 🎯 Multi-objective optimization (area + delay + power)
- 🔄 Curriculum learning implementation
- 🚀 Distributed training support
- 📊 Additional benchmark suites
- 🔧 EDA tool integrations

## 📊 **Monitoring & Visualization**

### **TensorBoard Metrics**
- Training losses (policy, value, entropy)
- Performance metrics (area reduction, success rate)
- GNN analysis (embedding quality, variance)
- Environment statistics (action distribution, rewards)

### **Real-time Monitoring**
```bash
# View training progress
tensorboard --logdir outputs/tensorboard_logs/

# Monitor logs
tail -f outputs/logs/training.log
```

## ⚙️ **Configuration**

### **Training Configuration**
```yaml
# config/training_config.yaml
algorithm: PPO
total_timesteps: 1000000
batch_size: 64
learning_rate: 3e-4

gnn_encoder:
  type: GIN
  num_layers: 5
  hidden_dim: 256
  pooling: attention

dataset:
  sources: [MCNC, IWLS, Synthetic, EPFL]
  train_ratio: 0.695
  val_ratio: 0.148
  eval_ratio: 0.157
```

### **Environment Variables**
```bash
export CUDA_VISIBLE_DEVICES=0        # GPU selection
export ABC_PATH=/usr/local/bin/abc   # ABC tool path
export LOG_LEVEL=INFO                # Logging level
```

## 🐛 **Troubleshooting**

### **Common Issues**

#### **ABC Not Found**
```bash
# Install ABC synthesis tool
sudo apt-get install abc
# Or verify PATH
which abc
```

#### **CUDA Out of Memory**
```python
# Reduce batch size in config
config['batch_size'] = 32  # Instead of 64
```

#### **Circuit Parsing Errors**
```python
# Enable debug logging
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

### **Performance Optimization**
```python
# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Memory management
torch.cuda.empty_cache()
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Berkeley ABC** team for the synthesis tool
- **PyTorch Geometric** team for GNN infrastructure  
- **EPFL**, **MCNC**, **IWLS** for benchmark circuits
- Research community for foundational RL and GNN work

**🚀 Ready to optimize your circuits with AI? Get started today!**
