# Learning to Rank with Top-K Fairness

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.4+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-10.2+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

This repository contains the implementation of fairness-aware learning-to-rank algorithms that jointly optimize ranking quality (NDCG) and group fairness in top-K positions. The framework extends state-of-the-art NDCG optimization techniques with novel fairness constraints to address bias in recommendation and ranking systems.


## Key Contributions

### 1. Fairness-Aware Loss Functions
We implement several novel fairness-aware ranking losses that extend classical ranking objectives:

- **Simple Fairness Loss**: A foundational fairness penalty that measures exposure disparity between groups using softmax-normalized predictions
- **ListNet Fair Loss**: Extends ListNet with group fairness constraints, balancing cross-entropy ranking loss with fairness penalties
- **ListMLE Fair Loss**: Incorporates fairness into the ListMLE framework for probabilistic ranking
- **BPR Fair Loss**: Augments Bayesian Personalized Ranking with fairness regularization
- **Listwise Cross-Entropy Fair Loss**: Combines listwise cross-entropy with moving average fairness estimation
- **NDCG Fair Loss**: Our main contribution - integrates fairness directly into NDCG optimization with multiple variants:
  - `exp_top1_fair`: Exponential fairness with top-1 probability
  - `exp_topk`: Top-K constrained exponential fairness
  - `rank_fair`: Position-based fairness using ranking utilities
  - `log_rank_fair`: Logarithmic discounting for position-based fairness

### 2. Advanced Optimization Techniques

- **Stochastic Gradient Estimation**: Uses exponential moving averages (γ parameters) to estimate fairness gradients efficiently
- **Top-K Threshold Learning**: Adaptive threshold optimization for constraining fairness to top-K positions using learnable thresholds
- **Bidirectional Fairness**: Option to penalize both over and under-representation (`use_balanced_fairness` mode)
- **Multi-scale Fairness**: Different fairness types operating at various granularities (item-level, rank-level, score-level)
- **Second-order Optimization**: Hessian approximation for improved convergence in theoretical top-K version

### 3. Comprehensive Evaluation Framework

- **Fairness Metrics**:
  - `exp_norm`: Exponential normalization-based fairness (default)
  - `sigmoid_thresh`: Sigmoid with top-K threshold
  - `ndcg_diff`: NDCG difference between groups
  - `rank_topk`: Average rank disparity in top-K
  - `log_rank_topk`: Logarithmic rank disparity


## Installation

### Requirements
- Python 3.7+ (tested with 3.7.11)
- PyTorch 1.4+ (tested with 1.4.0)
- CUDA 10.2+ (optional, for GPU acceleration)
- NumPy 1.21+
- Pandas 1.3+
- scikit-learn 1.0+
- einops 0.4+

### Setup
```bash
git clone https://github.com/boyang-zhang1/NDCG-fairness-opt.git
cd NDCG-fairness-opt

# Create environment from the provided yml file
conda env create -f ndcg_env.yml
conda activate pytorch14
```

## Quick Start

```bash
# Navigate to the source directory
cd RS_exp/src

# Run with default settings (NDCG loss with fairness)
python main.py --dataset ml-20m --model NeuMF --loss_type NDCG --fairness_c 0

# Run with different fairness types
python main.py --dataset ml-20m --model NeuMF --loss_type NDCG --fair_type exp_top1_fair --fairness_c 10000
```

## Dataset Preparation

The framework supports multiple datasets with sensitive attributes:

### Supported Datasets
- **MovieLens-20M (ml-20m)**:
  - Sensitive attributes: Movie genres (Action, Crime, etc.) or User demographics (Gender, Age)
- **Netflix Prize**:
  - Sensitive attributes: Movie year, genres

### Data Format
```
data/
├── [dataset_name]/
│   ├── train.csv       # Training interactions (user_id, item_id, rating, timestamp)
│   ├── dev.csv         # Validation interactions
│   ├── test.csv        # Test interactions
│   ├── movies.dat      # Item attributes with sensitive features
│   ├── users.dat       # User attributes (optional)
│   └── sensitive.json  # Sensitive attribute mappings
```

### Data Format Details
- **Interaction files (train/dev/test.csv)**: `user_id,item_id,rating,timestamp`
- **Item attributes (movies.dat)**: Contains item features including sensitive attributes
- **Sensitive attributes**: Binary or categorical features for fairness evaluation

## Code Structure

```
NDCG-fairness-opt/
├── RS_exp/
│   └── src/
│       ├── main.py                 # Main training script
│       ├── exp.py                  # Experiment configurations
│       ├── models/
│       │   ├── ndcg_loss.py       # Fairness-aware loss implementations
│       │   ├── BaseModel.py       # Base model with fairness support
│       │   ├── general/           # Model architectures (NeuMF, MF, etc.)
│       │   ├── sequential/        # Sequential models (GRU4Rec, SASRec)
│       │   └── developing/        # Experimental models
│       ├── helpers/
│       │   ├── BaseReader.py      # Data loading with sensitive attributes
│       │   └── BaseRunner.py      # Training loop and fairness evaluation
│       └── utils/
│           └── utils.py           # Utility functions
├── data/                          # Dataset directory
├── results/                       # Experimental results
└── README.md
```

## Evaluation

The framework provides comprehensive evaluation metrics:

### Running Evaluation
```bash
# Evaluate trained model
python main.py --dataset ml-20m --model NeuMF --load_model path/to/model.pth --test_only
```

### Key Parameters
- `--fairness_c`: Fairness weight
- `--fair_type`: Fairness type (`exp_top1_fair`, `rank_fair`, etc.)
- `--ndcg_topk`: Top-K positions (-1=all items)
- `--balance_fair`: Bidirectional fairness
- `--simple_fair`: Simplified fairness

## Acknowledgments

This work builds upon:
- [NDCG Optimization](https://arxiv.org/abs/2202.12183): Original NDCG optimization framework

