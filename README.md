# Learning to Rank with Top-K Fairness

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.4+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-10.2+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

Implementation of learning-to-rank algorithms that optimize both ranking quality (NDCG) and fairness in top-K positions.

## Installation

### Requirements
- Python 3.7+ (tested with 3.7.11)
- PyTorch 1.4+ (tested with 1.4.0)
- CUDA 10.2+ (optional)
- NumPy 1.21
- Pandas 1.3

### Quick Setup
```bash
git clone https://github.com/boyang-zhang1/NDCG-fairness-opt.git
cd NDCG-fairness-opt

# Create environment from the provided yml file
conda env create -f ndcg_env.yml
conda activate pytorch14
```

## Running Different Methods

The framework implements several methods mentioned in the paper. Navigate to `RS_exp/src` and use the following commands:

```bash
# KSO-RED: Our main contribution - top-K fairness optimization
python main.py --dataset nf-20m --sensitive_types 1990 --attribute movies_year --ndcg_topk 300 --fair_type exp_top1_fair_topk --fairness_c 10000

# SO-RED: Stochastic optimization for ranking exposure disparity (baseline version)
python main.py --dataset nf-20m --sensitive_types 1990 --attribute movies_year --ndcg_topk 300 --fairness_c 10000

# NG-DE: Integration of NDCG ranking loss and disparate exposure
python main.py --dataset nf-20m --sensitive_types 1990 --attribute movies_year --ndcg_topk 300 --simple_fair --fairness_c 10000

# DELTR: ListNet-based fair ranking baseline
python main.py --dataset nf-20m --sensitive_types 1990 --attribute movies_year --fairness_c 10000

# K-SONG: Color-blind ranking (no fairness constraints)
python main.py --dataset nf-20m --sensitive_types 1990 --attribute movies_year --ndcg_topk 300 --fairness_c 0
```

### Method Descriptions:
- **KSO-RED**: Our proposed top-K fairness optimization algorithm
- **SO-RED**: Exposure disparity optimization without top-K constraints
- **NG-DE**: Baseline combining NDCG and disparate exposure
- **DELTR**: ListNet-based fairness-aware ranking
- **K-SONG**: Standard ranking without fairness considerations

## Dataset Preparation

The framework supports recommendation datasets with sensitive group attributes for fairness evaluation:

### Supported Datasets
- **MovieLens-20M**: Comprises 20 million ratings from 138,000 users across 27,000 movies
- **Netflix Prize**: Random subset of 20 million ratings for 17,770 movies from the original dataset

### Derived Sensitive Group Datasets
To evaluate fairness between protected and non-protected groups, we derive three subsets:
- **MovieLens-20M-H**: horror vs. non-horror movies
- **MovieLens-20M-D**: documentary vs. non-documentary movies
- **Netflix-20M**: movies before 1990 vs. from 1990 onwards

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

The framework evaluates both ranking accuracy and fairness:

### Evaluation Metrics
- **Accuracy**: NDCG (Normalized Discounted Cumulative Gain) at top-K positions
- **Fairness**: Top-K exposure disparity measured as Mean Absolute Error (MAE) and Mean Squared Error (MSE) of the difference between averaged exposures of minority and majority groups

### Running Evaluation
```bash
# Evaluate trained model
python main.py --dataset ml-20m --model NeuMF --load_model path/to/model.pth --test_only
```

### Key Parameters
- `--fairness_c`: Fairness weight parameter C that balances ranking quality and fairness (C=0 for color-blind ranking)
- `--ndcg_topk`: Top-K positions for evaluation

## Acknowledgments

This work builds upon:
- K-SONG framework for stochastic optimization of NDCG surrogates
- Equal exposure fairness definitions from prior learning-to-rank literature
- Bilevel finite-sum coupled compositional optimization techniques

