# Deep Co-investment Network Learning for Financial Assets

A PyTorch-based implementation of Deep Co-investment Network Learning (DeepCNL) for financial market analysis and stock price prediction. This project combines Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN/LSTM/GRU) to learn relationships between financial assets and construct co-investment networks.

## Overview

This framework implements a novel approach to understanding financial asset relationships by:
- Learning co-investment networks from historical stock data
- Predicting stock price movements using deep learning
- Comparing multiple network construction methods
- Analyzing influential assets and market dynamics

## Architecture

The system uses a **CRNN (Convolutional Recurrent Neural Network)** architecture that:
1. **Convolutional Layer**: Processes windowed time series data from asset pairs
2. **Recurrent Layer**: Captures temporal dependencies using LSTM/GRU/RNN
3. **Network Construction**: Extracts learned weights to build co-investment graphs

## Key Features

### Deep Learning Models
- **CRNN_LSTM**: CNN + LSTM for temporal pattern learning
- **CRNN_RNN**: CNN + vanilla RNN implementation  
- **CRNN_GRU**: CNN + GRU for efficient sequence modeling

### Network Construction Methods
- **DeepCNL**: Deep Co-investment Network Learning (primary method)
- **PCC**: Pearson Correlation Coefficient baseline
- **DTW**: Dynamic Time Warping similarity
- **VWL**: Visibility Graph + Weisfeiler-Lehman kernel

### Analysis Tools
- Market volatility and return analysis
- Influential asset identification
- Network density comparison across ETF portfolios
- Rise/fall prediction accuracy evaluation

## Project Structure

```
├── stock_network_analysis.py    # Main experimental platform
├── data_util.py                # Data preprocessing and utilities
├── learner.py                  # Graph learning implementations
├── crnn_factory.py            # Model factory for CRNN variants
├── crnn.py                    # Base CRNN class
├── crnn_lstm.py               # LSTM implementation
├── crnn_gru.py                # GRU implementation  
├── crnn_rnn.py                # RNN implementation
├── wlkernel.py                # Weisfeiler-Lehman kernel
└── financial_index.py         # Financial metrics calculation
```

## Installation

### Requirements
```bash
pip install torch torchvision
pip install networkx pandas numpy scipy scikit-learn
pip install pandas-datareader
pip install fastdtw
pip install visibility-graph
```

### CUDA Support
The code is optimized for GPU training. Ensure you have CUDA-compatible PyTorch installed:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Configuration

Key hyperparameters in [`stock_network_analysis.py`](stock_network_analysis.py):

```python
LEARNER_CODE = 'DEEPCNL'        # Learning method
CRNN_CODE = 'CRNN_LSTM'         # Model architecture
WINDOW = 32                     # Time series window size
FEATURE_NUM = 4                 # Features (high, low, volume, adj_close)
FILTERS_NUM = 16                # CNN kernel number
TICKER_NUM = 470                # Number of stocks to analyze
HIDDEN_UNIT_NUM = 256           # RNN hidden units
EPOCH_NUM = 200                 # Training epochs
LR = 0.0005                     # Learning rate
```

## Data Format

The system expects CSV data with columns:
- `symbol`: Stock ticker
- `date`: Trading date
- `high`, `low`, `volume`: Market data
- `close`: Closing price (converted to adj_close)

## Usage

### Basic Network Learning
```python
from stock_network_analysis import Experimental_platform
from data_util import Data_util

# Initialize data utility
datatool = Data_util(TICKER_NUM, WINDOW, FEATURE_NUM, DATA_PATH, SPY_PATH)

# Create experimental platform
experiment = Experimental_platform(datatool)

# Run network learning for a specific year
seed = 2  # Corresponds to year 2012
train_period, test_period = experiment.period_generator(seed)
train_x = datatool.load_x(train_period)
train_y = datatool.load_y(train_period)

# Learn co-investment network
network = experiment.deep_CNL('igo', train_x, train_y, RARE_RATIO)
```

### Running Experiments
```python
# Rise/fall prediction
experiment.rise_fall_prediction(seed=2)

# Network density analysis
experiment.ALL_density_comparison(seed=2)

# Coverage comparison with market benchmarks
experiment.coverage_comparison()
```

## Key Components

### Data Utility ([`data_util.py`](data_util.py))
- **Data Loading**: Reads and preprocesses financial CSV data
- **Normalization**: Min-max scaling for features
- **Time Series Enumeration**: Creates pairwise combinations for network learning
- **Period Management**: Handles train/test data splitting

### CRNN Models
- **Base Class** ([`crnn.py`](crnn.py)): Abstract CRNN implementation
- **LSTM Variant** ([`crnn_lstm.py`](crnn_lstm.py)): Uses LSTM for sequence modeling
- **GRU Variant** ([`crnn_gru.py`](crnn_gru.py)): More efficient GRU implementation
- **RNN Variant** ([`crnn_rnn.py`](crnn_rnn.py)): Simple RNN baseline

### Network Construction
The [`Experimental_platform`](stock_network_analysis.py) class provides methods for:
- **DeepCNL**: Extracts learned LSTM weights to construct networks
- **Baseline Methods**: Implements PCC, DTW, and VWL for comparison
- **Analysis Tools**: Network metrics and visualization

## Benchmarks

The system evaluates against standard financial indices:
- **XLG**: Top 50 S&P 500 stocks
- **OEX**: S&P 100 index
- **IWL**: Russell Top 200

## Model Training

The training process:
1. **Data Preparation**: Load and normalize time series data
2. **Network Architecture**: Initialize CRNN model
3. **Training Loop**: 200 epochs with Adam optimizer
4. **Regularization**: L2 regularization on linear layers
5. **Loss Function**: CrossEntropyLoss for classification

### Training Features
- **GPU Acceleration**: CUDA support for faster training
- **Batch Normalization**: Improves convergence
- **Dropout**: Prevents overfitting (35% default)
- **Xavier Initialization**: Better weight initialization

## Research Applications

This codebase supports research in:
- **Financial Network Analysis**: Understanding asset correlations
- **Market Prediction**: Rise/fall classification
- **Systemic Risk**: Identifying influential assets
- **Portfolio Management**: Network-based asset selection

## Performance Notes

- **Memory Requirements**: Large ticker numbers (470+) require substantial GPU memory
- **Training Time**: ~200 epochs typically require 10-30 minutes on modern GPUs
- **Accuracy**: Achieves competitive performance compared to traditional methods