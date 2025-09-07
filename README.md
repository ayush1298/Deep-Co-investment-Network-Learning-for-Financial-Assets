# Deep Co-investment Network Learning for Financial Assets

A PyTorch-based implementation of Deep Co-investment Network Learning (DeepCNL) for financial market analysis and stock price prediction. This project combines Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN/LSTM/GRU) to learn relationships between financial assets and construct co-investment networks.

## 🎯 Overview

This framework implements a novel approach to understanding financial asset relationships by:
- **Learning co-investment networks** from historical stock data using deep learning
- **Predicting stock price movements** with CRNN (Convolutional Recurrent Neural Networks)
- **Comparing multiple network construction methods** (DeepCNL, PCC, DTW, VWL)
- **Analyzing influential assets** and market dynamics through graph theory
- **Providing interactive web visualization** for network exploration

## 🏗️ Architecture

The system uses a **CRNN (Convolutional Recurrent Neural Network)** architecture that:

1. **Convolutional Layer**: Processes windowed time series data from asset pairs
2. **Recurrent Layer**: Captures temporal dependencies using LSTM/GRU/RNN
3. **Network Construction**: Extracts learned weights to build co-investment graphs
4. **Web Interface**: Interactive visualization and analysis platform

## ✨ Key Features

### 🤖 Deep Learning Models
- **CRNN_LSTM**: CNN + LSTM for temporal pattern learning ([`crnn_lstm.py`](crnn_lstm.py))
- **CRNN_RNN**: CNN + vanilla RNN implementation ([`crnn_rnn.py`](crnn_rnn.py))
- **CRNN_GRU**: CNN + GRU for efficient sequence modeling ([`crnn_gru.py`](crnn_gru.py))

### 🕸️ Network Construction Methods
- **DeepCNL**: Deep Co-investment Network Learning (primary method)
- **PCC**: Pearson Correlation Coefficient baseline
- **DTW**: Dynamic Time Warping similarity
- **VWL**: Visibility Graph + Weisfeiler-Lehman kernel

### 📊 Analysis Tools
- Market volatility and return analysis
- Influential asset identification
- Network density comparison across ETF portfolios
- Rise/fall prediction accuracy evaluation
- Interactive web-based network exploration

### 🌐 Web Application Features
- **Interactive Network Explorer**: Visualize co-investment networks with D3.js
- **Stock Rankings Dashboard**: Track top-performing stocks and prediction accuracy
- **Performance Analysis**: Compare DeepCNL against traditional methods
- **Real-time Visualization**: Dynamic charts and graphs for market analysis

## 📁 Project Structure

```
├── 📊 Core Analysis
│   ├── stock_network_analysis.py    # Main experimental platform
│   ├── data_util.py                # Data preprocessing utilities
│   ├── learner.py                  # Graph learning implementations
│   └── financial_index.py          # Financial metrics calculation
│
├── 🧠 Models  
│   ├── crnn.py                     # Base CRNN class
│   ├── crnn_factory.py            # Model factory for CRNN variants
│   ├── crnn_lstm.py               # LSTM implementation
│   ├── crnn_gru.py                # GRU implementation  
│   └── crnn_rnn.py                # RNN implementation
│
├── 🔧 Utilities
│   ├── wlkernel.py                # Weisfeiler-Lehman kernel
│   └── generate_web_data.py       # Sample data generator
│
├── 🌐 Web Application
│   ├── web_app/
│   │   ├── app.py                 # Flask backend
│   │   ├── templates/             # HTML templates
│   │   │   ├── base.html          # Base template
│   │   │   ├── index.html         # Landing page
│   │   │   ├── network.html       # Network explorer
│   │   │   ├── rankings.html      # Stock rankings
│   │   │   └── performance.html   # Performance analysis
│   │   ├── static/
│   │   │   ├── css/style.css      # Styling
│   │   │   └── js/                # JavaScript components
│   │   │       ├── network_viz.js # Network visualization
│   │   │       ├── rankings.js    # Rankings functionality
│   │   │       └── performance.js # Performance charts
│   │   └── data/                  # JSON data files
│
└── 📚 Documentation
    ├── README.md                   # Project documentation
    ├── INSTALL.md                 # Installation guide
    └── requirements.txt           # Python dependencies
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- PyTorch (CPU or GPU)
- Node.js (for web development)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ayush1298/Deep-Co-investment-Network-Learning-for-Financial-Assets.git
cd Deep-Co-investment-Network-Learning-for-Financial-Assets

# Create virtual environment
python -m venv deepcnl_env
source deepcnl_env/bin/activate  # On Windows: deepcnl_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

Update data paths in [`stock_network_analysis.py`](stock_network_analysis.py):
```python
DATA_PATH = "path/to/your/prices-split-adjusted.csv"
SPY_PATH = "path/to/your/SPY20000101_20171111.csv"
```

### 3. Run Core Analysis

```bash
# Run network learning experiments
python stock_network_analysis.py
```

### 4. Launch Web Application

```bash
# Generate sample data for web app
python generate_web_data.py

# Start web server
cd web_app
python app.py
```

Visit `http://localhost:5000` to explore the interactive interface!

## 📊 Key Performance Indicators

- **🎯 Predictive Hit Ratio**: 57.14% average for top-10 performing stocks (2010-2016)
- **💰 Financial Influence**: $223.1B average market cap of identified firms
- **📈 Benchmark Comparison**: Outperforms PCC baseline by identifying firms 3.3x larger

## 🔧 Configuration

Key hyperparameters in [`stock_network_analysis.py`](stock_network_analysis.py):

```python
# Model Configuration
LEARNER_CODE = 'DEEPCNL'        # Learning method: DEEPCNL, PCC, DTW, VWL
CRNN_CODE = 'CRNN_LSTM'         # Model: CRNN_LSTM, CRNN_GRU, CRNN_RNN

# Data Parameters
WINDOW = 32                     # Time series window size
FEATURE_NUM = 4                 # Features: high, low, volume, adj_close
TICKER_NUM = 470                # Number of stocks to analyze

# Training Parameters
FILTERS_NUM = 16                # CNN kernel number
HIDDEN_UNIT_NUM = 256           # RNN hidden units
EPOCH_NUM = 200                 # Training epochs
LR = 0.0005                     # Learning rate
DROPOUT = 0.35                  # Dropout rate

# Network Construction
RARE_RATIO = 0.002              # Edge density control
```

## 📈 Usage Examples

### Basic Network Learning
```python
from stock_network_analysis import Experimental_platform
from data_util import Data_util

# Initialize components
datatool = Data_util(TICKER_NUM, WINDOW, FEATURE_NUM, DATA_PATH, SPY_PATH)
experiment = Experimental_platform(datatool)

# Run experiment for year 2012
seed = 2
train_period, test_period = experiment.period_generator(seed)
train_x = datatool.load_x(train_period)
train_y = datatool.load_y(train_period)

# Learn co-investment network
network = experiment.deep_CNL('igo', train_x, train_y, RARE_RATIO)
```

### Comprehensive Analysis
```python
# Stock prediction accuracy
experiment.rise_fall_prediction(seed=2)

# Network density analysis across ETFs
experiment.ALL_density_comparison(seed=2)

# Coverage comparison with market benchmarks
experiment.coverage_comparison()
```

## 🌐 Web Interface

The web application provides four main pages:

1. **📋 Project Overview**: Key metrics and project summary
2. **🕸️ Network Explorer**: Interactive graph visualization with search and filtering
3. **📊 Stock Rankings**: Performance tracking and prediction validation
4. **📈 Performance Analysis**: Comparative analysis with traditional methods

### API Endpoints
- `GET /api/network/<year>`: Retrieve network data for specific year
- `GET /api/rankings/<year>`: Get stock rankings and predictions
- `GET /api/performance`: Performance comparison metrics
- `GET /api/search_stock`: Search for specific stocks in networks

## 🔬 Research Applications

This codebase supports research in:
- **Financial Network Analysis**: Understanding asset correlations and market structure
- **Algorithmic Trading**: Network-based feature engineering for trading strategies
- **Risk Management**: Systemic risk identification through network topology
- **Portfolio Optimization**: Asset selection based on co-investment relationships
- **Market Microstructure**: High-frequency relationship analysis

## 📊 Benchmark Datasets

The system evaluates against standard financial indices:
- **XLG**: Top 50 S&P 500 stocks (Guggenheim S&P 500 Top 50 ETF)
- **OEX**: S&P 100 index (iShares S&P 100 ETF)
- **IWL**: Russell Top 200 (iShares Russell Top 200 ETF)

## 🛠️ Development

### CPU vs GPU Support
The code automatically detects and uses GPU when available:
```python
# Automatic device detection
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Model Architecture Details
- **Input**: Pairwise time series combinations (470 stocks → 110,265 pairs)
- **CNN**: 1D convolution for feature extraction from windowed data
- **RNN**: LSTM/GRU/RNN for temporal pattern learning
- **Output**: Binary classification (rise/fall prediction)

### Training Features
- **Regularization**: L2 regularization on linear layers
- **Optimization**: Adam optimizer with learning rate scheduling
- **Initialization**: Xavier/Glorot initialization for stable training
- **Batch Normalization**: Improved convergence and stability

## 🎯 Performance Notes

- **Memory Requirements**: ~8GB GPU memory for 470 tickers
- **Training Time**: 10-30 minutes for 200 epochs (GPU) / 2-4 hours (CPU)
- **Accuracy**: Competitive performance vs traditional correlation methods
- **Scalability**: Supports 100-500 assets with current implementation

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Additional Baseline Methods**: Implement more network construction algorithms
2. **Performance Optimization**: GPU memory efficiency and training speed
3. **Web Interface Enhancement**: Additional visualization features
4. **Documentation**: Code examples and tutorials
5. **Testing**: Unit tests and integration tests

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest jupyter black flake8

# Run tests
pytest tests/

# Code formatting
black *.py
```