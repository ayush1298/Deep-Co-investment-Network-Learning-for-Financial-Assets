ushmunot/Deep-Co-investment-Network-Learning-for-Financial-Assets/INSTALL.md
# Installation Guide

## Prerequisites

- Python 3.7 or higher
- CUDA 11.0+ (for GPU training)
- Git

## Step 1: Clone the Repository

```bash
git clone https://github.com/ayush1298/Deep-Co-investment-Network-Learning-for-Financial-Assets.git
cd Deep-Co-investment-Network-Learning-for-Financial-Assets
```

## Step 2: Create Virtual Environment

```bash
python -m venv deepcnl_env
source deepcnl_env/bin/activate  # On Windows: deepcnl_env\Scripts\activate
```

## Step 3: Install PyTorch (with CUDA support)

For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU only:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Step 4: Install Other Dependencies

```bash
pip install -r requirements.txt
```

## Step 5: Install Optional Dependencies

For visibility graph analysis:
```bash
pip install visibility-graph
```

For DTW analysis:
```bash
pip install fastdtw
```

## Step 6: Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 7: Set Up Data Paths

Update the data paths in `stock_network_analysis.py`:
```python
DATA_PATH = "path/to/your/prices-split-adjusted.csv"
SPY_PATH = "path/to/your/SPY20000101_20171111.csv"
```

## Step 8: Run Web Application

```bash
cd web_app
python app.py
```

Visit `http://localhost:5000` to access the web interface.

## Troubleshooting

### CUDA Issues
- Ensure NVIDIA drivers are up to date
- Check CUDA compatibility with your GPU
- Use CPU version if GPU training is not available

### Import Errors
- Verify all dependencies are installed: `pip list`
- Check Python version compatibility
- Ensure virtual environment is activated

### Memory Issues
- Reduce `TICKER_NUM` parameter for smaller datasets
- Use smaller batch sizes
- Enable gradient checkpointing for large models