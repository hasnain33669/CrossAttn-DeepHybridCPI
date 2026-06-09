# CrossAttn-DeepHybridCPI
# CrossAttn-DeepHybridCPI

**Bidirectional Cross-Attention Deep Hybrid Model for Compound-Protein Interaction Prediction**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

CrossAttn-DeepHybridCPI is a deep learning framework for predicting compound-protein interactions (CPI). The model integrates:

- **Multi-scale Graph Neural Network (MGNN)** for compound molecular graph encoding
- **ESM-2** for protein sequence encoding
- **Bidirectional Cross-Attention** for interaction modeling between compounds and proteins

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU 

### Setup

# Create virtual environment
python -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt
## Usage
Training
bash
python scripts/train_model.py --data_path data.txt --epochs 100 --batch_size 16 --lr 1e-4
Evaluation
bash
python scripts/evaluate_model.py --model_path saved_models/best_model.pt --test_data data_test.csv

### Project Structure
text
CrossAttn-DeepHybridCPI/
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup script
├── LICENSE               # MIT License
├── .gitignore            # Git ignore file
├── config/
│   └── config.yaml       # Configuration file
├── src/
│   ├── __init__.py
│   ├── data_loader.py    # Data loading utilities
│   ├── model.py          # Model architecture
│   ├── train.py          # Training functions
│   ├── evaluate.py       # Evaluation metrics
│   └── utils.py          # Helper functions
├── notebooks/
│   └── CrossAttn_DeepHybridCPI_Training.ipynb
├── scripts/
│   ├── train_model.py    # Training script
│   ├── evaluate_model.py # Evaluation script
│   └── run_ablation.py   # Ablation study script
└── tests/
    ├── __init__.py
    └── test_model.py     # Unit tests

### Citation
CrossAttn-DeepHybridCPI: Bidirectional Cross-Attention Deep Hybrid Model for Compound-Protein Interaction Prediction
