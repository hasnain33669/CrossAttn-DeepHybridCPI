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
### requirements.txt

```txt
torch>=2.0.0
torch-geometric>=2.3.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
tqdm>=4.65.0
rdkit>=2023.3.0
fair-esm>=2.0.0
transformers>=4.30.0
umap-learn>=0.5.5
pyyaml>=6.0

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
├── README.md              
├── requirements.txt       
├── setup.py              
├── LICENSE               
├── .gitignore            
├── config/
│   └── config.yaml       
├── src/
│   ├── __init__.py
│   ├── data_loader.py    
│   ├── model.py          
│   ├── train.py          
│   ├── evaluate.py       
│   └── utils.py          
├── notebooks/
│   └── CrossAttn_DeepHybridCPI_Training.ipynb
├── scripts/
│   ├── train_model.py    
│   ├── evaluate_model.py 
│   └── run_ablation.py   
└── tests/
    ├── __init__.py
    └── test_model.py     

### Citation
CrossAttn-DeepHybridCPI: Bidirectional Cross-Attention Deep Hybrid Model for Compound-Protein Interaction Prediction
