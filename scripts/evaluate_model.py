import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from src import (
    load_cpi_data, split_data, CrossAttnCPIDataset, CrossAttnDeepHybridCPI,
    evaluate, evaluate_with_bootstrap, set_seed, collate_fn, generate_esm2_embeddings
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate CrossAttn-DeepHybridCPI model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('--data_path', type=str, default='data.txt', help='Path to data file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--model_name', type=str, default='CrossAttnDeepHybridCPI', help='Model name')
    parser.add_argument('--dataset_name', type=str, default='dataset', help='Dataset name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_bootstrap', action='store_true', help='Skip bootstrapping')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    df = load_cpi_data(args.data_path)
    if df is None:
        return

    train_df, val_df, test_df = split_data(df)

    # Generate protein embeddings (or load from cache)
    all_sequences = set(train_df['target_sequence'].values) | set(val_df['target_sequence'].values) | set(test_df['target_sequence'].values)

    cache_path = 'esm2_embeddings_cache.pt'
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        protein_embeddings = torch.load(cache_path)
    else:
        protein_embeddings = generate_esm2_embeddings(list(all_sequences), device=device)
        torch.save(protein_embeddings, cache_path)

    # Create test dataset
    test_dataset = CrossAttnCPIDataset('./data', types='test', protein_embeddings=protein_embeddings)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Load model
    model = CrossAttnDeepHybridCPI().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # Evaluate
    test_loss, test_acc, test_prec, test_rec, test_auc, _, _, _ = evaluate(
        model, test_loader, device, criterion
    )

    test_f1 = 2 * (test_prec * test_rec) / (test_prec + test_rec) if (test_prec + test_rec) > 0 else 0.0

    print("\n" + "="*60)
    print("Test Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall: {test_rec:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    print(f"  F1: {test_f1:.4f}")
    print("="*60)

    # Bootstrapping
    if not args.no_bootstrap:
        bootstrap_results = evaluate_with_bootstrap(
            model, test_dataset, device, criterion,
            n_runs=5, subsample_ratio=0.8, batch_size=args.batch_size
        )

        print("\n" + "="*60)
        print("Test Results with Standard Deviation (Bootstrapping):")
        print("="*60)
        print(f"accuracy: {bootstrap_results['accuracy']}")
        print(f"auc_roc: {bootstrap_results['auc']}")
        print(f"precision: {bootstrap_results['precision']}")
        print(f"recall: {bootstrap_results['recall']}")
        print(f"f1: {bootstrap_results['f1']}")
        print("="*60)


if __name__ == "__main__":
    import os
    main()
