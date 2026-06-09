import os
import argparse
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src import (
    load_cpi_data, split_data, CrossAttnCPIDataset, CrossAttnDeepHybridCPI,
    train_model, set_seed, collate_fn, generate_esm2_embeddings
)


def main():
    parser = argparse.ArgumentParser(description='Train CrossAttn-DeepHybridCPI model')
    parser.add_argument('--data_path', type=str, default='data.txt', help='Path to data file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='CrossAttnDeepHybridCPI', help='Model name')
    parser.add_argument('--dataset_name', type=str, default='dataset', help='Dataset name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # Load data
    df = load_cpi_data(args.data_path)
    if df is None:
        return

    train_df, val_df, test_df = split_data(df)

    os.makedirs('data/raw', exist_ok=True)
    train_df.to_csv('data/raw/data_train.csv', index=False)
    val_df.to_csv('data/raw/data_val.csv', index=False)
    test_df.to_csv('data/raw/data_test.csv', index=False)

    # Generate protein embeddings
    all_sequences = set(train_df['target_sequence'].values) | set(val_df['target_sequence'].values) | set(test_df['target_sequence'].values)
    print(f"Unique protein sequences: {len(all_sequences)}")

    cache_path = 'esm2_embeddings_cache.pt'
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        protein_embeddings = torch.load(cache_path)
    else:
        protein_embeddings = generate_esm2_embeddings(list(all_sequences), device=device)
        torch.save(protein_embeddings, cache_path)
        print(f"Saved embeddings to {cache_path}")

    # Create datasets and loaders
    train_dataset = CrossAttnCPIDataset('./data', types='train', protein_embeddings=protein_embeddings)
    val_dataset = CrossAttnCPIDataset('./data', types='val', protein_embeddings=protein_embeddings)
    test_dataset = CrossAttnCPIDataset('./data', types='test', protein_embeddings=protein_embeddings)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = CrossAttnDeepHybridCPI().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train model
    model, train_losses, train_accs, val_accs, val_aucs, metrics_df = train_model(
        model, train_loader, val_loader, optimizer, criterion, device,
        epochs=args.epochs, model_name=args.model_name, dataset_name=args.dataset_name
    )

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
