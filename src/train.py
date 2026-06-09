
import time
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from src.utils import AverageMeter, collate_fn


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train model for one epoch."""
    model.train()
    running_loss = AverageMeter()
    all_preds = []
    all_labels = []

    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)

        labels = batch.y.view(-1)
        logits = model(batch.x, batch.edge_index, batch.batch, batch.protein_embedding)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.update(loss.item(), labels.size(0))
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(np.array(all_labels), np.array(all_preds))
    return running_loss.avg, train_acc


def train_model(model, train_loader, val_loader, optimizer, criterion, device,
                epochs=50, model_name="CrossAttnDeepHybridCPI", dataset_name="dataset",
                save_dir="saved_models"):
    """Train model with checkpoint saving."""
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    val_aucs = []
    times = []
    total_start_time = time.time()

    os.makedirs(save_dir, exist_ok=True)
    model_path = f'{save_dir}/best_model_{model_name}_{dataset_name}.pt'

    best_val_acc = 0.0

    print("=" * 100)
    print(f"{'Epoch':<8} {'Time(sec)':<12} {'Train_Loss':<12} {'Train_Acc':<12} {'Val_Loss':<12} {'Val_Acc':<12}")
    print("=" * 100)

    for epoch in range(epochs):
        epoch_start_time = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_prec, val_rec, val_auc, _, _, _ = evaluate(
            model, val_loader, device, criterion
        )

        epoch_time = time.time() - epoch_start_time

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_aucs.append(val_auc)
        times.append(epoch_time)

        print(f"{epoch+1:<8} {epoch_time:<12.6f} {train_loss:<12.6f} {train_acc:<12.6f} {val_loss:<12.6f} {val_acc:<12.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"  ✅ Best model saved! (Val Acc: {val_acc:.4f})")

    total_time = time.time() - total_start_time
    print("=" * 100)
    print(f"Training finished! Completed all {epochs} epochs")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("=" * 100)

    model.load_state_dict(torch.load(model_path))
    print(f"✅ Loaded best model from {model_path}")

    metrics_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Time(sec)': times,
        'Train_Loss': train_losses,
        'Train_Accuracy': train_accs,
        'Validation_Loss': val_losses,
        'Validation_Accuracy': val_accs
    })

    csv_filename = f'{model_name}_{dataset_name}_Training_Metrics.csv'
    metrics_df.to_csv(csv_filename, index=False)
    print(f"\nMetrics saved to: {csv_filename}")

    return model, train_losses, train_accs, val_accs, val_aucs, metrics_df


# Import evaluate function to avoid circular import
from src.evaluate import evaluate
