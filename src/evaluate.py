import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from torch_geometric.data import Batch, DataLoader
from tqdm import tqdm

from src.utils import AverageMeter


def accuracy(labels, predictions):
    """Calculate accuracy."""
    return accuracy_score(labels, predictions)


def precision(labels, predictions):
    """Calculate precision."""
    return precision_score(labels, predictions, zero_division=0)


def recall(labels, predictions):
    """Calculate recall."""
    return recall_score(labels, predictions, zero_division=0)


def auc_score(labels, predictions):
    """Calculate AUC."""
    return roc_auc_score(labels, predictions)


def evaluate(model, dataloader, device, criterion=None):
    """Evaluate model on given dataloader."""
    model.eval()
    running_loss = AverageMeter()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            logits = model(
                batch.x, batch.edge_index, batch.batch,
                batch.protein_embedding
            )

            if criterion:
                loss = criterion(logits, batch.y)
                running_loss.update(loss.item(), batch.y.size(0))

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    auc_val = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0

    if criterion:
        return running_loss.avg, acc, prec, rec, auc_val, all_probs, all_preds, all_labels

    return acc, prec, rec, auc_val, all_probs, all_preds, all_labels


def evaluate_with_bootstrap(model, dataset, device, criterion, n_runs=5, subsample_ratio=0.8, batch_size=16):
    """Evaluate model using bootstrapping to get mean ± std."""
    model.eval()
    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "auc": [],
        "f1": []
    }

    print(f"\nPerforming bootstrapping with {n_runs} runs ({subsample_ratio*100:.0f}% subsample each)...")

    for run in range(n_runs):
        indices = np.random.choice(len(dataset), size=int(len(dataset)*subsample_ratio), replace=True)
        subset = [dataset[i] for i in indices]

        def collate_fn(batch):
            return Batch.from_data_list(batch)

        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        _, acc, prec, rec, auc, _, _, _ = evaluate(model, loader, device, criterion)

        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        metrics["accuracy"].append(acc)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["auc"].append(auc)
        metrics["f1"].append(f1)

    results = {}
    for key in metrics:
        results[f"{key}"] = f"{np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}"
        results[f"{key}_mean"] = np.mean(metrics[key])
        results[f"{key}_std"] = np.std(metrics[key])

    return results
