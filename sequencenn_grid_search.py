"""
Grid search trainer for GenericRNN (PyTorch)

- Searches over rnn_type (rnn/gru/lstm), hidden_size, num_layers, dropout,
  learning rate, weight decay, and optional FC head sizes.
- **Bidirectional is fixed to False** (no future context in practice).
- Works with variable-length sequences via pad_collate from generic_rnn.py
- Early stopping on validation loss with patience.
- Prints a ranked leaderboard of configs by best val loss.

Replace `load_sequences_and_targets()` with your cleaned data.

Run:
    python grid_search_rnn.py

Requires:
    generic_rnn.py in the same directory.
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from sequencenn import GenericRNN, pad_collate


# -----------------------------
# Utilities
# -----------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 5  # early stopping patience (epochs)
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    loss_type: str = "bce"  # "bce" or "mse"
    val_split: float = 0.2


@dataclass
class GridResult:
    val_loss: float
    metrics: Dict[str, float]
    config: Dict[str, object]


class SequenceDataset(Dataset):
    """Simple dataset: list of (seq_len_i, feat) float tensors and [0,1] targets."""

    def __init__(self, sequences: Sequence[torch.Tensor], targets: Sequence[float]):
        assert len(sequences) == len(targets)
        self.sequences = [torch.as_tensor(s, dtype=torch.float32) for s in sequences]
        self.targets = [float(y) for y in targets]
        # Sanity check: variable lengths allowed
        feat_dims = {s.size(1) for s in self.sequences}
        if len(feat_dims) != 1:
            raise ValueError(f"All sequences must share the same feature dim; got {feat_dims}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float]:
        return self.sequences[idx], self.targets[idx]


# -----------------------------
# Data loading (REPLACE for your project)
# -----------------------------

def load_sequences_and_targets() -> Tuple[List[torch.Tensor], List[float]]:
    """TODO: Replace with your cleaned data loader.

    Must return:
        sequences: List of tensors shaped (seq_len_i, input_size)
        targets:   List of floats in [0,1]

    Below is a **synthetic fallback** to keep the script runnable.
    Set USE_SYNTHETIC=False once you wire real data.
    """
    USE_SYNTHETIC = True
    if not USE_SYNTHETIC:
        raise NotImplementedError("Implement your real data loader and set USE_SYNTHETIC=False")

    # Synthetic data: ~1000 variable-length sequences, input_size=16
    N = 1000
    input_size = 16
    rng = torch.Generator().manual_seed(0)
    lengths = torch.randint(low=1, high=60, size=(N,), generator=rng)
    sequences = [torch.randn(l.item(), input_size, generator=rng) for l in lengths]
    # Binary-ish targets correlated with mean of first feature
    logits = torch.stack([s[:, 0].mean() for s in sequences])
    probs = torch.sigmoid(logits)
    targets = probs.tolist()
    return sequences, targets


# -----------------------------
# Training / Evaluation
# -----------------------------

def make_loaders(ds: Dataset, cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    n_val = max(1, int(len(ds) * cfg.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate)
    return train_loader, val_loader


def train_and_validate(
    model: GenericRNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    optim_kwargs: Dict,
) -> Tuple[float, Dict[str, float]]:
    device = torch.device(cfg.device)
    model.to(device)

    if cfg.loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss()
        use_logits = True
    elif cfg.loss_type == "mse":
        criterion = nn.MSELoss()
        use_logits = False
    else:
        raise ValueError("loss_type must be 'bce' or 'mse'")

    optimizer = torch.optim.Adam(model.parameters(), **optim_kwargs)

    best_val_loss = math.inf
    best_metrics: Dict[str, float] = {}
    best_state = None
    epochs_no_improve = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            x, lengths, y = batch.x.to(device), batch.lengths.to(device), batch.y.to(device)
            optimizer.zero_grad()
            if use_logits:
                logits, _ = model(x, lengths=lengths, return_logits=True)
                loss = criterion(logits, y)
            else:
                probs, _ = model(x, lengths=lengths, return_logits=False)
                loss = criterion(probs, y)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            n_train += x.size(0)
        train_loss /= max(1, n_train)

        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                x, lengths, y = batch.x.to(device), batch.lengths.to(device), batch.y.to(device)
                if use_logits:
                    logits, _ = model(x, lengths=lengths, return_logits=True)
                    loss = criterion(logits, y)
                    probs = torch.sigmoid(logits)
                else:
                    probs, _ = model(x, lengths=lengths, return_logits=False)
                    loss = criterion(probs, y)
                val_loss += loss.item() * x.size(0)
                n_val += x.size(0)
                preds = (probs >= 0.5).float()
                correct += (preds == (y >= 0.5).float()).sum().item()
        val_loss /= max(1, n_val)
        val_acc = correct / max(1, n_val)

        # Early stopping check
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_metrics = {"val_loss": val_loss, "val_acc": val_acc, "train_loss": train_loss}
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break

    # load best state back before returning
    if best_state is None:
        raise RuntimeError("Training completed without capturing a best state.")
    model.load_state_dict(best_state)
    return best_val_loss, best_metrics


# -----------------------------
# Grid search
# -----------------------------

def run_grid_search():
    seed_everything(7)

    sequences, targets = load_sequences_and_targets()
    ds = SequenceDataset(sequences, targets)
    train_cfg = TrainConfig()
    train_loader, val_loader = make_loaders(ds, train_cfg)

    input_size = ds.sequences[0].size(1)

    param_grid = {
        "rnn_type": ["rnn", "gru", "lstm"],
        "hidden_size": [64, 128, 256],
        "num_layers": [1, 2],
        "dropout": [0.0, 0.2, 0.5],
        # Model head sizes: empty -> linear, or one hidden layer
        "fc_hidden_sizes": [[], [64], [128]],
        # Optimizer params
        "lr": [1e-3, 3e-4],
        "weight_decay": [0.0, 1e-4],
    }

    keys = list(param_grid.keys())
    grid_values = [param_grid[k] for k in keys]
    total_configs = math.prod(len(v) for v in grid_values)

    results: List[GridResult] = []

    print(f"Total configs: {total_configs}")

    for i, values in enumerate(itertools.product(*grid_values), 1):
        cfg_dict = dict(zip(keys, values))
        print(f"\n[{i}/{total_configs}] Config: {cfg_dict}")

        # Build model with bidirectional=False (as desired)
        model = GenericRNN(
            input_size=input_size,
            hidden_size=cfg_dict["hidden_size"],
            rnn_type=cfg_dict["rnn_type"],
            num_layers=cfg_dict["num_layers"],
            bidirectional=False,
            dropout=cfg_dict["dropout"],
            fc_hidden_sizes=cfg_dict["fc_hidden_sizes"],
            batch_first=True,
        )

        best_val, metrics = train_and_validate(
            model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=train_cfg,
            optim_kwargs={"lr": cfg_dict["lr"], "weight_decay": cfg_dict["weight_decay"]},
        )

        results.append(GridResult(val_loss=best_val, metrics=metrics, config=cfg_dict))
        print(f"=> best_val_loss={best_val:.4f}, val_acc={metrics['val_acc']:.4f}")

    # Sort results by best val loss ascending
    results.sort(key=lambda r: r.val_loss)

    print("\n===== Leaderboard (by val_loss) =====")
    for rank, result in enumerate(results[:10], 1):
        val_loss = result.val_loss
        metrics = result.metrics
        cfg_dict = result.config
        print(
            f"#{rank:>2} val_loss={val_loss:.4f} val_acc={metrics['val_acc']:.4f} "
            f"train_loss={metrics['train_loss']:.4f} | {cfg_dict}"
        )

    best_result = results[0]
    best_val = best_result.val_loss
    best_metrics = best_result.metrics
    best_cfg = best_result.config
    print("\nBest Config:")
    for k, v in best_cfg.items():
        print(f"  {k}: {v}")
    print("Best Metrics:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    run_grid_search()
