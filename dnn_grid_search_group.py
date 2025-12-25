# grid_search_mlp_bce.py
import math
import time
import random
import itertools
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

# >>> ADD
import os, json
from pathlib import Path
from datetime import datetime


# >>> ADD
def _atomic_write_bytes(path: Path, data: bytes):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def _atomic_write_json(path: Path, obj):
    _atomic_write_bytes(path, json.dumps(obj, indent=2).encode("utf-8"))


# ---------------------------
# Repro & Device
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)

def get_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script is GPU-only by design.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"[INFO] Using GPU: {torch.cuda.get_device_name(device)}")
    return device

# ---------------------------
# Model
# ---------------------------
class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_layers: Tuple[int, ...],
        activation: str = "relu",
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        acts = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        if activation not in acts:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: List[nn.Module] = []
        last = in_features
        for h in hidden_layers:
            layers.append(nn.Linear(last, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(acts[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.out = nn.Linear(last, 1)

        # Kaiming init for ReLU-like, Xavier otherwise
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if activation in ("relu", "silu"):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.backbone(x)
        logits = self.out(z)  # [B, 1]
        return logits

# ---------------------------
# Training / Eval
# ---------------------------
@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    grad_clip: Optional[float] = 1.0
    patience: int = 5  # early stopping
    amp: bool = True

def make_loaders(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    val_frac: float = 0.2,
    groups: Optional[torch.Tensor] = None,
    stratified_groups: bool = False,  # set True if you want class balance across groups (needs sklearn>=1.1)
):
    """
    If `groups` is provided (1D array-like of length N), ensures no group leaks across train/val.
    If `stratified_groups=True` (and sklearn is available), tries to balance classes across folds using StratifiedGroupKFold.
    """
    N = len(X)
    assert len(y) == N, "X and y must have same length"
    if groups is not None:
        assert len(groups) == N, "`groups` must be same length as X/y"

    # Make base dataset
    ds = TensorDataset(X, y)

    if groups is None:
        # fallback: simple random split (as before)
        n_val = int(N * val_frac)
        n_train = N - n_val
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(7))
        train_idx = train_ds.indices
        val_idx = val_ds.indices
    else:
        if not HAVE_SKLEARN:
            # Minimal dependency-free group split: keep whole groups together
            # Build index lists per group, then move full groups into val until target size reached.
            import numpy as np
            g = np.asarray(groups)
            uniq, counts = np.unique(g, return_counts=True)
            # Shuffle groups for randomness but deterministically
            rng = np.random.default_rng(7)
            order = np.arange(len(uniq))
            rng.shuffle(order)
            target_val = int(round(N * val_frac))
            val_mask = np.zeros(N, dtype=bool)
            total = 0
            for idx in order:
                grp = uniq[idx]
                grp_mask = (g == grp)
                if total < target_val:
                    val_mask |= grp_mask
                    total += grp_mask.sum()
            train_idx = torch.as_tensor(~val_mask).nonzero(as_tuple=False).view(-1).tolist()
            val_idx = torch.as_tensor(val_mask).nonzero(as_tuple=False).view(-1).tolist()
        else:
            import numpy as np
            if stratified_groups:
                # Requires sklearn>=1.1
                sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
                # Take the first split; validation size will be roughly 1/5
                train_idx, val_idx = next(sgkf.split(np.zeros(N), y.cpu().numpy(), groups=np.asarray(groups)))
            else:
                gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=7)
                train_idx, val_idx = next(gss.split(np.zeros(N), y.cpu().numpy(), groups=np.asarray(groups)))

    # Build subsets
    from torch.utils.data import Subset
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    # DataLoaders (unchanged)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True).float().view(-1, 1)

        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)

        all_logits.append(logits.detach().float().cpu())
        all_targets.append(yb.detach().float().cpu())

    logits = torch.cat(all_logits, dim=0).squeeze(1)
    targets = torch.cat(all_targets, dim=0).squeeze(1)

    probs = torch.sigmoid(logits).numpy()
    t = targets.numpy()

    # Metrics
    if HAVE_SKLEARN:
        try:
            auroc = roc_auc_score(t, probs)
        except Exception:
            auroc = float("nan")
    else:
        auroc = float("nan")

    # accuracy for reference
    preds = (probs >= 0.5).astype("float32")
    acc = (preds == t).mean()

    avg_loss = total_loss / len(loader.dataset)
    return {"loss": avg_loss, "auroc": auroc, "acc": acc}

def train_one(
    model: nn.Module,
    loaders: Tuple[DataLoader, DataLoader],
    cfg: TrainConfig,
    device: torch.device,
):
    train_loader, val_loader = loaders
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp)

    best_snapshot = None
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float().view(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=cfg.amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            if cfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * xb.size(0)

        train_loss = epoch_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, device)
        val_loss = val_metrics["loss"]
        val_auroc = val_metrics["auroc"]
        val_acc = val_metrics["acc"]

        # select the best model by smallest validation BCE loss
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_snapshot = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        elapsed = time.time() - t0
        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({elapsed:.1f}s)"
        )

        if epochs_no_improve >= cfg.patience:
            print("[EarlyStopping] No improvement, stopping.")
            break

    if best_snapshot is not None:
        model.load_state_dict(best_snapshot)

    final_val = evaluate(model, val_loader, device)
    return final_val

# ---------------------------
# Grid Search
# ---------------------------
# ---------------------------
# Grid Search
# ---------------------------
def grid_search(
    X,
    y,
    layer_options,
    activations,
    dropouts,
    use_batchnorm_options,
    lrs,
    weight_decays,
    batch_sizes,
    epochs: int = 30,
    patience: int = 5,
    amp: bool = True,
    seed: int = 42,
    groups=None,
    stratified_groups: bool = False,
    # >>> ADD
    save_dir: str = "grid_runs",  # folder to write leaderboard & best weights
):
    set_seed(seed)
    device = get_device()

    # Prepare tensors
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
    else:
        X = X.float()
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float32)
    else:
        y = y.float()
    if y.ndim != 1:
        y = y.view(-1)

    in_features = X.shape[1]
    print(f"[INFO] X shape: {tuple(X.shape)}, y shape: {tuple(y.shape)}")

    best: Dict[str, Any] = {"score": -float("inf"), "val_loss": float("inf"), "config": None, "model": None, "metrics": None}

    combos = list(
        itertools.product(
            layer_options,
            activations,
            dropouts,
            use_batchnorm_options,
            lrs,
            weight_decays,
            batch_sizes,
        )
    )
    print(f"[INFO] Grid size: {len(combos)}")

    # >>> ADD BEGIN: file setup
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    top5_path = out_dir / "top5_results.json"          # rolling leaderboard (small JSON array)
    best_weights_path = out_dir / "best_model.pt"       # weights of the current best model

    # initialize / create if missing
    if not top5_path.exists():
        _atomic_write_json(top5_path, [])
    # don't pre-create best_model.pt; write only on improvement

    # in-memory leaderboard
    top5 = []  # list of dicts, sorted by smallest val_loss
    # >>> ADD END


    for i, (layers, act, do, use_bn, lr, wd, bs) in enumerate(combos, start=1):
        print(
            f"\n=== Trial {i}/{len(combos)} ===\n"
            f"layers={layers} act={act} dropout={do} batchnorm={use_bn} "
            f"lr={lr} wd={wd} batch_size={bs}"
        )
        model = MLP(
            in_features=in_features,
            hidden_layers=layers,
            activation=act,
            dropout=do,
            use_batchnorm=use_bn,
        )

        loaders = make_loaders(X, y, batch_size=bs, val_frac=0.2, groups=groups, stratified_groups=stratified_groups)  # 👈 UPDATED
        cfg = TrainConfig(
            batch_size=bs,
            lr=lr,
            weight_decay=wd,
            epochs=epochs,
            patience=patience,
            amp=amp,
        )
        metrics = train_one(model, loaders, cfg, device)

        # Use AUROC primarily; fallback to inverse loss if AUROC NaN
        auroc = metrics["auroc"]
        selection_score = -metrics["loss"]  # smaller loss = better
        improved = metrics["loss"] < best["val_loss"]


        print(f"[RESULT] val_loss={metrics['loss']:.4f} val_acc={metrics['acc']:.4f} val_auroc={auroc:.4f}")

        # --- Maintain the single "best" (same logic, but avoid keeping large state in RAM) ---
        if improved:
            best.update(
                {
                    "score": selection_score,
                    "val_loss": metrics["loss"],
                    "config": {
                        "layers": layers,
                        "activation": act,
                        "dropout": do,
                        "batchnorm": use_bn,
                        "lr": lr,
                        "weight_decay": wd,
                        "batch_size": bs,
                        "epochs": epochs,
                        "patience": patience,
                        "amp": amp,
                    },
                    # Don't keep weights in-memory for the grid trial to reduce RAM pressure
                    "model": None,
                    "metrics": metrics,
                }
            )
            print("[BEST] Updated best configuration.")

            # >>> ADD: Save the *current best* weights to disk, atomically (CPU tensors)
            cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            tmp_w = best_weights_path.with_suffix(".pt.tmp")
            torch.save(cpu_state, tmp_w)
            os.replace(tmp_w, best_weights_path)

        # >>> ADD: Update the Top-5 leaderboard on disk (every iteration)
        trial_entry = {
            "trial": i,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "val_loss": float(metrics["loss"]),
            "val_acc": float(metrics["acc"]),
            "val_auroc": float(auroc) if auroc == auroc else None,  # NaN -> None
            "config": {
                "layers": list(layers),
                "activation": act,
                "dropout": do,
                "batchnorm": use_bn,
                "lr": lr,
                "weight_decay": wd,
                "batch_size": bs,
                "epochs": epochs,
                "patience": patience,
                "amp": amp,
            },
        }
        top5.append(trial_entry)
        top5.sort(key=lambda d: d["val_loss"])   # smaller loss is better
        if len(top5) > 5:
            top5 = top5[:5]
        _atomic_write_json(top5_path, top5)

        # >>> ADD: free memory aggressively between trials
        del model
        torch.cuda.empty_cache()

    print("\n===== Best Configuration =====")
    print(best["config"])
    print("Best metrics:", best["metrics"])

    # train best model and save weights
    print("\n[INFO] Re-training best model on full data...")
    best_model = MLP(
        in_features=in_features,
        hidden_layers=best["config"]["layers"],
        activation=best["config"]["activation"],
        dropout=best["config"]["dropout"],
        use_batchnorm=best["config"]["batchnorm"],
    )
    full_loader = DataLoader(
        TensorDataset(X, y),
        batch_size=best["config"]["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )
    full_cfg = TrainConfig(
        batch_size=best["config"]["batch_size"],
        lr=best["config"]["lr"],
        weight_decay=best["config"]["weight_decay"],
        epochs=best["config"]["epochs"],
        patience=best["config"]["patience"],
        amp=best["config"]["amp"],
    )
    final_metrics = train_one(best_model, (full_loader, full_loader), full_cfg, device)
    print("Final metrics on full data:", final_metrics)
    best["model"] = best_model.state_dict()
    best["metrics"] = final_metrics
    return best

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/MIMIC-ED/event_level_training_data.csv")
    X = df.drop(columns=["is_sepsis", "stay_id"]).to_numpy()
    groups = df["stay_id"].to_numpy()  # 👈 NEW
    y = df["is_sepsis"].to_numpy()
    print(f"Data shape: X={X.shape}, y={y.shape}, Positives={y.sum()}, Negatives={len(y)-y.sum()}")

    layer_options = [
        (64,),
        (128,),
        (128, 64),
        (256, 128, 64),
    ]
    activations = ["relu", "silu"]
    dropouts = [0.0, 0.2, 0.5]
    use_batchnorm_options = [False, True]
    lrs = [1e-3, 3e-4]
    weight_decays = [1e-4, 1e-5]
    batch_sizes = [9192]

    best = grid_search(
        X=X,
        y=y,
        layer_options=layer_options,
        activations=activations,
        dropouts=dropouts,
        use_batchnorm_options=use_batchnorm_options,
        lrs=lrs,
        weight_decays=weight_decays,
        batch_sizes=batch_sizes,
        epochs=30,
        patience=4,
        amp=True,
        seed=123,
        groups=groups,
        stratified_groups=True,  # 👈 NEW
    )

    # If you want to rebuild the best model later:
    mdl = MLP(
        in_features=X.shape[1],
        hidden_layers=tuple(best["config"]["layers"]),
        activation=best["config"]["activation"],
        dropout=best["config"]["dropout"],
        use_batchnorm=best["config"]["batchnorm"],
    )
    mdl.load_state_dict(best["model"])
    mdl.eval()
    print("[DONE] Best model restored.")
