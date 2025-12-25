# run_discrete_time_mlp.py
from __future__ import annotations
import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)
from lifelines.utils import concordance_index

# plotting (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------------
# Config
# ----------------------------------------------------
DATA_DIR = Path("data/MIMIC-ED")
DT_PATH  = DATA_DIR / "discrete_time_30min_train.csv"   # <-- from your formatter

OUT_DIR   = Path("outputs_discrete_mlp")
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Columns
ID_COL       = "stay_id"
EVENT_COL    = "event"
TBIN_COL     = "t_bin"
LM_TIME_COL  = "landmark_charttime"       # present if include_landmark_cols=True
HRS_SINCE_ADM_COL = "hours_since_intime"  # optional (often present)

# Training / eval
RNG_SEED     = 42
TEST_SIZE    = 0.2
VAL_SIZE     = 0.1             # fraction of *train* used for validation (grouped)
USE_SCALER   = True            # MLP benefits from scaling
DO_PLOTS     = True            # set False for faster dev iterations

# Discrete-time horizon
BIN_MINUTES  = 30.0
BIN_HOURS    = BIN_MINUTES / 60.0
K_HOURS      = 5
K_BINS       = int(np.ceil(K_HOURS / BIN_HOURS))

# Dataloader / model
BATCH_SIZE   = 32768            # large batches are fine for simple MLPs
NUM_WORKERS  = 2
HIDDEN_SIZES = [256, 128]
DROPOUT      = 0.15
WEIGHT_DECAY = 1e-5
LR           = 2e-3
MAX_EPOCHS   = 100
PATIENCE     = 10               # early stopping patience on val AP
AMP          = True             # mixed precision on GPU

# ----------------------------------------------------
# Utils
# ----------------------------------------------------
def set_seeds(seed: int = RNG_SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available, but GPU training was requested.")
    return torch.device("cuda")

def _save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _save_current_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def _plot_roc_pr(y_true: np.ndarray, y_score: np.ndarray, prefix: str):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_score):.3f}")
    plt.plot([0,1],[0,1],'--',lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {prefix}")
    plt.legend(loc="lower right")
    _save_current_fig(PLOTS_DIR / f"{prefix}_roc.png")

    plt.figure(figsize=(6,5))
    plt.plot(rec, prec, label=f"AP={average_precision_score(y_true, y_score):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR — {prefix}")
    plt.legend(loc="lower left")
    _save_current_fig(PLOTS_DIR / f"{prefix}_pr.png")

def _coerce_numeric_inplace(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_bool_dtype(df[c]):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if not pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype("float32")
    df[cols] = df[cols].fillna(0)

def _numeric_or_coercible_feature_cols(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    num = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    base = [c for c in num if c not in exclude]
    cand = [c for c in df.columns if (c not in exclude) and (c not in base)]
    looks_numeric = []
    for c in cand:
        if pd.api.types.is_object_dtype(df[c]):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().mean() >= 0.5:
                looks_numeric.append(c)
    cols = base + looks_numeric
    _coerce_numeric_inplace(df, cols)
    cols = [c for c in cols if df[c].std(ddof=0) > 0]
    return cols

# ---- Snapshot outcomes (same idea as XGB runner) ----
def _build_snapshot_outcomes_from_pp(
    df_pp: pd.DataFrame, K_bins: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if LM_TIME_COL not in df_pp.columns:
        raise ValueError(
            f"{LM_TIME_COL} not found; enable include_landmark_cols=True in the formatter."
        )
    idx_latest = (
        df_pp.groupby(ID_COL, as_index=False)[LM_TIME_COL]
        .max()
        .rename(columns={LM_TIME_COL: "latest_lm"})
    )
    df_pp2 = df_pp.merge(idx_latest, on=ID_COL, how="inner")
    latest_rows = df_pp2[df_pp2[LM_TIME_COL] == df_pp2["latest_lm"]].copy()
    if latest_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    agg = latest_rows.groupby(ID_COL).agg(
        max_bin=(TBIN_COL, "max"),
        any_event=(EVENT_COL, "max"),
        event_bin=(EVENT_COL, lambda s: np.nan if s.sum()==0 else int(np.argmax(s.values != 0) + 1))
    ).reset_index()

    def _dur(row):
        if int(row["any_event"]) == 1 and not pd.isna(row["event_bin"]):
            return float(int(row["event_bin"]) * BIN_HOURS)
        else:
            return float(int(row["max_bin"]) * BIN_HOURS)

    outcomes = agg.assign(
        duration_hours=agg.apply(_dur, axis=1).astype(float),
        event=agg["any_event"].astype(int)
    )[[ID_COL, "duration_hours", "event"]]

    latest_rows = latest_rows.sort_values([ID_COL, TBIN_COL], kind="mergesort")
    snaps = latest_rows.groupby(ID_COL, as_index=False).head(1).reset_index(drop=True)
    return snaps, outcomes

# ----------------------------------------------------
# MLP Model
# ----------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], dropout: float):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, 1)]  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits

# ----------------------------------------------------
# Training / Eval helpers
# ----------------------------------------------------
def _make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te, device):
    def tens(x): return torch.tensor(x, dtype=torch.float32)
    tr_ds = TensorDataset(tens(X_tr), torch.tensor(y_tr.astype(np.float32)))
    va_ds = TensorDataset(tens(X_va), torch.tensor(y_va.astype(np.float32)))
    te_ds = TensorDataset(tens(X_te), torch.tensor(y_te.astype(np.float32)))
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=(device.type=='cuda'))
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type=='cuda'))
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(device.type=='cuda'))
    return tr_ld, va_ld, te_ld

@torch.no_grad()
def _predict_proba_loader(model: nn.Module, loader: DataLoader, device) -> np.ndarray:
    model.eval()
    preds = []
    for xb, *_ in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=(AMP and device.type=='cuda')):
            logits = model(xb)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        preds.append(probs)
    return np.concatenate(preds, axis=0)

def _train_mlp(
    X_tr, y_tr, X_va, y_va, device, pos_weight: float
) -> nn.Module:
    model = MLP(in_dim=X_tr.shape[1], hidden=HIDDEN_SIZES, dropout=DROPOUT).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    scaler = torch.cuda.amp.GradScaler(enabled=(AMP and device.type=='cuda'))

    # loaders
    tr_loader, va_loader, _ = _make_loaders(X_tr, y_tr, X_va, y_va, X_va, y_va, device)

    best_ap = -np.inf
    best_state = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS+1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(AMP and device.type=='cuda')):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        # Validation AP for early stopping
        va_probs = _predict_proba_loader(model, va_loader, device)
        va_ap = average_precision_score(y_va, va_probs)

        print(f"Epoch {epoch:03d} | train_loss={epoch_loss/len(tr_loader):.4f} | val_AP={va_ap:.4f}")

        if va_ap > best_ap + 1e-5:
            best_ap = va_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"[EarlyStopping] no improvement in {PATIENCE} epochs (best val AP={best_ap:.4f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# ----------------------------------------------------
# Vectorized K-horizon risk scoring (GPU)
# ----------------------------------------------------
@torch.no_grad()
def _score_snaps_vectorized_mlp(
    model: nn.Module, snaps_df: pd.DataFrame, base_cols: list[str], K_bins: int, device
) -> pd.DataFrame:
    if snaps_df.empty:
        return pd.DataFrame(columns=[ID_COL, f"prob_event_within_{K_bins}bins"])

    snapped = snaps_df[[ID_COL] + base_cols].copy()
    snapped[base_cols] = snapped[base_cols].astype("float32")
    ids = snapped[ID_COL].to_numpy()
    base_block = snapped[base_cols].to_numpy(dtype=np.float32, copy=False)

    # Repeat rows K times and build t_bin
    X_base = np.repeat(base_block, K_bins, axis=0)
    t_bins = np.tile(np.arange(1, K_bins + 1, dtype=np.float32), reps=len(snapped)).reshape(-1, 1)
    X = np.concatenate([t_bins, X_base], axis=1)  # [t_bin] + base_feats

    model.eval()
    probs = []
    bs = 131072
    for i in range(0, len(X), bs):
        # --- FIX: construct on CPU, then move with .to(device, non_blocking=True)
        xb_np = X[i:i+bs]
        xb = torch.from_numpy(xb_np).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=(AMP and device.type == "cuda")):
            logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)

    proba = np.concatenate(probs, axis=0).astype(np.float32)
    proba = proba.reshape(len(snapped), K_bins)
    surv  = np.prod(1.0 - proba, axis=1)
    riskK = 1.0 - surv

    return pd.DataFrame({ID_COL: ids, f"prob_event_within_{K_bins}bins": riskK})


# ----------------------------------------------------
# Main runner
# ----------------------------------------------------
def run_discrete_time_mlp(dt_path: Path) -> dict:
    set_seeds(RNG_SEED)
    device = require_gpu()

    if not dt_path.exists():
        raise FileNotFoundError(f"Person-period file not found: {dt_path}")

    # Read once to get columns for parse_dates
    head_cols = pd.read_csv(dt_path, nrows=0).columns.tolist()
    parse_cols = [c for c in [LM_TIME_COL] if c in head_cols]
    df = pd.read_csv(dt_path, parse_dates=parse_cols)

    # Required columns
    need = {ID_COL, TBIN_COL, EVENT_COL}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in person-period data: {missing}")

    # Features: numeric/boolean (coerce objects) excluding id/t_bin/event and landmark timestamp
    exclude = [ID_COL, TBIN_COL, EVENT_COL]
    if LM_TIME_COL in df.columns:
        exclude.append(LM_TIME_COL)
    base_feats = _numeric_or_coercible_feature_cols(df, exclude=exclude)

    # Ensure t_bin/event numeric
    _coerce_numeric_inplace(df, [TBIN_COL, EVENT_COL])

    # Group split into train/test
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RNG_SEED)
    idx = np.arange(len(df))
    tr_idx, te_idx = next(gss.split(idx, groups=df[ID_COL].values))
    tr_all, te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()

    # Validation split from train (grouped by IDs)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=RNG_SEED)
    tr_i, va_i = next(gss2.split(tr_all.index.values, groups=tr_all[ID_COL].values))
    tr = tr_all.iloc[tr_i].copy()
    va = tr_all.iloc[va_i].copy()

    # Optional scaling (recommended for MLP)
    if USE_SCALER:
        scaler = StandardScaler()
        tr_base = tr[base_feats].astype("float32")
        va_base = va[base_feats].astype("float32")
        te_base = te[base_feats].astype("float32")

        tr_scaled = scaler.fit_transform(tr_base.to_numpy())
        va_scaled = scaler.transform(va_base.to_numpy())
        te_scaled = scaler.transform(te_base.to_numpy())

        # Save scaler
        with open(OUT_DIR / "mlp_discrete_scaler.pkl", "wb") as f:
            pickle.dump({"features": base_feats, "scaler": scaler}, f)
    else:
        tr_scaled = tr[base_feats].astype("float32").to_numpy()
        va_scaled = va[base_feats].astype("float32").to_numpy()
        te_scaled = te[base_feats].astype("float32").to_numpy()

    # Final input matrices: [t_bin] + scaled base features
    X_tr = np.concatenate([tr[[TBIN_COL]].astype("float32").to_numpy(), tr_scaled], axis=1)
    y_tr = tr[EVENT_COL].astype(np.float32).to_numpy()

    X_va = np.concatenate([va[[TBIN_COL]].astype("float32").to_numpy(), va_scaled], axis=1)
    y_va = va[EVENT_COL].astype(np.float32).to_numpy()

    X_te = np.concatenate([te[[TBIN_COL]].astype("float32").to_numpy(), te_scaled], axis=1)
    y_te = te[EVENT_COL].astype(np.float32).to_numpy()

    # Class imbalance
    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    pos_weight = (neg / max(1.0, pos)) if pos > 0 else 1.0

    # Train
    model = _train_mlp(X_tr, y_tr, X_va, y_va, device, pos_weight=pos_weight)

    # Row-level metrics on train/test
    _, _, te_loader = _make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te, device)
    tr_loader, _, _ = _make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te, device)

    p_tr = _predict_proba_loader(model, tr_loader, device)
    p_te = _predict_proba_loader(model, te_loader, device)

    metrics = {
        "train_ap":  float(average_precision_score(y_tr, p_tr)),
        "test_ap":   float(average_precision_score(y_te, p_te)),
        "train_auc": float(roc_auc_score(y_tr, p_tr)) if len(np.unique(y_tr))>1 else None,
        "test_auc":  float(roc_auc_score(y_te, p_te)) if len(np.unique(y_te))>1 else None,
        "n_train_pos": int((y_tr==1).sum()),
        "n_train_neg": int((y_tr==0).sum()),
        "n_test_pos":  int((y_te==1).sum()),
        "n_test_neg":  int((y_te==0).sum()),
        "hidden_sizes": HIDDEN_SIZES,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "use_amp": AMP,
    }

    if DO_PLOTS:
        try:
            _plot_roc_pr(y_te, p_te, prefix="mlp_discrete_test")
        except Exception as e:
            print(f"[WARN] Plotting ROC/PR failed: {e}")

    # -------------------------------
    # Patient-level Harrell's C-index
    # -------------------------------
    snaps_all, outcomes_all = _build_snapshot_outcomes_from_pp(df, K_BINS)
    snaps_tr  = snaps_all[snaps_all[ID_COL].isin(set(tr[ID_COL]))].reset_index(drop=True)
    snaps_te  = snaps_all[snaps_all[ID_COL].isin(set(te[ID_COL]))].reset_index(drop=True)
    out_tr = outcomes_all[outcomes_all[ID_COL].isin(set(snaps_tr[ID_COL]))].reset_index(drop=True)
    out_te = outcomes_all[outcomes_all[ID_COL].isin(set(snaps_te[ID_COL]))].reset_index(drop=True)

    # Build base snapshot features (apply same scaling)
    def _prepare_snaps(snaps_df: pd.DataFrame) -> pd.DataFrame:
        snaps_df = snaps_df.copy()
        snaps_df[base_feats] = snaps_df[base_feats].astype("float32")
        if USE_SCALER:
            with open(OUT_DIR / "mlp_discrete_scaler.pkl", "rb") as f:
                s_payload = pickle.load(f)
            scaler = s_payload["scaler"]
            snaps_df[base_feats] = scaler.transform(snaps_df[base_feats].to_numpy())
        return snaps_df

    snaps_tr = _prepare_snaps(snaps_tr)
    snaps_te = _prepare_snaps(snaps_te)

    # Vectorized GPU scoring for K-horizon risk
    risk_tr = _score_snaps_vectorized_mlp(model, snaps_tr, base_feats, K_bins=K_BINS, device=device)
    risk_te = _score_snaps_vectorized_mlp(model, snaps_te, base_feats, K_bins=K_BINS, device=device)

    def _cindex(outcomes: pd.DataFrame, risk: pd.DataFrame) -> Tuple[float|None, int]:
        if outcomes.empty or risk.empty:
            return None, 0
        m = outcomes.merge(risk, on=ID_COL, how="inner")
        m = m[(m["duration_hours"] > 0) & m["duration_hours"].notna()]
        if m.empty:
            return None, 0
        c = float(concordance_index(
            event_times=m["duration_hours"].values,
            predicted_scores=m[f"prob_event_within_{K_BINS}bins"].values,
            event_observed=m["event"].values
        ))
        return c, int(len(m))

    c_train, n_train = _cindex(out_tr, risk_tr)
    c_test,  n_test  = _cindex(out_te, risk_te)
    metrics.update({
        "train_cindex": c_train,
        "test_cindex":  c_test,
        "n_train_for_cindex": n_train,
        "n_test_for_cindex":  n_test,
        "cindex_horizon_hours": K_HOURS,
        "bin_minutes": BIN_MINUTES,
    })

    # Save model + metrics
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": X_tr.shape[1],
        "hidden_sizes": HIDDEN_SIZES,
        "dropout": DROPOUT
    }, OUT_DIR / "mlp_discrete.pt")

    with open(OUT_DIR / "mlp_discrete_features.pkl", "wb") as f:
        pickle.dump({"order": [TBIN_COL] + base_feats, "base_features": base_feats}, f)

    _save_json(metrics, OUT_DIR / "metrics.json")

    # Prints
    print("\n[MLP Discrete-Time] Row-level metrics")
    print(f"  Train AP={metrics['train_ap']:.4f} | AUC={metrics['train_auc'] if metrics['train_auc'] is not None else 'NA'}")
    print(f"  Test  AP={metrics['test_ap']:.4f} | AUC={metrics['test_auc'] if metrics['test_auc'] is not None else 'NA'}")

    print("\n[MLP Discrete-Time] Patient-level C-index (latest snapshot, horizon K)")
    print(f"  K = {K_HOURS:.1f} hours ({K_BINS} bins of {BIN_MINUTES:.0f} min)")
    print(f"  Train C-index: {metrics['train_cindex'] if metrics['train_cindex'] is not None else 'NA'} on N={metrics['n_train_for_cindex']}")
    print(f"  Test  C-index: {metrics['test_cindex'] if metrics['test_cindex'] is not None else 'NA'} on N={metrics['n_test_for_cindex']}")

    print(f"\n✅ Done. Artifacts -> {OUT_DIR}/")
    print(f"   Plots -> {PLOTS_DIR}/")

    return metrics

# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():
    run_discrete_time_mlp(DT_PATH)

if __name__ == "__main__":
    main()
