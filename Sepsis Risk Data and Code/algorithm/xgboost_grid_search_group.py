# grid_search_xgb.py
import os
import json
import time
import random
import itertools
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import xgboost as xgb
except Exception as e:
    raise RuntimeError(
        "xgboost is required. Install with `pip install xgboost` (GPU build recommended)."
    ) from e

# sklearn (optional but preferred for AUROC and group splits)
try:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

# ---------------------------
# Repro
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# ---------------------------
# Atomic file helpers
# ---------------------------
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
# Split utilities (match MLP behavior)
# ---------------------------
def make_indices(
    N: int,
    y: np.ndarray,
    val_frac: float = 0.2,
    groups: Optional[np.ndarray] = None,
    stratified_groups: bool = False,
):
    if groups is None:
        # simple random split (deterministic)
        rng = np.random.default_rng(7)
        idx = np.arange(N)
        rng.shuffle(idx)
        n_val = int(round(N * val_frac))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        return train_idx, val_idx

    if not HAVE_SKLEARN:
        # Minimal dependency-free group split: keep whole groups together
        g = np.asarray(groups)
        uniq, counts = np.unique(g, return_counts=True)
        rng = np.random.default_rng(7)
        order = np.arange(len(uniq))
        rng.shuffle(order)
        target_val = int(round(N * val_frac))
        val_mask = np.zeros(N, dtype=bool)
        total = 0
        for ix in order:
            grp = uniq[ix]
            grp_mask = (g == grp)
            if total < target_val:
                val_mask |= grp_mask
                total += grp_mask.sum()
        train_idx = np.flatnonzero(~val_mask)
        val_idx = np.flatnonzero(val_mask)
        return train_idx, val_idx
    else:
        if stratified_groups:
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
            train_idx, val_idx = next(sgkf.split(np.zeros(N), y, groups=np.asarray(groups)))
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=7)
            train_idx, val_idx = next(gss.split(np.zeros(N), y, groups=np.asarray(groups)))
        return train_idx, val_idx

# ---------------------------
# Metrics
# ---------------------------
def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(np.float32).reshape(-1)
    y_prob = y_prob.astype(np.float32).reshape(-1)
    # log loss (manual, stable)
    eps = 1e-12
    p = np.clip(y_prob, eps, 1.0 - eps)
    logloss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    # accuracy
    acc = float(np.mean((y_prob >= 0.5).astype(np.float32) == y_true))
    # auroc
    if HAVE_SKLEARN:
        try:
            auroc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auroc = float("nan")
    else:
        auroc = float("nan")
    return {"loss": float(logloss), "acc": acc, "auroc": auroc}

# ---------------------------
# GPU params helper (XGBoost >=2.0 uses device='cuda'; older uses gpu_hist)
# ---------------------------
def gpu_params():
    params = {}
    try:
        ver = tuple(int(x) for x in xgb.__version__.split(".")[:2])
        if ver >= (2, 0):
            params["device"] = "cuda"
            params["tree_method"] = "hist"
        else:
            params["tree_method"] = "gpu_hist"
            params["predictor"] = "gpu_predictor"
    except Exception:
        params["tree_method"] = "gpu_hist"
        params["predictor"] = "gpu_predictor"
    return params

# ---------------------------
# Grid Search (XGBoost, GPU)
# ---------------------------
@dataclass
class TrainConfigXGB:
    early_stopping_rounds: int = 50
    eval_metric: str = "logloss"  # also track auc via sklearn
    val_frac: float = 0.2

def grid_search_xgb(
    X,
    y,
    # Hyperparameter grids (lists)
    max_depths: List[int],
    learning_rates: List[float],
    n_estimators_list: List[int],
    subsamples: List[float],
    colsample_bytree_list: List[float],
    reg_lambdas: List[float],
    reg_alphas: List[float],
    min_child_weights: List[float],
    gammas: List[float],
    max_bins: Optional[List[int]] = None,          # XGBoost 2.x histogram bins
    scale_pos_weight_list: Optional[List[float]] = None,
    # General
    seed: int = 42,
    cfg: TrainConfigXGB = TrainConfigXGB(),
    groups: Optional[np.ndarray] = None,
    stratified_groups: bool = False,
    save_dir: str = "grid_runs_xgb",
):
    """
    Returns:
        best: Dict with keys: config, metrics, model_path, top5_path
    """
    set_seed(seed)

    # Prepare arrays
    if not isinstance(X, np.ndarray):
        X = np.asarray(X, dtype=np.float32)
    else:
        X = X.astype(np.float32, copy=False)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y, dtype=np.float32)
    else:
        y = y.astype(np.float32, copy=False)
    y = y.reshape(-1)

    N, D = X.shape
    print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}, positives={int(y.sum())}, negatives={len(y)-int(y.sum())}")

    # Setup output dir & files
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top5_path = out_dir / "top5_results_xgb.json"
    best_model_path = out_dir / "best_xgb.json"

    if not top5_path.exists():
        _atomic_write_json(top5_path, [])
    top5: List[Dict[str, Any]] = []

    # Build hyperparameter combinations
    if max_bins is None:
        max_bins = [256]
    if scale_pos_weight_list is None:
        scale_pos_weight_list = [1.0]

    combos = list(itertools.product(
        max_depths,
        learning_rates,
        n_estimators_list,
        subsamples,
        colsample_bytree_list,
        reg_lambdas,
        reg_alphas,
        min_child_weights,
        gammas,
        max_bins,
        scale_pos_weight_list,
    ))
    print(f"[INFO] Grid size: {len(combos)}")

    # Train/val split indices (fixed for all trials)
    train_idx, val_idx = make_indices(N, y, cfg.val_frac, groups, stratified_groups)
    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[val_idx], y[val_idx]

    best: Dict[str, Any] = {
        "val_loss": float("inf"),
        "config": None,
        "metrics": None,
        "model_path": None,
        "top5_path": str(top5_path),
    }

    # Main loop
    for i, (max_depth, lr, n_estimators, subsample, colsample_bytree,
            reg_lambda, reg_alpha, min_child_weight, gamma, max_bin, spw) in enumerate(combos, start=1):

        print(
            f"\n=== XGB Trial {i}/{len(combos)} ===\n"
            f"max_depth={max_depth} lr={lr} n_estimators={n_estimators} subsample={subsample} "
            f"colsample_bytree={colsample_bytree} lambda={reg_lambda} alpha={reg_alpha} "
            f"min_child_weight={min_child_weight} gamma={gamma} max_bin={max_bin} "
            f"scale_pos_weight={spw}"
        )

        params = {
            "objective": "binary:logistic",
            "random_state": seed,
            "max_depth": max_depth,
            "learning_rate": lr,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "max_bin": max_bin,
            "enable_categorical": False,
            "verbosity": 1,
            **gpu_params(),
        }
        if spw is not None:
            params["scale_pos_weight"] = spw

        model = xgb.XGBClassifier(**params)

        t0 = time.time()
        # Early stopping on validation split
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric=cfg.eval_metric,
            verbose=False,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )

        # Predict proba on validation (XGBClassifier uses best_iteration automatically when ES is used)
        y_prob = model.predict_proba(Xva)[:, 1]
        metrics = binary_metrics(yva, y_prob)
        elapsed = time.time() - t0

        print(f"[RESULT] val_loss={metrics['loss']:.4f} val_acc={metrics['acc']:.4f} val_auroc={metrics['auroc']:.4f} ({elapsed:.1f}s)")

        # Keep Top-5 (by smallest loss), persist every trial
        trial_entry = {
            "trial": i,
            "val_loss": float(metrics["loss"]),
            "val_acc": float(metrics["acc"]),
            "val_auroc": float(metrics["auroc"]),
            "config": {
                "max_depth": max_depth,
                "learning_rate": lr,
                "n_estimators": n_estimators,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "reg_lambda": reg_lambda,
                "reg_alpha": reg_alpha,
                "min_child_weight": min_child_weight,
                "gamma": gamma,
                "max_bin": max_bin,
                "scale_pos_weight": spw,
                "early_stopping_rounds": cfg.early_stopping_rounds,
            },
        }
        top5.append(trial_entry)
        top5.sort(key=lambda d: d["val_loss"])
        if len(top5) > 5:
            top5 = top5[:5]
        _atomic_write_json(top5_path, top5)

        # Update "best" and save model atomically
        if metrics["loss"] < best["val_loss"]:
            best["val_loss"] = metrics["loss"]
            best["config"] = trial_entry["config"]
            best["metrics"] = metrics

            # Save current best model
            tmp_path = best_model_path.with_suffix(".json.tmp")
            model.save_model(tmp_path.with_suffix(".bin"))
            os.replace(tmp_path, best_model_path)
            best["model_path"] = str(best_model_path)
            print("[BEST] Updated best model & saved to disk.")

        # Free per-trial objects
        del model

    # Re-train best model on FULL data (optional, like your MLP script)
    print("\n[INFO] Re-training best XGB model on full data...")
    bc = best["config"]
    full_params = {
        "objective": "binary:logistic",
        "random_state": seed,
        "max_depth": bc["max_depth"],
        "learning_rate": bc["learning_rate"],
        "n_estimators": bc["n_estimators"],
        "subsample": bc["subsample"],
        "colsample_bytree": bc["colsample_bytree"],
        "reg_lambda": bc["reg_lambda"],
        "reg_alpha": bc["reg_alpha"],
        "min_child_weight": bc["min_child_weight"],
        "gamma": bc["gamma"],
        "max_bin": bc["max_bin"],
        "enable_categorical": False,
        **gpu_params(),
    }
    if bc.get("scale_pos_weight", None) is not None:
        full_params["scale_pos_weight"] = bc["scale_pos_weight"]

    best_model = xgb.XGBClassifier(**full_params)
    best_model.fit(
        X, y,
        eval_set=[(X, y)],
        eval_metric=cfg.eval_metric,
        verbose=False,
        early_stopping_rounds=bc["early_stopping_rounds"],
    )
    # Persist the final full-data model
    tmp_path = best_model_path.with_suffix(".json.tmp")
    best_model.save_model(tmp_path.with_suffix(".bin"))
    os.replace(tmp_path, best_model_path)
    print(f"[DONE] Saved full-data best XGB model to: {best_model_path}")

    return best

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/MIMIC-ED/event_level_training_data.csv")
    X = df.drop(columns=["is_sepsis", "stay_id"]).to_numpy(dtype=np.float32)
    y = df["is_sepsis"].to_numpy(dtype=np.float32)
    groups = df["stay_id"].to_numpy()

    print(f"Data shape: X={X.shape}, y={y.shape}, Positives={y.sum()}, Negatives={len(y)-y.sum()}")

    best = grid_search_xgb(
        X=X,
        y=y,
        max_depths=[4, 6, 8],
        learning_rates=[0.03, 0.1],
        n_estimators_list=[400, 800],
        subsamples=[0.8, 1.0],
        colsample_bytree_list=[0.8, 1.0],
        reg_lambdas=[1.0, 2.0],
        reg_alphas=[0.0, 0.5],
        min_child_weights=[1.0, 5.0],
        gammas=[0.0, 1.0],
        max_bins=[256],
        scale_pos_weight_list=[1.0],  # set >1.0 for imbalanced data
        seed=123,
        cfg=TrainConfigXGB(early_stopping_rounds=50, eval_metric="logloss", val_frac=0.2),
        groups=groups,
        stratified_groups=True,
        save_dir="grid_runs_xgb",
    )

    print("\n===== Best XGB Configuration =====")
    print(best["config"])
    print("Best metrics on val split:", best["metrics"])
    print("Model saved at:", best["model_path"])
    print("Top-5 file:", best["top5_path"])
