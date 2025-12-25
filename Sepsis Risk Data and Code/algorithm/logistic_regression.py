# grid_search_logreg.py
import time
import random
import itertools
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

# --- sklearn imports (CPU only, no torch) ---
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

# ---------------------------
# Repro
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

# ---------------------------
# Data splitting (mirrors your torch version)
# ---------------------------
def make_split(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.2,
    groups: Optional[np.ndarray] = None,
    stratified_groups: bool = False,
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (train_idx, val_idx), respecting groups if provided.
    If stratified_groups=True, uses StratifiedGroupKFold(5) and takes the first split (~20% val).
    """
    N = len(X)
    assert len(y) == N
    if groups is not None:
        assert len(groups) == N

    if groups is None:
        # Simple, deterministic random split
        rng = np.random.default_rng(seed)
        idx = np.arange(N)
        rng.shuffle(idx)
        n_val = int(round(N * val_frac))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        return train_idx, val_idx

    # Group-aware
    if stratified_groups:
        # roughly 1/5 for validation
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train_idx, val_idx = next(sgkf.split(np.zeros(N), y, groups=groups))
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        train_idx, val_idx = next(gss.split(np.zeros(N), y, groups=groups))
    return train_idx, val_idx

# ---------------------------
# Evaluation helper
# ---------------------------
def evaluate(model: Pipeline, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Computes log_loss (binary), AUROC, and accuracy at 0.5.
    """
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(np.float32)

    metrics = {
        "loss": log_loss(y, probs, labels=[0, 1]),
        "auroc": roc_auc_score(y, probs) if len(np.unique(y)) == 2 else float("nan"),
        "acc": accuracy_score(y, preds),
    }
    return metrics

# ---------------------------
# Single training run
# ---------------------------
@dataclass
class TrainConfig:
    # Mirrors your API surface; not all fields used for sklearn
    max_iter: int = 3000
    n_jobs: int = -1  # parallelism for solvers that support it (saga, lbfgs)
    # No early stopping here; LR is convex and fast

def train_one(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
) -> Dict[str, float]:
    t0 = time.time()
    # Configure solver iterations if present
    lr = pipeline.named_steps["clf"]
    lr.set_params(max_iter=cfg.max_iter, n_jobs=cfg.n_jobs if "n_jobs" in lr.get_params() else None)

    pipeline.fit(X_train, y_train)
    val_metrics = evaluate(pipeline, X_val, y_val)
    elapsed = time.time() - t0
    print(
        f"[FIT] val_loss={val_metrics['loss']:.4f} "
        f"val_acc={val_metrics['acc']:.4f} val_auroc={val_metrics['auroc']:.4f} ({elapsed:.1f}s)"
    )
    return val_metrics

# ---------------------------
# Grid Search (manual, like your torch version)
# ---------------------------
def grid_search(
    X,
    y,
    *,
    # Hyperparameter options
    solvers: List[str],
    penalties_by_solver: Dict[str, List[str]],
    Cs: List[float],
    l1_ratios: List[Optional[float]],  # only used when penalty="elasticnet" with solver="saga"
    class_weights: List[Optional[str]],  # [None, "balanced"]
    # Split/training controls
    val_frac: float = 0.2,
    groups: Optional[np.ndarray] = None,
    stratified_groups: bool = True,
    seed: int = 42,
    max_iter: int = 3000,
    n_jobs: int = -1,
):
    """
    Mirrors your torch grid search:
    - Build group-aware train/val split
    - Try combos
    - Select by smallest validation log-loss
    - Refit best on FULL data and report metrics (on full data, to match your flow)
    """
    set_seed(seed)

    # Coerce to numpy
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)

    N, D = X.shape
    print(f"[INFO] X shape: {(N, D)}, y pos={y.sum()} neg={N - y.sum()}")

    # Split indices once (like your torch code calling make_loaders per trial)
    train_idx, val_idx = make_split(
        X, y, val_frac=val_frac, groups=groups, stratified_groups=stratified_groups, seed=7
    )
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Base pipeline: Standardize -> LogisticRegression
    # We will vary solver/penalty/C/l1_ratio/class_weight
    def make_pipeline(solver, penalty, C, l1_ratio, class_weight):
        lr_kwargs = dict(
            solver=solver,
            penalty=penalty,
            C=C,
            random_state=seed,
            class_weight=class_weight,
            # max_iter/n_jobs set in train_one
        )
        if penalty == "elasticnet":
            lr_kwargs["l1_ratio"] = l1_ratio
        clf = LogisticRegression(**lr_kwargs)
        return Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", clf),
            ]
        )

    # Build grid manually to honor solver/penalty compatibility
    combos = []
    for solver in solvers:
        for penalty in penalties_by_solver.get(solver, []):
            for C in Cs:
                if penalty == "elasticnet":
                    for l1r in l1_ratios:
                        for cw in class_weights:
                            combos.append((solver, penalty, C, l1r, cw))
                else:
                    for cw in class_weights:
                        combos.append((solver, penalty, C, None, cw))

    print(f"[INFO] Grid size: {len(combos)}")

    best: Dict[str, Any] = {
        "val_loss": float("inf"),
        "config": None,
        "model": None,
        "metrics": None,
    }

    cfg = TrainConfig(max_iter=max_iter, n_jobs=n_jobs)

    for i, (solver, penalty, C, l1r, cw) in enumerate(combos, start=1):
        print(
            f"\n=== Trial {i}/{len(combos)} ===\n"
            f"solver={solver} penalty={penalty} C={C} l1_ratio={l1r} class_weight={cw}"
        )
        # Skip incompatible combos defensively (sklearn would error)
        if penalty == "l1" and solver not in ("liblinear", "saga"):
            print("[SKIP] l1 requires solver in {'liblinear','saga'}")
            continue
        if penalty == "elasticnet" and solver != "saga":
            print("[SKIP] elasticnet requires solver='saga'")
            continue
        if penalty == "none" and solver in ("liblinear",):
            print("[SKIP] penalty='none' not supported by liblinear")
            continue

        pipe = make_pipeline(solver, penalty, C, l1r, cw)
        metrics = train_one(pipe, X_train, y_train, X_val, y_val, cfg)

        improved = metrics["loss"] < best["val_loss"]
        print(
            f"[RESULT] val_loss={metrics['loss']:.4f} val_acc={metrics['acc']:.4f} "
            f"val_auroc={metrics['auroc']:.4f}"
        )
        if improved:
            best.update(
                {
                    "val_loss": metrics["loss"],
                    "config": {
                        "solver": solver,
                        "penalty": penalty,
                        "C": C,
                        "l1_ratio": l1r,
                        "class_weight": cw,
                        "max_iter": max_iter,
                        "n_jobs": n_jobs,
                        "val_frac": val_frac,
                        "stratified_groups": stratified_groups,
                        "seed": seed,
                    },
                    "model": pipe,  # keep the fitted pipeline for convenience (val-trained)
                    "metrics": metrics,
                }
            )
            print("[BEST] Updated best configuration.")

    print("\n===== Best Configuration (by smallest val log-loss) =====")
    print(best["config"])
    print("Best val metrics:", best["metrics"])

    # Re-train on FULL data with best hyperparameters (to mirror your 'full fit')
    print("\n[INFO] Re-training best logistic regression on full data...")
    bc = best["config"]
    final_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    solver=bc["solver"],
                    penalty=bc["penalty"],
                    C=bc["C"],
                    l1_ratio=bc["l1_ratio"] if bc["penalty"] == "elasticnet" else None,
                    class_weight=bc["class_weight"],
                    random_state=bc["seed"],
                    max_iter=bc["max_iter"],
                    n_jobs=bc["n_jobs"] if bc["solver"] in ("lbfgs", "saga", "newton-cg", "sag") else None,
                ),
            ),
        ]
    )
    final_pipe.fit(X, y)
    # Evaluate on full data (same as your final torch call which used the same loader twice)
    final_metrics = evaluate(final_pipe, X, y)
    print("Final metrics on full data:", final_metrics)

    best["model"] = final_pipe  # overwrite with the refit-on-full pipeline
    best["metrics"] = final_metrics
    return best

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    import pandas as pd

    # Load your same dataset
    df = pd.read_csv("data/MIMIC-ED/event_level_training_data.csv")
    X = df.drop(columns=["is_sepsis", "stay_id"]).to_numpy(dtype=np.float32)
    y = df["is_sepsis"].to_numpy(dtype=np.int64)
    groups = df["stay_id"].to_numpy()

    print(f"Data shape: X={X.shape}, y={y.shape}, Positives={y.sum()}, Negatives={len(y)-y.sum()}")

    # Hyperparameter options:
    # - We include widely-used solvers and compatible penalties.
    # - 'elasticnet' only works with 'saga'.
    # - Add/trim to control grid size.
    solvers = ["liblinear", "lbfgs", "saga"]
    penalties_by_solver = {
        "liblinear": ["l1", "l2"],         # no 'none', no 'elasticnet'
        "lbfgs": ["l2"],            # no l1/elasticnet
        "saga": ["l1", "l2", "elasticnet"],
    }
    Cs = [0.01, 0.1, 1.0, 3.0, 10.0]
    l1_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]  # only used for elasticnet+saga
    class_weights = [None, "balanced"]

    best = grid_search(
        X=X,
        y=y,
        solvers=solvers,
        penalties_by_solver=penalties_by_solver,
        Cs=Cs,
        l1_ratios=l1_ratios,
        class_weights=class_weights,
        val_frac=0.2,
        groups=groups,
        stratified_groups=True,
        seed=42,
        max_iter=100,
        n_jobs=-1,
    )

    # If you want to reuse the best model later:
    mdl: Pipeline = best["model"]  # This is a fitted sklearn Pipeline (StandardScaler + LogisticRegression)
    print("[DONE] Best sklearn logistic regression restored.")

    # output best config and metrics
    print("Best hyperparameters:", best["config"])
    print("Best metrics on full data:", best["metrics"])
    # save the model if desired
    import joblib
    joblib.dump(mdl, "best_logistic_regression_model.joblib")
