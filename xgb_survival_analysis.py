# run_discrete_time_xgb.py
from __future__ import annotations
import os
import json
import io
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)
from lifelines.utils import concordance_index

# XGBoost
import xgboost as xgb

# Headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# --- add near the top, with other imports/utilities ---
def _coerce_numeric_inplace(df: pd.DataFrame, cols: list[str]) -> None:
    """Inplace: make columns numeric (float32) where possible; keep bools as is."""
    for c in cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_bool_dtype(df[c]):
            # leave as bool
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # after coercion, cast to float32 for XGB
        if not pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype("float32")
    # fill any remaining NaNs (safer for XGB)
    df[cols] = df[cols].fillna(0)

def _numeric_or_coercible_feature_cols(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    """
    Like _numeric_feature_cols, but if an object column becomes numeric after coercion
    on a sample, we treat it as numeric. Then we coerce full-frame before returning.
    """
    # Start with numeric + bool
    num = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    base = [c for c in num if c not in exclude]

    # Consider object columns that look numeric
    cand = [c for c in df.columns if (c not in exclude) and (c not in base)]
    looks_numeric = []
    for c in cand:
        if pd.api.types.is_object_dtype(df[c]):
            # quick sniff using first 1000 non-null values
            s = pd.to_numeric(df[c], errors="coerce")
            # treat as numeric if at least half of the sampled values are numeric
            if s.notna().mean() >= 0.5:
                looks_numeric.append(c)

    cols = base + looks_numeric
    # Coerce in place now so downstream is strictly numeric
    _coerce_numeric_inplace(df, cols)
    # Drop any constant columns after coercion
    cols = [c for c in cols if df[c].std(ddof=0) > 0]
    return cols


# ----------------------------------------------------
# Config
# ----------------------------------------------------
DATA_DIR = Path("data/MIMIC-ED")
DT_PATH  = DATA_DIR / "discrete_time_30min_train.csv"   # <-- from your formatter

OUT_DIR   = Path("outputs_discrete_xgb")
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Columns
ID_COL       = "stay_id"
EVENT_COL    = "event"
TBIN_COL     = "t_bin"
LM_TIME_COL  = "landmark_charttime"     # present if you kept include_landmark_cols=True
HRS_SINCE_ADM_COL = "hours_since_intime"  # optional (present if enabled)

# Training / eval
RNG         = 42
TEST_SIZE   = 0.2
USE_SCALER  = False     # XGBoost doesn’t need it; set True if you want z-scoring

# Discrete-time horizon
BIN_MINUTES = 30.0
BIN_HOURS   = BIN_MINUTES / 60.0
K_HOURS     = 10.0       # patient-level C-index horizon from latest snapshot (e.g., 6h)
K_BINS      = int(np.ceil(K_HOURS / BIN_HOURS))

# XGB params
N_ESTIMATORS       = 400
MAX_DEPTH          = 4
LEARNING_RATE      = 0.1
SUBSAMPLE          = 0.7
COLSAMPLE_BYTREE   = 0.9
EARLY_STOP_ROUNDS  = 100
EVAL_METRIC        = "aucpr"

# ----------------------------------------------------
# Utilities
# ----------------------------------------------------
def _numeric_feature_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cols = [c for c in cols if c not in exclude]
    # drop constant cols
    return [c for c in cols if df[c].std(ddof=0) > 0]

def _save_fig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _save_current_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def _save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _xgb_gpu_kwargs():
    ver = tuple(int(x) for x in xgb.__version__.split(".")[:2])
    if ver >= (2, 0):
        return dict(tree_method="hist", device="cuda")
    else:
        return dict(tree_method="gpu_hist", gpu_id=0)

def _fit_scaler_if_needed(tr_df: pd.DataFrame, te_df: pd.DataFrame, feats: List[str], name: str|None=None):
    if not USE_SCALER:
        return None
    scaler = StandardScaler()
    tr_df[feats] = scaler.fit_transform(tr_df[feats].astype("float64"))
    te_df[feats] = scaler.transform(te_df[feats].astype("float64"))
    if name:
        with open(OUT_DIR / f"{name}_scaler.pkl", "wb") as f:
            pickle.dump({"features": feats, "scaler": scaler}, f)
    return scaler

def _plot_roc_pr(y_true: np.ndarray, y_score: np.ndarray, prefix: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_score):.3f}")
    plt.plot([0,1],[0,1],"--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC — {prefix}")
    plt.legend(loc="lower right")
    _save_current_fig(PLOTS_DIR / f"{prefix}_roc.png")

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"AP={average_precision_score(y_true, y_score):.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR — {prefix}")
    plt.legend(loc="lower left")
    _save_current_fig(PLOTS_DIR / f"{prefix}_pr.png")

def _predict_event_prob_within_K(
    model: xgb.XGBClassifier, base_feats: List[str], snapshot_row: pd.Series, K_bins: int
) -> float:
    base_vals = snapshot_row[base_feats].to_frame().T.copy()
    # Coerce snapshot numerics (important if the CSV carried strings like "0"/"1")
    _coerce_numeric_inplace(base_vals, base_feats)

    G = pd.concat([base_vals.assign(**{TBIN_COL: k}) for k in range(1, K_bins+1)], ignore_index=True)
    _coerce_numeric_inplace(G, [TBIN_COL] + base_feats)

    proba = model.predict_proba(G[[TBIN_COL] + base_feats])[:, 1]
    if len(proba) > K_bins:
        proba = proba.reshape(-1, K_bins).mean(axis=0)
    surv = float(np.prod(1.0 - proba))
    return 1.0 - surv


def _latest_snapshot_per_stay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one latest landmark row per stay, using LM_TIME_COL (if present)
    or the max observed TBIN within each stay/landmark surrogate.
    """
    if LM_TIME_COL in df.columns:
        order = df.sort_values([ID_COL, LM_TIME_COL, TBIN_COL], kind="mergesort")
        snaps = order.groupby(ID_COL, as_index=False).tail(1)
        return snaps.reset_index(drop=True)
    else:
        # Fallback: approximate latest snapshot by using the row with the largest t_bin per stay
        order = df.sort_values([ID_COL, TBIN_COL], kind="mergesort")
        snaps = order.groupby(ID_COL, as_index=False).tail(1)
        return snaps.reset_index(drop=True)

def _build_snapshot_outcomes_from_pp(
    df_pp: pd.DataFrame, K_bins: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (snaps, outcomes) for patient-level C-index using the latest snapshot per stay.
    For each stay:
      - Find its latest snapshot landmark (by landmark_charttime if present, else heuristic).
      - Within that landmark, look at all emitted person-period rows (same stay & same landmark).
      - Duration from the landmark to terminal = (#bins until terminal) * BIN_HOURS.
      - Event indicator = 1 if any emitted row for that landmark has event==1, else 0.
    Returns:
      snaps:    latest snapshot rows per stay (one row each)
      outcomes: columns ['stay_id','duration_hours','event']
    """
    if LM_TIME_COL not in df_pp.columns:
        raise ValueError(
            f"{LM_TIME_COL} not found in person-period file; enable include_landmark_cols=True in the formatter."
        )

    # latest snapshot time per stay
    idx_latest = (
        df_pp.groupby(ID_COL, as_index=False)[LM_TIME_COL]
        .max()
        .rename(columns={LM_TIME_COL: "latest_lm"})
    )
    df_pp2 = df_pp.merge(idx_latest, on=ID_COL, how="inner")

    # rows corresponding to each stay's latest landmark
    latest_rows = df_pp2[df_pp2[LM_TIME_COL] == df_pp2["latest_lm"]].copy()
    if latest_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    # For outcomes per stay (from that landmark)
    agg = latest_rows.groupby(ID_COL).agg(
        max_bin=(TBIN_COL, "max"),
        any_event=(EVENT_COL, "max"),
        # if an event is present, the first event bin:
        event_bin=(EVENT_COL, lambda s: np.nan if s.sum()==0 else int(np.argmax(s.values != 0) + 1))
    ).reset_index()

    # duration in hours from latest landmark to terminal:
    # - if event: event_bin * BIN_HOURS
    # - else:     max_bin * BIN_HOURS
    def _dur(row):
        if int(row["any_event"]) == 1 and not pd.isna(row["event_bin"]):
            return float(int(row["event_bin"]) * BIN_HOURS)
        else:
            return float(int(row["max_bin"]) * BIN_HOURS)

    outcomes = agg.assign(
        duration_hours=agg.apply(_dur, axis=1).astype(float),
        event=agg["any_event"].astype(int)
    )[[ID_COL, "duration_hours", "event"]]

    # Build 1-row snapshots per stay (covariates at the latest landmark)
    # Take one arbitrary line per stay at the latest landmark (t_bin doesn’t matter for covariates)
    # Prefer the smallest t_bin there to reduce redundancy.
    latest_rows = latest_rows.sort_values([ID_COL, TBIN_COL], kind="mergesort")
    snaps = latest_rows.groupby(ID_COL, as_index=False).head(1).reset_index(drop=True)

    return snaps, outcomes

def run_discrete_time_xgb(dt_path: Path) -> dict:
    if not dt_path.exists():
        raise FileNotFoundError(f"Person-period file not found: {dt_path}")

    # Read once to inspect columns, then parse datetime cols if present
    head_cols = pd.read_csv(dt_path, nrows=0).columns.tolist()
    parse_cols = [c for c in [LM_TIME_COL] if c in head_cols]
    df = pd.read_csv(dt_path, parse_dates=parse_cols)

    need = {ID_COL, TBIN_COL, EVENT_COL}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in person-period data: {missing}")

    # Build exclude list; DO NOT exclude 'hours_since_intime' if you want it as a feature
    exclude = [ID_COL, TBIN_COL, EVENT_COL]
    if LM_TIME_COL in df.columns:
        exclude.append(LM_TIME_COL)

    # >>> NEW: choose features with coercion (handles 'object' numerics and one-hots) <<<
    base_feats = _numeric_or_coercible_feature_cols(df, exclude=exclude)

    # Ensure TBIN and EVENT are numeric
    _coerce_numeric_inplace(df, [TBIN_COL, EVENT_COL])

    # Group split
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RNG)
    idx = np.arange(len(df))
    tr_idx, te_idx = next(gss.split(idx, groups=df[ID_COL].values))
    tr, te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()

    # Optional scaling (after coercion; not required for XGB)
    _fit_scaler_if_needed(tr, te, base_feats, name="xgb_dt")

    # Recompute pos/neg after coercion just in case
    pos = int((tr[EVENT_COL] == 1).sum())
    neg = int((tr[EVENT_COL] == 0).sum())
    spw = (neg / max(1, pos)) if pos > 0 else 1.0

    mdl = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        random_state=RNG,
        scale_pos_weight=spw,
        eval_metric=EVAL_METRIC,
        verbosity=1,
        **_xgb_gpu_kwargs(),
    )

    # Ensure features used for training are numeric in both splits
    _coerce_numeric_inplace(tr, [TBIN_COL] + base_feats)
    _coerce_numeric_inplace(te, [TBIN_COL] + base_feats)

    X_tr = tr[[TBIN_COL] + base_feats]
    y_tr = tr[EVENT_COL].astype(int).values
    X_te = te[[TBIN_COL] + base_feats]
    y_te = te[EVENT_COL].astype(int).values

    # ... (rest of fit/metrics unchanged)


    # Early stopping (new-style callback if available; otherwise legacy arg)
    fit_common = dict(eval_set=[(X_te, y_te)], verbose=True)
    fitted = False
    try:
        from xgboost.callback import EarlyStopping
        es = EarlyStopping(rounds=EARLY_STOP_ROUNDS, metric_name=EVAL_METRIC, save_best=True, maximize=True)
        mdl.fit(X_tr, y_tr, callbacks=[es], **fit_common)
        fitted = True
    except Exception:
        pass
    if not fitted:
        try:
            mdl.fit(X_tr, y_tr, early_stopping_rounds=EARLY_STOP_ROUNDS, **fit_common)
            fitted = True
        except TypeError:
            mdl.fit(X_tr, y_tr, **fit_common)

    # Row-level metrics
    p_tr = mdl.predict_proba(X_tr)[:, 1]
    p_te = mdl.predict_proba(X_te)[:, 1]
    metrics = {
        "train_ap": float(average_precision_score(y_tr, p_tr)),
        "test_ap":  float(average_precision_score(y_te, p_te)),
        "train_auc": float(roc_auc_score(y_tr, p_tr)) if len(np.unique(y_tr)) > 1 else None,
        "test_auc":  float(roc_auc_score(y_te, p_te)) if len(np.unique(y_te)) > 1 else None,
        "n_train_pos": int((y_tr == 1).sum()),
        "n_train_neg": int((y_tr == 0).sum()),
        "n_test_pos":  int((y_te == 1).sum()),
        "n_test_neg":  int((y_te == 0).sum()),
    }

    # Plots
    try:
        _plot_roc_pr(y_te, p_te, prefix="xgb_discrete_test")
    except Exception as e:
        print(f"[WARN] Plotting ROC/PR failed: {e}")

    # Feature importance
    try:
        fi = {k: float(v) for k, v in zip([TBIN_COL] + base_feats, mdl.feature_importances_)}
        metrics["feature_importance"] = fi
        pd.Series(fi).sort_values(ascending=False).to_csv(OUT_DIR / "feature_importance.csv")
    except Exception as e:
        print(f"[WARN] Could not extract feature importances: {e}")

    # -------------------------------
    # Patient-level Harrell's C-index
    # -------------------------------
    # Compute snapshots/outcomes from the SAME person-period file (no external ED/Pyxis needed):
    snaps_all, outcomes_all = _build_snapshot_outcomes_from_pp(df, K_BINS)
    snaps_tr  = snaps_all[snaps_all[ID_COL].isin(set(tr[ID_COL]))].reset_index(drop=True)
    snaps_te  = snaps_all[snaps_all[ID_COL].isin(set(te[ID_COL]))].reset_index(drop=True)

    out_tr = outcomes_all[outcomes_all[ID_COL].isin(set(snaps_tr[ID_COL]))].reset_index(drop=True)
    out_te = outcomes_all[outcomes_all[ID_COL].isin(set(snaps_te[ID_COL]))].reset_index(drop=True)

    # Build base snapshot feature rows (without t_bin) for risk scoring
    # We take the covariates from the snapshot rows; remove t_bin/event/id from the feature set.
    base_cols = [c for c in base_feats]  # already excludes t_bin/event/id

    def _score_snaps(snaps_df: pd.DataFrame) -> pd.DataFrame:
        if snaps_df.empty:
            return pd.DataFrame(columns=[ID_COL, f"prob_event_within_{K_BINS}bins"])
        scores = []
        for _, row in snaps_df.iterrows():
            probK = _predict_event_prob_within_K(mdl, base_cols, row, K_bins=K_BINS)
            scores.append((int(row[ID_COL]), probK))
        return pd.DataFrame(scores, columns=[ID_COL, f"prob_event_within_{K_BINS}bins"])
        
    def _score_snaps_vectorized(model: xgb.XGBClassifier, snaps_df: pd.DataFrame,
                                base_cols: list[str], K_bins: int) -> pd.DataFrame:
        if snaps_df.empty:
            return pd.DataFrame(columns=[ID_COL, f"prob_event_within_{K_bins}bins"])

        # Build one big design matrix: repeat each row K times with t_bin=1..K
        snapped = snaps_df[[ID_COL] + base_cols].copy()
        # ensure numeric & compact
        snapped[base_cols] = snapped[base_cols].astype("float32")
        ids = snapped[ID_COL].to_numpy()
        base_block = snapped[base_cols].to_numpy()

        # Repeat base rows K times
        X_base = np.repeat(base_block, K_bins, axis=0)
        # Build t_bin column
        t_bins = np.tile(np.arange(1, K_bins + 1, dtype=np.int16), reps=len(snapped))
        # Concatenate into a DataFrame in the same feature order used in training
        G = pd.DataFrame(X_base, columns=base_cols)
        G.insert(0, TBIN_COL, t_bins)

        # Predict proba in one shot
        proba = model.predict_proba(G[[TBIN_COL] + base_cols])[:, 1].astype("float32")

        # Reshape to (n_snaps, K)
        proba = proba.reshape(len(snapped), K_bins)
        surv  = np.prod(1.0 - proba, axis=1)
        riskK = 1.0 - surv

        return pd.DataFrame({
            ID_COL: ids,
            f"prob_event_within_{K_bins}bins": riskK
        })


    risk_tr = _score_snaps_vectorized(mdl, snaps_tr, base_feats, K_BINS)
    risk_te = _score_snaps_vectorized(mdl, snaps_te, base_feats, K_BINS)

    # Merge with outcomes and compute C-index
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

    # Save artifacts
    mdl.save_model(str(OUT_DIR / "xgb_discrete_model.json"))
    with open(OUT_DIR / "xgb_discrete_features.pkl", "wb") as f:
        pickle.dump({"order": [TBIN_COL] + base_feats, "base_features": base_feats}, f)
    _save_json(metrics, OUT_DIR / "metrics.json")

    # Final prints
    print("\n[XGB Discrete-Time] Row-level metrics")
    print(f"  Train AP={metrics['train_ap']:.4f} | AUC={metrics['train_auc'] if metrics['train_auc'] is not None else 'NA'}")
    print(f"  Test  AP={metrics['test_ap']:.4f} | AUC={metrics['test_auc'] if metrics['test_auc'] is not None else 'NA'}")

    print("\n[XGB Discrete-Time] Patient-level C-index (latest snapshot, horizon K)")
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
    run_discrete_time_xgb(DT_PATH)

if __name__ == "__main__":
    main()
