# xgb_discrete_inference.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xgboost as xgb
import pickle

# -------------------------
# Config
# -------------------------
MODEL_DIR   = Path("outputs_discrete_xgb")  # where you saved artifacts during training
MODEL_PATH  = MODEL_DIR / "xgb_discrete_model.json"
FEATS_PATH  = MODEL_DIR / "xgb_discrete_features.pkl"  # {"order":[tbin + base_feats], "base_features":[...]}
INPUT_JSON  = Path("folder/Design/src/mockData.json")
OUTPUT_JSON = Path("folder/Design/src/mockData_xgb.json")

ID_COL      = "stay_id"         # not used by dashboard, but kept for consistency
TBIN_COL    = "t_bin"
EVENT_COL   = "event"
LM_TIME_COL = "landmark_charttime"    # not needed for inference
BIN_MINUTES = 30.0
BIN_HOURS   = BIN_MINUTES / 60.0
K_HOURS     = 5.0                    # <<< pick your horizon for risk (e.g., 3.0 or 6.0)
K_BINS      = int(np.ceil(K_HOURS / BIN_HOURS))

# Map your dashboard column names -> model feature names (rename if needed)
RENAME_MAP = {
    "temp": "temperature",
    "hr":   "heartrate",
    # add other renames if your dashboard keys differ
}

# -------------------------
# Helpers
# -------------------------
def _coerce_numeric_inplace(df: pd.DataFrame, cols: List[str]) -> None:
    """Make columns numeric/float32 (leave bools). Fill NaNs with 0."""
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

def _ensure_columns(df: pd.DataFrame, needed: List[str]) -> pd.DataFrame:
    missing = [c for c in needed if c not in df.columns]
    for c in missing:
        df[c] = 0.0
    return df

def _score_within_K_vectorized(model: xgb.XGBClassifier, snaps_df: pd.DataFrame,
                               base_feats: List[str], K_bins: int) -> np.ndarray:
    """Vectorized K-horizon discrete-time risk: 1 - prod_k (1 - p_k)."""
    if snaps_df.empty:
        return np.array([], dtype="float32")

    # Build big design matrix: repeat each row K times & add t_bin 1..K
    base_block = snaps_df[base_feats].to_numpy(dtype="float32")
    X_base = np.repeat(base_block, K_bins, axis=0)
    t_bins = np.tile(np.arange(1, K_bins + 1, dtype=np.int16), reps=len(snaps_df))

    G = pd.DataFrame(X_base, columns=base_feats)
    G.insert(0, TBIN_COL, t_bins)

    proba = model.predict_proba(G[[TBIN_COL] + base_feats])[:, 1].astype("float32")  # (N*K,)
    proba = proba.reshape(len(snaps_df), K_bins)
    surv  = np.prod(1.0 - proba, axis=1)
    riskK = 1.0 - surv
    return riskK

# -------------------------
# Main inference
# -------------------------
def main():
    # 1) Load artifacts
    with open(FEATS_PATH, "rb") as f:
        feats_payload = pickle.load(f)
    order = feats_payload["order"]           # [t_bin] + base_features (training order)
    base_feats = feats_payload["base_features"]

    model = xgb.XGBClassifier(device="cpu")  # force CPU for inference
    model.load_model(str(MODEL_PATH))

    # 2) Load dashboard JSON
    df = pd.read_json(INPUT_JSON, lines=False)
    # rename dashboard columns to match training, if needed
    if RENAME_MAP:
        df = df.rename(columns=RENAME_MAP)

    # 3) Align features: add missing columns, coerce types, keep order
    df = _ensure_columns(df, base_feats)
    _coerce_numeric_inplace(df, base_feats)

    # 4) Vectorized K-horizon risk from snapshots
    riskK = _score_within_K_vectorized(model, df, base_feats, K_bins=K_BINS)

    # Convert to 0–100 score like your current UI (round to 0.1)
    sepsis_scores = np.round(riskK * 100.0, 1)

    # 5) Update JSON fields (mirror your Cox update)
    df["sepsisScore"] = sepsis_scores
    # If your UI expects a "hazardRate", we can store the same risk number
    df["hazardRate"] = sepsis_scores

    # Priority rank: higher risk first
    df["priorityRank"] = df["hazardRate"].rank(ascending=False, method="first").astype(int)

    # Optional: preserve your trend updates (use your original field names if different)
    # (This assumes `trends` is an array of dicts per row, last entry is the latest.)
    for i in range(len(df)):
        try:
            df.at[i, "trends"][-1]["temp"]      = float(df.at[i, "temperature"])
            df.at[i, "trends"][-1]["heartRate"] = float(df.at[i, "heartrate"])
            if "lactate" in df.columns:
                df.at[i, "trends"][-1]["lactate"]   = float(df.at[i, "lactate"])
        except Exception:
            # trends may be missing for some rows; skip silently
            pass

    # 6) Save back to JSON for the dashboard
    df.to_json(OUTPUT_JSON, orient="records", lines=False, indent=1)

    print(f"✅ Updated {OUTPUT_JSON}")
    print(f"   Horizon K = {K_HOURS:.1f}h ({K_BINS} bins of {BIN_MINUTES:.0f}m)")
    print(f"   Mean risk: {sepsis_scores.mean():.2f}  | Max: {sepsis_scores.max():.2f}")

if __name__ == "__main__":
    main()
