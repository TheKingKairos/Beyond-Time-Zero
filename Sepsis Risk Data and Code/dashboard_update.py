#!/usr/bin/env python3
"""
Update dashboard JSON with required sepsis features and compute sepsisScore
using a saved logistic regression model.

- Input:
    - dashboard JSON (list of dicts, must include "patientId")
    - dataset file with required features (CSV or Parquet)
    - best_logistic_regression_model.joblib (scikit-learn LogisticRegression)

- Output:
    - updated JSON with required features + recalculated sepsisScore

Assumptions:
- The dataset has a "patientId" column to join on.
- If some required columns are missing, we create them and fill with
  dataset medians (numeric) or 0 (binary flags), as a safe default.
- sepsisScore is scaled to 0–100 from predicted probability of class 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from joblib import load as joblib_load


# ---- Configuration (edit these paths) ---------------------------------------
DASHBOARD_JSON_PATH = Path("folder/Design/src/mockData.json")  # your current JSON (input)
DATASET_PATH        = Path("data/MIMIC-ED/simple_event_level_training_data.csv")  # or .csv
MODEL_PATH          = Path("best_logistic_regression_model.joblib")
OUTPUT_JSON_PATH    = Path("folder/Design/src/mockData_with_features.json")

# If your dataset is a CSV, set this to True
DATASET_IS_CSV = True
CSV_READ_KWARGS = dict()  # e.g., {"dtype": {...}}


# ---- Required feature columns (exact names expected by the model) ------------
REQUIRED_FEATURES: List[str] = [
    'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain',
    'rhythm_flag', 'is_white', 'is_black', 'is_asian', 'is_hispanic', 'is_other_race',
    'gender_F', 'gender_M',
    'arrival_transport_AMBULANCE', 'arrival_transport_HELICOPTER',
    'arrival_transport_OTHER', 'arrival_transport_UNKNOWN', 'arrival_transport_WALK IN',
    'lactate', 'wbc', 'time_since_adm',
    'gsn_16599.0', 'gsn_43952.0', 'gsn_4490.0', 'gsn_66419.0', 'gsn_61716.0'
]

# Optional helper: map common alternate column names in your dataset to the exact
# REQUIRED_FEATURES names. Extend as needed.
ALIAS_MAP: Dict[str, str] = {
    # vitals
    'temp': 'temperature',
    'hr': 'heartrate',
    'resp_rate': 'resprate',
    'respiration': 'resprate',
    'spo2': 'o2sat',
    'o2_sat': 'o2sat',
    'sbp_mmHg': 'sbp',
    'dbp_mmHg': 'dbp',
    # demographics/flags
    'white': 'is_white',
    'black': 'is_black',
    'asian': 'is_asian',
    'hispanic': 'is_hispanic',
    'other_race': 'is_other_race',
    'gender_female': 'gender_F',
    'gender_male': 'gender_M',
    # labs
    'wbc_count': 'wbc',
    'time_since_admit_hours': 'time_since_adm',
}


# ---- I/O helpers -------------------------------------------------------------
def load_dashboard_json(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dashboard JSON must be a list of records.")
    df = pd.DataFrame(data)
    if "patientId" not in df.columns:
        raise ValueError("Dashboard JSON must include 'patientId' for each entry.")
    return df


def load_feature_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet" and not DATASET_IS_CSV:
        return pd.read_parquet(path)
    return pd.read_csv(path, **CSV_READ_KWARGS).rename({"stay_id": "patientId"})


def unify_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases to required names; keeps originals if unmatched."""
    rename_map = {src: dst for src, dst in ALIAS_MAP.items() if src in df.columns and dst not in df.columns}
    return df.rename(columns=rename_map)


def ensure_required_columns(df: pd.DataFrame, ref_df_for_medians: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has all REQUIRED_FEATURES. For missing numeric columns, fill with
    medians from ref_df_for_medians (if available) or 0. For binary flags,
    default to 0.
    """
    df = df.copy()

    # Determine which columns are treated as binary flags (0/1)
    binary_like = {
        'rhythm_flag', 'is_white', 'is_black', 'is_asian', 'is_hispanic', 'is_other_race',
        'gender_F', 'gender_M',
        'arrival_transport_AMBULANCE', 'arrival_transport_HELICOPTER',
        'arrival_transport_OTHER', 'arrival_transport_UNKNOWN', 'arrival_transport_WALK IN',
        'gsn_16599.0', 'gsn_43952.0', 'gsn_4490.0', 'gsn_66419.0', 'gsn_61716.0'
    }

    for col in REQUIRED_FEATURES:
        if col not in df.columns:
            if col in binary_like:
                df[col] = 0
            else:
                # Try median from reference; fallback to 0
                if col in ref_df_for_medians.columns and pd.api.types.is_numeric_dtype(ref_df_for_medians[col]):
                    df[col] = ref_df_for_medians[col].median()
                else:
                    df[col] = 0.0

        # Coerce dtypes: binary to {0,1}, numerics to float
        if col in binary_like:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0, upper=1).astype(int)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float).fillna(0.0)

    return df


def align_feature_order(df_features: pd.DataFrame, model) -> pd.DataFrame:
    """
    Align feature column order to what the model expects (if available).
    Fallback to REQUIRED_FEATURES.
    """
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    else:
        expected = REQUIRED_FEATURES

    missing_for_model = [c for c in expected if c not in df_features.columns]
    if missing_for_model:
        raise ValueError(
            "The following features expected by the model are missing after preparation: "
            + ", ".join(missing_for_model)
        )
    return df_features[expected]


# ---- Core pipeline -----------------------------------------------------------
def main():
    # 1) Load inputs
    print(f"Loading dashboard: {DASHBOARD_JSON_PATH}")
    df_dash = load_dashboard_json(DASHBOARD_JSON_PATH)

    print(f"Loading dataset: {DATASET_PATH}")
    df_data_raw = load_feature_dataset(DATASET_PATH)

    if "patientId" not in df_data_raw.columns:
        raise ValueError("The dataset must have a 'patientId' column for joining.")

    # 2) Prepare dataset feature columns
    df_data = unify_column_names(df_data_raw)

    # If arrival_transport is categorical in a single column, one-hot it into the expected names.
    if "arrival_transport" in df_data.columns:
        one_hot = pd.get_dummies(df_data["arrival_transport"], prefix="arrival_transport", dtype=int)
        # Ensure all five expected transport columns exist
        for col in [
            "arrival_transport_AMBULANCE",
            "arrival_transport_HELICOPTER",
            "arrival_transport_OTHER",
            "arrival_transport_UNKNOWN",
            "arrival_transport_WALK IN",
        ]:
            if col not in one_hot.columns:
                one_hot[col] = 0
        df_data = pd.concat([df_data.drop(columns=["arrival_transport"]), one_hot], axis=1)

    # 3) Build a feature frame for the dashboard patients
    #    Keep ONLY patients present in the dashboard; preserve dashboard order via merge.
    df_features = df_dash[["patientId"]].merge(
        df_data[["patientId"] + list(set(REQUIRED_FEATURES) & set(df_data.columns))],
        on="patientId",
        how="left",
        sort=False,
        validate="m:1"
    )

    # 4) Ensure all required columns exist and are well-typed
    df_features = ensure_required_columns(df_features, ref_df_for_medians=df_data)

    # 5) Load model and align feature order
    print(f"Loading model: {MODEL_PATH}")
    model = joblib_load(MODEL_PATH)

    X = align_feature_order(df_features.drop(columns=["patientId"]), model)

    # 6) Predict sepsis probability → scale to 0–100 for sepsisScore
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        # For linear models without predict_proba (unlikely here), use decision_function as proxy
        # and pass through a logistic transform.
        scores = model.decision_function(X)
        proba = 1.0 / (1.0 + np.exp(-scores))

    sepsis_score = np.round(100.0 * proba, 0).astype(int)

    # 7) Build output by updating the original dashboard records
    #    (add REQUIRED_FEATURES + new sepsisScore; keep existing hazardRate unchanged)
    df_out = df_dash.merge(df_features, on="patientId", how="left", suffixes=("", "_feat"))
    df_out["sepsisScore"] = sepsis_score

    # 8) Serialize to JSON (list of dicts)
    records = df_out.to_dict(orient="records")
    with OUTPUT_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Wrote updated dashboard JSON → {OUTPUT_JSON_PATH.resolve()}")


if __name__ == "__main__":
    main()
