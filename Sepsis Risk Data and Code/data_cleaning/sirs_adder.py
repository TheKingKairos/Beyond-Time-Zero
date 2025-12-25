import pandas as pd
import numpy as np
from typing import Tuple

def _coerce_ids(df: pd.DataFrame, cols=("subject_id", "stay_id")) -> pd.DataFrame:
    # Use pandas nullable Int64 to preserve NAs that may exist in WBC file
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df

def _compute_sirs_flags(df: pd.DataFrame,
                        wbc_col: str = "wbc") -> Tuple[pd.Series, pd.Series]:
    """Return (sirs_count, sirs_ge2) as integer Series."""
    # SIRS components available in your vitals
    s_temp = (df["temperature"] > 100.4) | (df["temperature"] < 96.8)
    s_hr   = df["heartrate"] > 90
    s_rr   = df["resprate"]  > 20   # PaCO2 not available here

    # WBC (NaN -> False; thresholds are 12 and 4 in your units)
    s_wbc  = (df[wbc_col] > 12) | (df[wbc_col] < 4)
    s_wbc  = s_wbc.fillna(False)

    sirs_count = (
        s_temp.astype(int) +
        s_hr.astype(int)   +
        s_rr.astype(int)   +
        s_wbc.astype(int)
    )
    sirs_ge2 = (sirs_count >= 2).astype(int)
    return sirs_count, sirs_ge2

def add_sirs_and_redefine_onset(
    vitals_df: pd.DataFrame,
    wbc_csv_path: str = "data/MIMIC-ED/ed/cleaned_labevents_wbc.csv",
    wbc_backward_hours: int = 24,
    treat_wbc_zero_as_missing: bool = True,
) -> pd.DataFrame:
    """
    Upgrade your dataset so sepsis onset per (subject_id, stay_id) is:
        min{ earliest time with SIRS >= 2, earliest time with antibiotics }
    - Keeps row count identical to vitals_df.
    - Does NOT keep WBC as a column in the final output.
    - WBCs are merged by as-of join on (subject_id, stay_id, charttime), backward within a tolerance window.

    New columns:
      - sirs_count (int)
      - sirs_ge2 (0/1)
      - sepsis_onset_time (timestamp, per-stay constant on/after onset rows; NaT if never triggered)
      - is_sepsis_onset (0/1)  # exactly the first row at onset time
      - is_sepsis (0/1)        # 1 for all rows at/after onset within the stay
    """
    df = vitals_df.copy()

    # Ensure datetimes & IDs are consistent
    if "charttime" not in df.columns:
        raise ValueError("vitals_df must contain a 'charttime' column.")
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    _coerce_ids(df)

    # Load WBC file; rename storetime->charttime; keep only needed columns
    wbc = pd.read_csv(wbc_csv_path, parse_dates=["storetime"])
    wbc = wbc.rename(columns={"storetime": "charttime", "White Blood Cells": "wbc"})
    wbc = wbc[["subject_id", "stay_id", "charttime", "wbc"]]
    _coerce_ids(wbc)

    # Sort for merge_asof: must be sorted by 'by' keys and 'on'
    df = df.sort_values(["charttime"])
    wbc = wbc.sort_values(["charttime"])

    # As-of merge (backward) so we DO NOT add any rows. Row count is preserved.
    tol = pd.Timedelta(hours=wbc_backward_hours)
    merged = pd.merge_asof(
        df,
        wbc,
        by=["subject_id", "stay_id"],
        on="charttime",
        direction="backward",
        tolerance=tol,
    )

    # Optional: treat WBC==0.0 as missing (often an artifact)
    if treat_wbc_zero_as_missing and "wbc" in merged.columns:
        merged.loc[merged["wbc"] == 0, "wbc"] = np.nan

    # Compute SIRS per row (WBC included but will be dropped after)
    sirs_count, sirs_ge2 = _compute_sirs_flags(merged, wbc_col="wbc")
    merged["sirs_count"] = sirs_count
    merged["sirs_ge2"]   = sirs_ge2

    # Masks
    abx_mask = (merged["is_antibiotic"] > 0) if "is_antibiotic" in merged.columns else pd.Series(False, index=merged.index)

    # ---- Vectorized earliest times per stay (no joblib, no Python loops) ----
    keys = ["subject_id", "stay_id"]
    # Mask charttime for each condition; others -> NaT
    ct_abx  = merged["charttime"].where(abx_mask,  pd.NaT)
    ct_sirs = merged["charttime"].where(merged["sirs_ge2"].eq(1), pd.NaT)

    # Per-stay earliest times via transform('min')
    earliest_abx  = ct_abx.groupby(merged[keys].agg(tuple, axis=1)).transform("min")
    earliest_sirs = ct_sirs.groupby(merged[keys].agg(tuple, axis=1)).transform("min")

    # Onset is rowwise min of the two datetimes
    merged["sepsis_onset_time"] = earliest_abx.where(earliest_abx.notna(), earliest_sirs)
    need_min = earliest_abx.notna() & earliest_sirs.notna()
    merged.loc[need_min, "sepsis_onset_time"] = np.minimum(
        earliest_abx[need_min].values, earliest_sirs[need_min].values
    )

    # Mark the single row that hits the per-stay onset time
    merged["is_sepsis_onset"] = (merged["charttime"] == merged["sepsis_onset_time"]).astype(int)

    # --- Ensure all labels at/after onset are 1 within each stay ---
    # If no onset in a stay (NaT), label remains 0 for all rows in that stay.
    merged["is_sepsis"] = (merged["charttime"] >= merged["sepsis_onset_time"]).astype(int).fillna(0).astype(int)

    # Drop WBC column to honor "wbc isn't an entry in our columns"
    if "wbc" in merged.columns:
        merged = merged.drop(columns=["wbc"])

    # Keep chronological order
    merged = merged.sort_values(["subject_id", "stay_id", "charttime"]).reset_index(drop=True)
    return merged


# --- Example usage ---
if __name__ == "__main__":
    df = pd.read_csv("data/MIMIC-ED/event_level_cleaned_v2.csv", parse_dates=["charttime"])
    upgraded = add_sirs_and_redefine_onset(
        df,
        wbc_csv_path="data/MIMIC-ED/ed/cleaned_labevents_wbc.csv",
        wbc_backward_hours=24,
        treat_wbc_zero_as_missing=True,
    )
    # # reorder columns
    # cols = upgraded.columns.tolist()
    # labels = ["sepsis_dx_any", "sepsis_dx", "sirs_count", "sirs_ge2", "sepsis_onset_time", "is_sepsis_onset", "is_sepsis"]
    # non_label_cols = [c for c in cols if c not in labels]
    # upgraded = upgraded[non_label_cols + labels]
    upgraded.to_csv("data/MIMIC-ED/event_level_cleaned_sirs_v2.csv", index=False)
