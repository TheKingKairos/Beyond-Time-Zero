#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lab_adder.py

Merge wide lab data into event_level.csv and recompute the time-varying sepsis label:
    flip_time = min{ first_abx_time, first time with >=2 SIRS criteria },
    applied only if sepsis_dx_any == 1 (ICD-confirmed stays), else label stays 0.

Assumptions
-----------
- You have already created:
    data/MIMIC-ED/stay_level.csv  (includes sepsis_dx_any, first_abx_time)
    data/MIMIC-ED/event_level.csv (already one-hot encoded, numeric vitals retained)
- Labs are a WIDE CSV with:
    subject_id, hadm_id (optional), stay_id, storetime, <many lab columns>
  Example columns include: pCO2, WBC (or synonyms), etc.

CLI
---
python lab_adder.py \
  --labs_path data/MIMIC-ED/ed/labs.csv \
  --event_path data/MIMIC-ED/event_level.csv \
  --stay_path  data/MIMIC-ED/stay_level.csv \
  [--out_event data/MIMIC-ED/event_level.csv] \
  [--out_stay  data/MIMIC-ED/stay_level.csv]

Memory/Runtimes
---------------
- Works in-place on CSVs; avoids re-creating the whole pipeline.
- Only keeps lab columns needed for SIRS by default (PaCO2, WBC, immature % if present).
- Optionally keep all lab columns by setting KEEP_ALL_LABS = True below.
"""

import argparse
import pandas as pd
import numpy as np
import re
from typing import Optional, Dict, List

# ------------------------------
# Tunables
# ------------------------------
LOOKBACK_VITALS_HOURS = 6
LOOKBACK_LABS_HOURS   = 12
KEEP_ALL_LABS         = True # set True to append all numeric lab columns to event_level

# Column name hints (lowercased, punctuation-insensitive matching)
VITAL_CANDIDATES = {
    "temp":  [r"\btemp(?:erature)?(?:_c|_cel(?:sius)?)?\b", r"\btemperature\b", r"\btemp_c\b", r"\btemperature_c\b"],
    "hr":    [r"\bhr\b", r"\bheart[_\s]?rate\b", r"\bheartrate\b"],
    "rr":    [r"\brr\b", r"\bresp(?:iratory)?[_\s]?rate\b", r"\bresprate\b"]
}
LAB_CANDIDATES = {
    "pco2":  [r"\bpco2\b", r"\bpa?co2\b"],
    "wbc":   [r"\bwbc\b", r"white[_\s-]*blood[_\s-]*cell", r"\bwhite[_\s-]*blood\b"],
    # immature forms % (bands/immature granulocytes)
    "immature_pct": [r"\bband(?:s)?\s*%\b", r"\bimmature\b.*\b%\b", r"\big%?\b", r"\bimmature[_\s-]*granulocyte[s]?\s*%"]
}

ID_COLS = ["subject_id", "hadm_id", "stay_id"]
TIME_COLS = ["charttime", "storetime"]


# ------------------------------
# Helpers
# ------------------------------
def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df

def find_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    cols = list(df.columns)
    for pat in patterns:
        rx = re.compile(pat, flags=re.IGNORECASE)
        for c in cols:
            if rx.search(c):
                return c
    return None

def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def add_if_missing(df: pd.DataFrame, col: str, value=0):
    if col not in df.columns:
        df[col] = value

def guess_and_convert_temp_c(temp_series: pd.Series) -> pd.Series:
    """Heuristic: if many values > 45, assume Fahrenheit and convert to Celsius."""
    s = coerce_numeric(temp_series)
    if s.notna().sum() == 0:
        return s
    # proportion above 45
    prop_high = (s > 45).mean()
    if prop_high > 0.2:
        # likely Fahrenheit
        return (s - 32) * 5.0 / 9.0
    return s

def last_valid_within(df: pd.DataFrame, value_col: str, time_col: str, hours: float, group_keys: List[str]) -> pd.Series:
    """
    Forward-fill value and then null it out if the last observation is older than window.
    Returns a series of 'valid' (time-limited LOCF) values.
    """
    # time at which the value was observed
    seen_time = df[time_col].where(df[value_col].notna())
    last_seen = seen_time.groupby(group_keys).ffill()
    valid_age = (df[time_col] - last_seen).dt.total_seconds() / 3600.0
    valid_val = df[value_col].groupby(group_keys).ffill()
    return valid_val.where(valid_age <= hours)

def first_time_where(df: pd.DataFrame, cond: pd.Series, keys: List[str], time_col: str) -> pd.DataFrame:
    ix = cond.fillna(False)
    sub = df.loc[ix, keys + [time_col]]
    if sub.empty:
        # return empty frame with proper dtypes
        out = pd.DataFrame(columns=keys + [time_col])
        for k in keys:
            out[k] = out[k].astype(df[k].dtype if k in df.columns else "int64")
        out[time_col] = pd.to_datetime(out[time_col])
        return out
    grp = sub.groupby(keys, as_index=False)[time_col].min()
    return grp


# ------------------------------
# Core
# ------------------------------
def main(args):
    # Load stay and event
    stay = pd.read_csv(args.stay_path, parse_dates=["intime","outtime","first_abx_time"], infer_datetime_format=True)
    stay = clean_cols(stay)
    # event_level is already big; parse times carefully
    event = pd.read_csv(args.event_path, parse_dates=["charttime","first_abx_time","intime","outtime"], infer_datetime_format=True, low_memory=False)
    event = clean_cols(event)

    # Ensure essential columns exist
    add_if_missing(event, "event_type_lab", 0)  # so we can safely append lab rows with 1s

    # Load lab wide CSV
    labs = pd.read_csv(args.labs_path, parse_dates=["storetime"], infer_datetime_format=True, low_memory=False)
    labs = clean_cols(labs)

    # ID/time housekeeping
    # Prefer 'storetime' for labs; unify to 'charttime' for concatenation
    if "storetime" not in labs.columns:
        raise ValueError("Labs file must include a 'storetime' column (timestamp of lab availability).")
    labs = labs.rename(columns={"storetime": "charttime"})

    # Keep only relevant ID/time + SIRS-needed columns (unless KEEP_ALL_LABS)
    keep_cols = [c for c in ID_COLS if c in labs.columns] + ["charttime"]

    # Detect SIRS columns in vitals (from event) and labs (from labs)
    # ---- From event (vitals) ----
    temp_col = find_col(event, VITAL_CANDIDATES["temp"])
    hr_col   = find_col(event, VITAL_CANDIDATES["hr"])
    rr_col   = find_col(event, VITAL_CANDIDATES["rr"])

    # ---- From labs ----
    pco2_col = find_col(labs, LAB_CANDIDATES["pco2"])
    wbc_col  = find_col(labs, LAB_CANDIDATES["wbc"])
    imm_col  = find_col(labs, LAB_CANDIDATES["immature_pct"])

    # Narrow lab columns (for features)
    lab_feature_cols = []
    if pco2_col: lab_feature_cols.append(pco2_col)
    if wbc_col:  lab_feature_cols.append(wbc_col)
    if imm_col:  lab_feature_cols.append(imm_col)

    if KEEP_ALL_LABS:
        # keep all numeric lab columns as features (may be large!)
        numeric_labs = labs.select_dtypes(include=[np.number]).columns.tolist()
        lab_feature_cols = sorted(set(lab_feature_cols + numeric_labs))

    # Build a compact Lab event frame for appending to event_level
    lab_events_cols = [c for c in keep_cols + lab_feature_cols if c in labs.columns]
    lab_events = labs[lab_events_cols].copy()

    # Merge stay-level columns so lab rows carry demographics/labels
    # (This mirrors what you did when building event_level initially)
    lab_events = lab_events.merge(
        stay[["subject_id","stay_id","first_abx_time","sepsis_dx_any"]],
        on=["subject_id","stay_id"],
        how="left"
    )

    # Tag as lab events + ensure dummy column exists
    lab_events["event_type_lab"] = 1

    # Align columns with event_level before concat
    for col in event.columns:
        if col not in lab_events.columns:
            # Fill missing columns: default 0 for dummy flags, else NaN
            if col.startswith("event_type_"):
                lab_events[col] = 0
            else:
                lab_events[col] = np.nan

    # Conversely, if new lab columns don't exist in event, add them (filled NaN/0)
    for col in lab_events.columns:
        if col not in event.columns:
            # add to event so concat aligns
            if col.startswith("event_type_"):
                event[col] = 0
            else:
                event[col] = np.nan

    # Coerce SIRS lab features to numeric
    for c in [pco2_col, wbc_col, imm_col]:
        if c and c in lab_events.columns:
            lab_events[c] = coerce_numeric(lab_events[c])

    # Append labs to event_level
    # Keep column order stable: use event.columns order
    lab_events = lab_events[event.columns]
    combined_event = pd.concat([event, lab_events], ignore_index=True)

    # --------------------------
    # Compute SIRS timeseries
    # --------------------------
    # Build a compact frame of only what we need for SIRS detection
    sirs_cols = ["subject_id","stay_id","charttime"]
    if temp_col: sirs_cols.append(temp_col)
    if hr_col:   sirs_cols.append(hr_col)
    if rr_col:   sirs_cols.append(rr_col)

    sirs_from_event = event[[c for c in sirs_cols if c in event.columns]].copy()

    sirs_from_labs = labs[[c for c in (["subject_id","stay_id","charttime", pco2_col, wbc_col, imm_col]) if c and c in labs.columns]].copy()

    # Outer combine by timepoints
    sirs = pd.concat([sirs_from_event, sirs_from_labs], ignore_index=True, sort=False)
    sirs = sirs.dropna(subset=["charttime"])
    sirs = sirs.sort_values(["subject_id","stay_id","charttime"]).reset_index(drop=True)

    # Coerce numerics for vitals too
    if temp_col and temp_col in sirs.columns:
        sirs[temp_col] = guess_and_convert_temp_c(sirs[temp_col])
    if hr_col and hr_col in sirs.columns:
        sirs[hr_col] = coerce_numeric(sirs[hr_col])
    if rr_col and rr_col in sirs.columns:
        sirs[rr_col] = coerce_numeric(sirs[rr_col])

    # LOCF within lookback windows
    keys = ["subject_id","stay_id"]
    # Valid temp/hr/rr within vitals window
    temp_valid = last_valid_within(sirs, temp_col, "charttime", LOOKBACK_VITALS_HOURS, keys) if temp_col else pd.Series([np.nan]*len(sirs))
    hr_valid   = last_valid_within(sirs, hr_col,   "charttime", LOOKBACK_VITALS_HOURS, keys) if hr_col   else pd.Series([np.nan]*len(sirs))
    rr_valid   = last_valid_within(sirs, rr_col,   "charttime", LOOKBACK_VITALS_HOURS, keys) if rr_col   else pd.Series([np.nan]*len(sirs))
    # Valid PaCO2/WBC within labs window
    pco2_valid = last_valid_within(sirs, pco2_col, "charttime", LOOKBACK_LABS_HOURS,   keys) if pco2_col else pd.Series([np.nan]*len(sirs))
    wbc_valid  = last_valid_within(sirs, wbc_col,  "charttime", LOOKBACK_LABS_HOURS,   keys) if wbc_col  else pd.Series([np.nan]*len(sirs))
    imm_valid  = last_valid_within(sirs, imm_col,  "charttime", LOOKBACK_LABS_HOURS,   keys) if imm_col  else pd.Series([np.nan]*len(sirs))

    # Criteria flags
    temp_flag = (temp_valid > 38.0) | (temp_valid < 36.0)
    hr_flag   = (hr_valid > 90.0)
    resp_flag = ( (rr_valid > 20.0) | (pco2_valid < 32.0) )
    wbc_flag  = ( (wbc_valid > 12000.0) | (wbc_valid < 4000.0) | (imm_valid > 10.0) )

    # SIRS count
    sirs_count = temp_flag.astype("int8") + hr_flag.astype("int8") + resp_flag.astype("int8") + wbc_flag.astype("int8")
    sirs["sirs_count"] = sirs_count

    # First time SIRS >= 2
    sirs2 = first_time_where(sirs, sirs["sirs_count"] >= 2, keys, "charttime")
    sirs2 = sirs2.rename(columns={"charttime": "sirs2_time"})

    # Merge SIRS times into stay; compute sepsis flip time
    stay2 = stay.merge(sirs2, on=["subject_id","stay_id"], how="left")

    # flip_time = min(first_abx_time, sirs2_time) if sepsis_dx_any == 1; else NaT
    flip_time = stay2[["first_abx_time","sirs2_time"]].min(axis=1)
    flip_time = np.where(stay2.get("sepsis_dx_any", 0) == 1, flip_time, pd.NaT)
    flip_time = pd.to_datetime(flip_time)

    stay2["sepsis_flip_time"] = flip_time

    # --------------------------
    # Update event-level sepsis label
    # --------------------------
    # Attach sepsis_flip_time to combined_event
    combined_event = combined_event.merge(
        stay2[["subject_id","stay_id","sepsis_flip_time","sepsis_dx_any"]],
        on=["subject_id","stay_id"],
        how="left"
    )

    # Recompute sepsis_dx (int8)
    combined_event["sepsis_dx"] = np.where(
        (combined_event["sepsis_dx_any"] == 1)
        & combined_event["sepsis_flip_time"].notna()
        & (combined_event["charttime"] >= combined_event["sepsis_flip_time"]),
        1, 0
    ).astype("int8")

    # (Optional) keep sirs_count on event rows at exact SIRS rows (join-on-time); otherwise omit to save space.
    # Here we omit to keep file small. Uncomment below to keep it (sparse join).
    # combined_event = combined_event.merge(
    #     sirs[["subject_id","stay_id","charttime","sirs_count"]],
    #     on=["subject_id","stay_id","charttime"], how="left"
    # )

    # --------------------------
    # Save outputs
    # --------------------------
    # Sort for readability
    sort_cols = [c for c in ["subject_id","stay_id","charttime"] if c in combined_event.columns]
    combined_event = combined_event.sort_values(sort_cols, na_position="last")

    # Ensure compact dtypes
    for col in ["sepsis_dx","sepsis_dx_any","event_type_lab"]:
        if col in combined_event.columns:
            combined_event[col] = combined_event[col].astype("int8")

    # Write
    combined_event.to_csv(args.out_event_path, index=False)
    stay2.to_csv(args.out_stay_path, index=False)

    print("✅ Updated event_level saved to:", args.out_event_path)
    print("✅ Updated stay_level (with sirs2_time & sepsis_flip_time) saved to:", args.out_stay_path)

    # Brief summary
    n_lab_rows = len(lab_events)
    n_stays_with_sirs2 = stay2["sirs2_time"].notna().sum()
    print(f"Info: appended {n_lab_rows:,} lab event rows.")
    print(f"Info: stays with SIRS>=2 time detected: {n_stays_with_sirs2:,}.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Merge lab data and recompute time-varying sepsis label.")
    p.add_argument("--labs_path",  required=True, help="Path to wide labs CSV (with storetime).")
    p.add_argument("--event_path", required=True, help="Path to existing event_level.csv.")
    p.add_argument("--stay_path",  required=True, help="Path to existing stay_level.csv.")
    p.add_argument("--out_event_path", default=None, help="Output path for updated event_level.csv (default: overwrite input).")
    p.add_argument("--out_stay_path",  default=None, help="Output path for updated stay_level.csv (default: overwrite input).")
    args = p.parse_args()

    if args.out_event_path is None:
        args.out_event_path = args.event_path
    if args.out_stay_path is None:
        args.out_stay_path = args.stay_path

    main(args)

