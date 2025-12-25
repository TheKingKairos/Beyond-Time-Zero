import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
INPUT_PATH  = "data/MIMIC-ED/event_level_cleaned_sirs.csv"
OUTPUT_PATH_DISCRETE_PP = "data/MIMIC-ED/discrete_time_pp_30min.csv"

RNG_SEED = 42

ID_COLS    = ["subject_id", "hadm_id", "stay_id"]
TIME_COLS  = ["intime", "outtime", "charttime"]
LABEL_COLS = ["sepsis_dx", "sepsis_dx_any", "is_sepsis"]  # exclude from features

# Discrete-time settings
BIN_MINUTES = 30           # 30-minute bins
MAX_HOURS   = 6            # set to None to use full remaining time; else cap horizon (e.g., 6 hours)

# Output schema core columns
T_BIN_COL   = "t_bin"      # 1,2,... per landmark
EVENT_COL   = "event"      # 0/1: event happens in this bin from the landmark
LANDMARK_TS = "landmark_time"
STAY_ID_COL = "stay_id"

# ----------------------------
# Helpers
# ----------------------------
def _to_dt(s):
    return pd.to_datetime(s, errors="coerce")

def hours_between(a, b):
    """Return positive hours (b - a)."""
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return max(0.0, (b - a).total_seconds() / 3600.0)

def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric covariates only, excluding IDs/time/labels."""
    drop = set([c for c in ID_COLS if c in df.columns] +
               [c for c in TIME_COLS if c in df.columns] +
               [c for c in LABEL_COLS if c in df.columns])
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in drop]

def _first_event_time(g: pd.DataFrame) -> pd.Timestamp | None:
    """First time is_sepsis flips to 1 within a stay."""
    if "is_sepsis" not in g.columns:
        raise ValueError("Input must contain 'is_sepsis' to locate event times.")
    idx = g.index[g["is_sepsis"] == 1]
    if len(idx) == 0:
        return None
    return g.loc[idx[0], "charttime"]

def _eligible_landmarks(g: pd.DataFrame, terminal_time: pd.Timestamp) -> pd.DataFrame:
    """
    Landmarks are rows strictly before terminal_time.
    (For censored stays, terminal_time is last observed charttime—so the last row is excluded.)
    """
    return g[g["charttime"] < terminal_time]

# ----------------------------
# Discrete-time person-period builder
# ----------------------------
def make_discrete_time_person_period(
    df: pd.DataFrame,
    *,
    id_col: str = STAY_ID_COL,
    bin_minutes: int = BIN_MINUTES,
    max_hours: float | None = MAX_HOURS,
    event_col: str = EVENT_COL,
    tbin_col: str = T_BIN_COL,
) -> pd.DataFrame:
    """
    Build a *snapshot-based* discrete-time person-period table with fixed-width bins.

    For each stay:
      - Determine event_time = first time is_sepsis==1 (if any).
      - terminal_time = event_time if observed, else last observed charttime.
      - Landmarks = all rows with charttime < terminal_time (pre-event or pre-last-observed).
      - For each landmark row:
          * Remaining time (rem_hr) = terminal_time - landmark_time (in hours)
          * If max_hours is not None, cap remaining to max_hours.
          * Expand into K = floor/ceil bins of width `bin_minutes`.
            - If event observed and falls within the capped horizon:
                mark the bin containing the event as event=1, others 0.
            - If censored or event beyond horizon: all event=0.
      - Covariates are taken *only* from the landmark row (no future information, no change-rates).

    Returns columns:
      [id_col, LANDMARK_TS, tbin_col, event_col] + covariates_from_landmark
    """
    if id_col not in df.columns:
        raise ValueError(f"Missing id column: {id_col}")
    if "charttime" not in df.columns:
        raise ValueError("Input must contain 'charttime' for temporal ordering.")
    if "is_sepsis" not in df.columns:
        raise ValueError("Input must contain 'is_sepsis' to determine events.")

    # Ensure datetimes & sort
    df = df.copy()
    for c in ["charttime", "intime", "outtime"]:
        if c in df.columns:
            df[c] = _to_dt(df[c])
    df = df.dropna(subset=["charttime"])
    df = df.sort_values([id_col, "charttime"], kind="mergesort")

    feats = _feature_columns(df)
    bin_hours = float(bin_minutes) / 60.0

    rows = []
    n_no_landmarks = 0
    n_skipped_nonpos = 0

    for sid, g in df.groupby(id_col, sort=False):
        g = g.reset_index(drop=True)

        # Determine event/terminal time
        event_time = _first_event_time(g)
        has_event = event_time is not None

        terminal_time = event_time if has_event else g["charttime"].iloc[-1]

        # Candidate landmarks: strictly before terminal_time
        candidates = _eligible_landmarks(g, terminal_time)
        if candidates.empty:
            n_no_landmarks += 1
            continue

        for _, sel in candidates.iterrows():
            landmark_time = sel["charttime"]
            rem_full = hours_between(landmark_time, terminal_time)
            if rem_full <= 0 or not np.isfinite(rem_full):
                n_skipped_nonpos += 1
                continue

            # Horizon capping
            rem_capped = rem_full if (max_hours is None) else min(rem_full, float(max_hours))

            # Number of bins & event bin calculation
            if has_event:
                # distance from landmark to actual event
                dist_to_event = hours_between(landmark_time, event_time)
                if dist_to_event > rem_capped + 1e-12:
                    # Event exists but beyond capped horizon → all zeros
                    last_bin = int(np.floor(rem_capped / bin_hours))
                    event_bin = None
                else:
                    # Event within capped window: the bin that contains the event
                    # Use ceil to mark the bin whose right-edge crosses the event time
                    event_bin = int(np.ceil(dist_to_event / bin_hours))
                    # If event occurs exactly at landmark_time (rare): place in bin 1
                    if event_bin <= 0:
                        event_bin = 1
                    # Last bin should include the event bin
                    last_bin = int(np.ceil(rem_capped / bin_hours))
            else:
                # censored: use floor (exclude a zero-length trailing bin)
                last_bin = int(np.floor(rem_capped / bin_hours))
                event_bin = None

            if last_bin <= 0:
                n_skipped_nonpos += 1
                continue

            # Emit person-period rows
            for k in range(1, last_bin + 1):
                row = {
                    id_col: sid,
                    LANDMARK_TS: landmark_time,
                    tbin_col: k,
                    event_col: 1 if (event_bin is not None and k == event_bin) else 0,
                }
                # Covariates from the landmark snapshot ONLY
                for c in feats:
                    row[c] = sel[c]
                rows.append(row)

    out = pd.DataFrame(rows)

    # Final ordering & type checks
    if not out.empty:
        covs = [c for c in out.columns if c not in [id_col, LANDMARK_TS, tbin_col, event_col]]
        out = out[[id_col, LANDMARK_TS, tbin_col, event_col] + covs]
        out[tbin_col] = out[tbin_col].astype(int)
        out[event_col] = out[event_col].astype(int)
        uniq_events = set(out[event_col].unique().tolist())
        if not uniq_events.issubset({0, 1}):
            raise ValueError(f"event must be binary, found: {uniq_events}")

    print(
        "[OK] Discrete-time person-period table built: "
        f"rows={len(out)} | no_landmarks={n_no_landmarks} | skipped_nonpos={n_skipped_nonpos} | "
        f"bin_minutes={bin_minutes} | max_hours={'None' if max_hours is None else max_hours}"
    )
    return out

def make_discrete_time_person_period_fast(
    df: pd.DataFrame,
    *,
    id_col: str = STAY_ID_COL,
    bin_minutes: int = BIN_MINUTES,
    max_hours: float | None = MAX_HOURS,
    event_col: str = EVENT_COL,
    tbin_col: str = T_BIN_COL,
) -> pd.DataFrame:
    """
    Vectorized discrete-time person-period constructor (fast).

    Output columns:
      [id_col, LANDMARK_TS, tbin_col, event_col] + landmark covariates
    """
    if id_col not in df.columns:
        raise ValueError(f"Missing id column: {id_col}")
    if "charttime" not in df.columns:
        raise ValueError("Input must contain 'charttime'.")
    if "is_sepsis" not in df.columns:
        raise ValueError("Input must contain 'is_sepsis'.")

    # --- prep & sort ---
    d = df.copy()
    for c in ["charttime", "intime", "outtime"]:
        if c in d.columns:
            d[c] = _to_dt(d[c])
    d = d.dropna(subset=["charttime"])
    d = d.sort_values([id_col, "charttime"], kind="mergesort")
    # de-dup exact timestamps within stay to avoid 0-length issues
    d = d.drop_duplicates(subset=[id_col, "charttime"], keep="last")

    feats = _feature_columns(d)
    bin_hours = float(bin_minutes) / 60.0

    # --- per-stay event/terminal times (vectorized) ---
    mask_ev = d["is_sepsis"] == 1
    first_evt = (
        d.loc[mask_ev, [id_col, "charttime"]]
         .groupby(id_col, sort=False)["charttime"].min()
    )
    last_obs = d.groupby(id_col, sort=False)["charttime"].max()

    meta = pd.DataFrame({id_col: last_obs.index}).set_index(id_col)
    meta["event_time"] = first_evt
    meta["has_event"]  = meta["event_time"].notna()
    # terminal = event_time if exists else last observation
    meta["terminal_time"] = meta["event_time"].where(meta["has_event"], last_obs)
    meta = meta.reset_index()

    # --- join back & choose landmarks (< terminal) ---
    d = d.merge(meta[[id_col, "event_time", "terminal_time", "has_event"]], on=id_col, how="left")
    landmarks = d[d["charttime"] < d["terminal_time"]].copy()
    if landmarks.empty:
        return pd.DataFrame(columns=[id_col, LANDMARK_TS, tbin_col, event_col] + feats)

    # --- remaining time (hours) & horizon cap ---
    rem_full = (landmarks["terminal_time"] - landmarks["charttime"]).dt.total_seconds() / 3600.0
    if max_hours is None:
        rem_cap = rem_full.to_numpy()
    else:
        rem_cap = np.minimum(rem_full.to_numpy(), float(max_hours))

    # --- bin counts & event-bin (vectorized) ---
    has_ev = landmarks["has_event"].to_numpy()
    # distance to event (NaT-safe)
    dist_to_ev = np.where(
        has_ev,
        (landmarks["event_time"] - landmarks["charttime"]).dt.total_seconds().to_numpy() / 3600.0,
        np.nan,
    )

    # last_bin: event → ceil(rem_cap/bin); censored → floor(rem_cap/bin)
    with np.errstate(divide="ignore", invalid="ignore"):
        lb_event   = np.ceil(rem_cap / bin_hours)
        lb_censor  = np.floor(rem_cap / bin_hours)
    last_bin = np.where(has_ev, lb_event, lb_censor).astype(int)
    valid = last_bin > 0

    landmarks = landmarks.loc[valid].copy()
    last_bin = last_bin[valid]
    rem_cap  = rem_cap[valid]
    has_ev   = has_ev[valid]
    dist_to_ev = dist_to_ev[valid]

    # event_bin: ceil(dist/bin) if event within window; else 0 (no event in window)
    with np.errstate(divide="ignore", invalid="ignore"):
        eb = np.ceil(dist_to_ev / bin_hours)
    # put events exactly at landmark into bin 1
    eb = np.where(has_ev & np.isfinite(eb) & (eb <= 0), 1, eb)
    # if event is outside capped window, turn off
    eb = np.where(has_ev & np.isfinite(dist_to_ev) & (dist_to_ev <= rem_cap + 1e-12), eb, 0)
    event_bin = np.nan_to_num(eb, nan=0.0).astype(int)

    # --- fast expansion via repeat-index ---
    rep_idx = landmarks.index.to_numpy().repeat(last_bin)
    out = landmarks.loc[rep_idx, [id_col] + feats].reset_index(drop=True)
    out[LANDMARK_TS] = landmarks.loc[rep_idx, "charttime"].reset_index(drop=True)

    # build t_bin with one concatenation
    t_bins = np.concatenate([np.arange(1, n + 1, dtype=int) for n in last_bin])
    out[tbin_col] = t_bins

    # broadcast event_bin per expanded row
    eb_rep = np.repeat(event_bin, last_bin)
    out[event_col] = (t_bins == eb_rep).astype(int)

    # final ordering/types
    cols = [id_col, LANDMARK_TS, tbin_col, event_col] + feats
    out = out[cols]
    out[tbin_col] = out[tbin_col].astype(int)
    out[event_col] = out[event_col].astype(int)

    print(
        "[OK-fast] Discrete-time PP: rows="
        f"{len(out)} | landmarks={len(landmarks)} | mean_bins={last_bin.mean():.2f} | "
        f"bin_minutes={bin_minutes} | max_hours={'None' if max_hours is None else max_hours}"
    )
    return out


# ----------------------------
# Main
# ----------------------------
def main():
    Path(OUTPUT_PATH_DISCRETE_PP).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)

    # Ensure required columns
    need = {STAY_ID_COL, "charttime", "is_sepsis"}
    missing = need - set(df.columns)
    if missing:
        raise FileNotFoundError(f"Input missing required columns: {missing}")

    # Build discrete-time person-period dataset (30-min bins, 6h horizon by default)
    # dpp = make_discrete_time_person_period(
    #     df,
    #     id_col=STAY_ID_COL,
    #     bin_minutes=BIN_MINUTES,
    #     max_hours=MAX_HOURS,     # set to None to use full remaining time
    # )
    dpp = make_discrete_time_person_period_fast(
    df,
    id_col=STAY_ID_COL,
    bin_minutes=BIN_MINUTES,
    max_hours=MAX_HOURS,
    )

    dpp.to_csv(OUTPUT_PATH_DISCRETE_PP, index=False)
    print(f"✅ Wrote {OUTPUT_PATH_DISCRETE_PP} (cols: {dpp.shape[1]}, rows: {dpp.shape[0]})")

if __name__ == "__main__":
    main()
