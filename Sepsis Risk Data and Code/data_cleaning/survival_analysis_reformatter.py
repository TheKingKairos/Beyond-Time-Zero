import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
INPUT_PATH  = "data/MIMIC-ED/event_level_cleaned_sirs_v2.csv"
OUTPUT_PATH_STATIC = "data/MIMIC-ED/cox_static_random_v2.csv"
OUTPUT_PATH_TVC    = "data/MIMIC-ED/cox_timevarying_v2.csv"
OUTPUT_PATH_STATIC_STACKED = "data/MIMIC-ED/cox_static_landmark_stacked_v2.csv"
RNG_SEED = 42

ID_COLS   = ["subject_id", "hadm_id", "stay_id"]
TIME_COLS = ["intime", "outtime", "charttime"]
LABEL_COLS = ["sepsis_dx", "sepsis_dx_any", "is_sepsis"]  # exclude from features

DURATION_COL = "duration"
EVENT_COL    = "event"

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

def _first_event_time(g: pd.DataFrame) -> pd.Timestamp | None:
    """First time is_sepsis flips to 1 within a stay."""
    idx = g.index[g["is_sepsis"] == 1]
    if len(idx) == 0:
        return None
    return g.loc[idx[0], "charttime"]

def _choose_random_baseline(g: pd.DataFrame, event_time: pd.Timestamp | None, rng: np.random.Generator):
    """
    For event stays: choose a random row with charttime < event_time.
    For censored stays: choose a random row among all rows.
    Returns the chosen row (Series) or None if not possible.
    """
    if event_time is not None:
        candidates = g[g["charttime"] < event_time]
        if candidates.empty:
            return None
        sel = candidates.iloc[rng.integers(0, len(candidates))]
        return sel
    else:
        sel = g.iloc[rng.integers(0, len(g))]
        return sel

def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric covariates only, excluding IDs/time/labels."""
    drop = set([c for c in ID_COLS if c in df.columns] +
               [c for c in TIME_COLS if c in df.columns] +
               [c for c in LABEL_COLS if c in df.columns])
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in drop]

# ----------------------------
# Static Cox (one row per stay)
# ----------------------------
def make_static_random_snapshot(
    df: pd.DataFrame,
    *,
    id_col: str = "stay_id",
    duration_col: str = DURATION_COL,
    event_col: str = EVENT_COL,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    """
    Build a leakage-safe CoxPH table:
      - one random pre-event snapshot for event stays
      - one random snapshot for censored stays
      - duration = time from snapshot to event or censoring
      - event = 1 if event observed, else 0
    """
    assert id_col in df.columns, f"Missing id column: {id_col}"
    # Ensure datetimes
    for c in ["charttime", "intime", "outtime"]:
        if c in df.columns:
            df[c] = _to_dt(df[c])

    # Sort by time within stay
    if "charttime" in df.columns:
        df = df.sort_values([id_col, "charttime"], kind="mergesort")
    else:
        raise ValueError("charttime is required in the input for temporal ordering.")

    rng = np.random.default_rng(seed)
    feats = _feature_columns(df)

    rows = []
    dropped_no_pre_event = 0
    dropped_nonpos_duration = 0

    for sid, g in df.groupby(id_col, sort=False):
        g = g.reset_index(drop=True)
        event_time = _first_event_time(g)
        event_flag = int(event_time is not None)

        # Choose baseline
        sel = _choose_random_baseline(g, event_time, rng)
        if sel is None:
            dropped_no_pre_event += 1
            continue

        t0 = sel["charttime"]
        # Define stop time
        if event_flag == 1:
            t1 = event_time
        else:
            # censor at last observation time (or outtime if present and earlier/later—use last observation)
            t1 = g["charttime"].iloc[-1]

        dur = hours_between(t0, t1)
        if dur <= 0:
            dropped_nonpos_duration += 1
            continue

        row = {
            duration_col: dur,
            event_col: event_flag,
        }
        # Add covariates from the selected snapshot (numeric only)
        for c in feats:
            row[c] = sel[c]

        rows.append(row)

    out = pd.DataFrame(rows)
    # Reorder: duration, event, covariates
    covs = [c for c in out.columns if c not in [duration_col, event_col]]
    out = out[[duration_col, event_col] + covs]

    # ---------------- Validations ----------------
    bad_dur = (out[duration_col] <= 0).sum()
    uniq_events = set(out[event_col].unique().tolist())
    if bad_dur:
        print(f"[WARN] {bad_dur} rows have non-positive duration after filtering.")
    if not uniq_events.issubset({0, 1}):
        raise ValueError(f"[ERROR] event column has non-binary values: {uniq_events}")

    # Types
    out[duration_col] = out[duration_col].astype(float)
    out[event_col] = out[event_col].astype(int)

    print(f"[OK] Built static Cox dataset: n={len(out)} "
          f"(dropped_no_pre_event={dropped_no_pre_event}, dropped_nonpos_duration={dropped_nonpos_duration})")
    return out

def make_static_landmark_stack(
    df: pd.DataFrame,
    *,
    id_col: str = "stay_id",
    duration_col: str = DURATION_COL,
    event_col: str = EVENT_COL,
) -> pd.DataFrame:
    """
    Survival-stacked (landmark) dataset for CoxPH / regression-style models.

    For each stay:
      - If an event occurs, create ONE ROW for EVERY row with charttime < event_time.
      - If censored, create ONE ROW for EVERY row with charttime < last_observed_time.
      - Each row's covariates come from that landmark row only (past/present info).
      - duration = hours from landmark to event/censor; event=1 if event observed.

    This prevents leakage (no future covariates) while greatly increasing sample size.
    """
    assert id_col in df.columns, f"Missing id column: {id_col}"
    if "charttime" not in df.columns:
        raise ValueError("charttime is required in the input for temporal ordering.")

    # Ensure datetimes & sort
    for c in ["charttime", "intime", "outtime"]:
        if c in df.columns:
            df[c] = _to_dt(df[c])
    df = df.dropna(subset=["charttime"])
    df = df.sort_values([id_col, "charttime"], kind="mergesort")

    feats = _feature_columns(df)

    rows = []
    n_dropped_nonpos = 0
    n_no_pre_event = 0
    n_censored_last_dropped = 0

    for sid, g in df.groupby(id_col, sort=False):
        g = g.reset_index(drop=True)

        # first event time (where is_sepsis == 1)
        idx = g.index[g["is_sepsis"] == 1]
        event_time = g.loc[idx[0], "charttime"] if len(idx) else None
        event_flag = int(event_time is not None)

        # define the terminal time to count toward: event time if observed else last observed charttime
        if event_flag == 1:
            terminal_time = event_time
            # eligible landmarks: strictly before event
            candidates = g[g["charttime"] < event_time]
            if candidates.empty:
                # No pre-event rows to landmark on (e.g., first row already event)
                n_no_pre_event += 1
                continue
        else:
            terminal_time = g["charttime"].iloc[-1]
            # eligible landmarks: strictly before last observation
            # (the last row would give duration 0 → drop)
            candidates = g[g["charttime"] < terminal_time]
            if len(g) >= 1 and (g["charttime"].iloc[-1] == terminal_time):
                n_censored_last_dropped += 1

        # build rows
        for _, sel in candidates.iterrows():
            t0 = sel["charttime"]
            dur = hours_between(t0, terminal_time)
            if not np.isfinite(dur) or dur <= 0:
                n_dropped_nonpos += 1
                continue

            row = {
                duration_col: float(dur),
                event_col: event_flag,
            }
            for c in feats:
                row[c] = sel[c]
            rows.append(row)

    out = pd.DataFrame(rows)

    # reorder & type checks
    if not out.empty:
        covs = [c for c in out.columns if c not in [duration_col, event_col]]
        out = out[[duration_col, event_col] + covs]
        out[duration_col] = out[duration_col].astype(float)
        out[event_col] = out[event_col].astype(int)

        if not set(out[event_col].unique()).issubset({0, 1}):
            raise ValueError("event must be binary (0/1).")

    print(
        "[OK] Built survival-stacked (landmark) Cox dataset: "
        f"n={len(out)} | dropped_nonpos={n_dropped_nonpos} | "
        f"no_pre_event={n_no_pre_event} | cens_last_dropped={n_censored_last_dropped}"
    )
    return out


# ----------------------------
# OPTIONAL: Time-varying Cox (start/stop per stay)
# ----------------------------
def make_time_varying_long(
    df: pd.DataFrame,
    *,
    id_col: str = "stay_id",
    start_col: str = "start",
    stop_col: str = "stop",
    event_col: str = "event",
    epsilon_seconds: int = 1,  # tiny positive interval when event == last charttime
) -> pd.DataFrame:
    """
    Andersen–Gill long data for lifelines.CoxTimeVaryingFitter.

    - Per-stay, timestamps are sorted and deduplicated (keep='last').
    - Intervals are [start, stop) in HOURS since the *first* observed time in that stay.
    - Covariates are taken from the 'start' row (piecewise-constant).
    - Event is assigned with right-closed rule: start <= event_time < stop.
    - If event_time equals the final timestamp, create a tiny terminal interval
      [last_time, last_time + epsilon] with event=1 (or use outtime if later).

    Returns columns: [id_col, start_col, stop_col, event_col] + covariates.
    """
    required = ["charttime"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    # ensure datetime and sort
    df = df.copy()
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df = df.dropna(subset=["charttime"])
    df = df.sort_values([id_col, "charttime"], kind="mergesort")

    # feature set excludes IDs/time/labels
    feats = _feature_columns(df)

    rows = []
    n_skipped_short = 0
    n_eps_used = 0

    # optional outtime presence
    has_outtime = "outtime" in df.columns

    for sid, g in df.groupby(id_col, sort=False):
        g = g.copy()

        # Deduplicate identical timestamps within a stay (keep last obs at that time)
        g = g.drop_duplicates(subset=["charttime"], keep="last").reset_index(drop=True)

        # Need at least 1 row to anchor; to form an interval we need >=2 OR an event at last with epsilon/outtime
        if len(g) == 0:
            continue

        base_t = g.loc[0, "charttime"]
        times = g["charttime"].tolist()

        # First event time (where is_sepsis == 1)
        if "is_sepsis" not in g.columns:
            raise ValueError("Column 'is_sepsis' is required to locate event times.")
        idx_ev = g.index[g["is_sepsis"] == 1]
        event_time = g.loc[idx_ev[0], "charttime"] if len(idx_ev) else None

        # Build standard intervals between consecutive unique times
        for i in range(len(times) - 1):
            t_start = times[i]
            t_stop  = times[i + 1]

            start_hr = hours_between(base_t, t_start)
            stop_hr  = hours_between(base_t, t_stop)
            if stop_hr <= start_hr:
                # Shouldn't happen after dedup/sort, but guard anyway
                continue

            # Right-closed rule: fire if start <= event_time < stop
            ev = 0
            if event_time is not None and (t_start <= event_time < t_stop):
                ev = 1

            row = {
                id_col: sid,
                start_col: start_hr,
                stop_col:  stop_hr,
                event_col: ev,
            }
            # covariates from the 'start' row i
            for c in feats:
                row[c] = g.loc[i, c]
            rows.append(row)

            if ev == 1:
                # stop after first event
                break

        else:
            # no event inside the built intervals
            if event_time is not None:
                # If event_time is >= last observed time, attach a terminal interval
                last_time = times[-1]
                if event_time >= last_time:
                    # Prefer outtime if present and later; else epsilon past last_time
                    if has_outtime:
                        ot = pd.to_datetime(g["outtime"].iloc[0], errors="coerce")
                    else:
                        ot = pd.NaT
                    if pd.notna(ot) and ot > last_time:
                        t_stop = ot
                    else:
                        t_stop = last_time + pd.Timedelta(seconds=epsilon_seconds)
                        n_eps_used += 1

                    start_hr = hours_between(base_t, last_time)
                    stop_hr  = hours_between(base_t, t_stop)
                    if stop_hr > start_hr:
                        row = {
                            id_col: sid,
                            start_col: start_hr,
                            stop_col:  stop_hr,
                            event_col: 1 if (last_time <= event_time < t_stop) else 0,
                        }
                        for c in feats:
                            row[c] = g.iloc[-1][c]
                        rows.append(row)
                    else:
                        n_skipped_short += 1
            else:
                # censored stay: nothing more to add (already added all non-event intervals)
                pass

    out = pd.DataFrame(rows)

    # Final validation: drop any residual non-positive lengths (defensive)
    if not out.empty:
        bad = out[out[stop_col] <= out[start_col]]
        if len(bad):
            print(f"[WARN] Dropping {len(bad)} non-positive intervals after construction.")
            out = out[out[stop_col] > out[start_col]]

        # Binary check
        if not set(out[event_col].unique()).issubset({0, 1}):
            raise ValueError("event must be binary in time-varying data.")

    print(f"[OK] Time-varying dataset: rows={len(out)} | used_epsilon={n_eps_used} | skipped_short={n_skipped_short}")
    return out


# ----------------------------
# Main
# ----------------------------
def main():
    Path(OUTPUT_PATH_STATIC).parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)

    # Ensure required label/time columns exist
    need = {"stay_id", "charttime", "is_sepsis"}
    missing = need - set(df.columns)
    if missing:
        raise FileNotFoundError(f"Input missing required columns: {missing}")

    # Build STATIC Cox table (leakage-safe)
    cox_df = make_static_random_snapshot(df, id_col="stay_id")
    cox_df.to_csv(OUTPUT_PATH_STATIC, index=False)
    print(f"✅ Wrote {OUTPUT_PATH_STATIC} (cols: {cox_df.shape[1]}, rows: {cox_df.shape[0]})")

    # build TIME-VARYING table for CoxTimeVaryingFitter
    tv_df = make_time_varying_long(df, id_col="stay_id")
    tv_df.to_csv(OUTPUT_PATH_TVC, index=False)
    print(f"✅ Wrote {OUTPUT_PATH_TVC} (cols: {tv_df.shape[1]}, rows: {tv_df.shape[0]})")

    # Build SURVIVAL-STACKED (landmark) Cox table — leakage-safe, many rows per stay
    cox_stack = make_static_landmark_stack(df, id_col="stay_id")
    cox_stack.to_csv(OUTPUT_PATH_STATIC_STACKED, index=False)
    print(f"✅ Wrote {OUTPUT_PATH_STATIC_STACKED} (cols: {cox_stack.shape[1]}, rows: {cox_stack.shape[0]})")


if __name__ == "__main__":
    main()
