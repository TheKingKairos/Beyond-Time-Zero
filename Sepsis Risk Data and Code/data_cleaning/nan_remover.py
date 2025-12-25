# data_cleaner.py
from __future__ import annotations

import os
import glob
from typing import Iterable, Optional, Sequence, Union, List, Tuple, Dict
import pandas as pd
import numpy as np


# ---------- Utilities ----------

def _infer_sep_from_ext(path: str, default: str = ",") -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return ","
    if ext == ".psv":
        return "|"
    if ext in {".tsv", ".tab"}:
        return "\t"
    return default


def _ensure_list(x: Union[str, Sequence[str], None]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _numeric_imputable_columns(
    df: pd.DataFrame,
    exclude_cols: Iterable[str],
    groupby_cols: Iterable[str],
    time_col: Optional[str],
) -> List[str]:
    exclude = set(_ensure_list(exclude_cols)) | set(_ensure_list(groupby_cols))
    if time_col:
        exclude.add(time_col)
    # Only numeric columns get a median; non-numeric will still get ffill/bfill if included.
    # For safety, we impute only numeric by default.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric_cols if c not in exclude]


def _sorted_for_group_ops(
    df: pd.DataFrame, groupby_cols: Sequence[str], time_col: Optional[str]
) -> pd.DataFrame:
    if time_col and time_col in df.columns:
        sort_cols = list(_ensure_list(groupby_cols)) + [time_col]
        return df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    # Always sort by group to make ffill/bfill safe across contiguous groups
    return df.sort_values(list(_ensure_list(groupby_cols)), kind="mergesort").reset_index(drop=True)


# ---------- Core function ----------

def nan_remover(
    df: pd.DataFrame,
    *,
    groupby: Union[str, Sequence[str]],
    time_col: Optional[str] = None,
    exclude_cols: Optional[Iterable[str]] = None,
    global_medians: Optional[pd.Series] = None,
    only_numeric: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Fill NaNs for sequential/grouped data using:
      (1) within-group forward-fill (most recent past),
      (2) within-group backward-fill (nearest future),
      (3) remaining NaNs -> global column medians.

    Behavior:
      - Operations are performed independently within each group defined by `groupby`.
      - If no past value for an ID: step (2) uses the next future value.
      - If an ID never has a value for a column: step (3) uses the *global* median.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    groupby : str | Sequence[str]
        Column(s) defining the group/ID (e.g., "PatientID"). Required.
    time_col : str | None
        Optional column defining chronological order inside each group. If provided,
        rows are sorted by [groupby..., time_col] before filling.
    exclude_cols : Iterable[str] | None
        Columns to skip from imputation (e.g., labels, IDs, timestamps).
    global_medians : pd.Series | None
        Pre-computed global medians by column (index = column names). If None,
        medians are computed from `df` (pre-imputation) over numeric columns.
    only_numeric : bool
        If True (default), only numeric columns are imputed. Non-numeric columns
        are left unchanged.
    inplace : bool
        If True, modify `df` in place. Otherwise returns a copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with NaNs imputed as described.
    """
    if not isinstance(groupby, (list, tuple)):
        groupby_cols = [groupby]
    else:
        groupby_cols = list(groupby)
    if not groupby_cols:
        raise ValueError("`groupby` must contain at least one column name.")

    work_df = df if inplace else df.copy()

    # Determine columns to impute
    if only_numeric:
        cols_to_impute = _numeric_imputable_columns(
            work_df, exclude_cols or [], groupby_cols, time_col
        )
    else:
        exclude = set(_ensure_list(exclude_cols)) | set(groupby_cols)
        if time_col:
            exclude.add(time_col)
        cols_to_impute = [c for c in work_df.columns if c not in exclude]

    if not cols_to_impute:
        return work_df

    # Compute global medians if not provided (from original values)
    if global_medians is None:
        global_medians = work_df[cols_to_impute].median(numeric_only=True)

    # Sort to ensure correct forward/backward semantics within groups
    work_df = _sorted_for_group_ops(work_df, groupby_cols, time_col)

    # 1) within-group forward fill
    g = work_df.groupby(groupby_cols, sort=False, group_keys=False)
    ffilled = g[cols_to_impute].ffill()

    # # 2) within-group backward fill ONLY where still NaN
    # remaining_mask = ffilled.isna()
    # if remaining_mask.values.any():
    #     bfilled = g[cols_to_impute].bfill()
    #     # Preserve any non-NaNs from ffill
    #     ffilled = ffilled.where(~remaining_mask, bfilled)

    work_df[cols_to_impute] = ffilled

    # 3) global medians for any remaining NaNs
    still_nan = work_df[cols_to_impute].isna()
    if still_nan.values.any():
        for col in cols_to_impute:
            if still_nan[col].any():
                m = global_medians.get(col, np.nan)
                if pd.notna(m):
                    work_df.loc[still_nan[col], col] = m
                # If median is NaN (all-missing column), we leave as NaN.

    return work_df


# ---------- Folder-level convenience ----------

def _collect_global_numeric_medians_across_files(
    paths: List[str],
    sep: Optional[str] = None,
    usecols: Optional[Sequence[str]] = None,
    read_kwargs: Optional[Dict] = None,
) -> pd.Series:
    """
    Compute true global medians across many files by concatenating numeric columns.
    This is simple and correct; for huge corpora, consider chunking or sampling.
    """
    read_kwargs = dict(read_kwargs or {})
    frames = []
    for p in paths:
        this_sep = sep or _infer_sep_from_ext(p, default=",")
        df = pd.read_csv(p, sep=this_sep, **read_kwargs)
        if usecols:
            df = df[list(set(usecols) & set(df.columns))]
        frames.append(df.select_dtypes(include=[np.number]))
    if not frames:
        return pd.Series(dtype=float)
    big = pd.concat(frames, axis=0, ignore_index=True)
    return big.median(numeric_only=True)


def impute_folder(
    input_dir: str,
    output_dir: str,
    *,
    groupby: Union[str, Sequence[str]],
    time_col: Optional[str] = None,
    exclude_cols: Optional[Iterable[str]] = None,
    pattern: str = "*.csv",          # <-- default to CSV
    sep: Optional[str] = None,        # if None, inferred per file by extension
    sort_by_time: bool = True,
    compute_medians_across_folder: bool = True,
    read_kwargs: Optional[Dict] = None,
    write_kwargs: Optional[Dict] = None,
) -> List[str]:
    """
    Impute all files in `input_dir` matching `pattern` and write to `output_dir`.
    Default file type is CSV; PSV/TSV also supported via extension or `sep`.

    Steps per file:
        1) group-wise forward-fill
        2) group-wise backward-fill (only remaining NaNs)
        3) fill any remaining with *global medians* (computed across the folder by default)

    Parameters
    ----------
    input_dir : str
        Folder containing input files.
    output_dir : str
        Destination folder (created if missing).
    groupby : str | Sequence[str]
        Group/ID column(s).
    time_col : str | None
        Optional time column for chronological order within group.
    exclude_cols : Iterable[str] | None
        Columns to skip (e.g., IDs, time, labels).
    pattern : str
        Glob pattern (default "*.csv"). Change to "*.psv" if needed.
    sep : str | None
        Delimiter to use for both reading & writing. If None, inferred by extension.
    sort_by_time : bool
        If True and `time_col` is present, sort by group + time before filling.
    compute_medians_across_folder : bool
        If True, compute column medians across all files; else per-file.
    read_kwargs : dict | None
        Extra kwargs for pd.read_csv (e.g., dtype maps).
    write_kwargs : dict | None
        Extra kwargs for DataFrame.to_csv (exclude sep/index; handled here).

    Returns
    -------
    list[str]
        Paths of written files.
    """
    os.makedirs(output_dir, exist_ok=True)
    read_kwargs = dict(read_kwargs or {})
    write_kwargs = dict(write_kwargs or {})
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' in: {input_dir}")

    # Figure out which columns will be considered for medians (numeric only).
    # We detect using the first file and then compute medians across all files for those columns.
    first_path = paths[0]
    first_sep = sep or _infer_sep_from_ext(first_path, default=",")
    probe_df = pd.read_csv(first_path, sep=first_sep, nrows=100, **read_kwargs)
    candidate_cols = _numeric_imputable_columns(
        probe_df, exclude_cols or [], _ensure_list(groupby), time_col
    )

    global_medians = None
    if compute_medians_across_folder and candidate_cols:
        global_medians = _collect_global_numeric_medians_across_files(
            paths, sep=sep, usecols=candidate_cols, read_kwargs=read_kwargs
        )

    written: List[str] = []
    for p in paths:
        this_sep = sep or _infer_sep_from_ext(p, default=",")
        df = pd.read_csv(p, sep=this_sep, **read_kwargs)

        # Optional sort for correctness (nan_remover also sorts, but do it here if desired)
        if sort_by_time and time_col and time_col in df.columns:
            df = _sorted_for_group_ops(df, _ensure_list(groupby), time_col)

        # Per-file medians (fallback) if folder medians not computed
        med = global_medians
        if med is None and candidate_cols:
            med = df[candidate_cols].median(numeric_only=True)

        out = nan_remover(
            df,
            groupby=groupby,
            time_col=time_col,
            exclude_cols=exclude_cols,
            global_medians=med,
            only_numeric=True,
            inplace=False,
        )

        # Preserve original extension & delimiter unless `sep` provided
        out_ext = os.path.splitext(p)[1]
        out_sep = this_sep
        if sep is not None:
            # caller wants an explicit delimiter; keep file extension but use provided sep
            out_sep = sep

        out_path = os.path.join(output_dir, os.path.basename(p))
        out.to_csv(out_path, sep=out_sep, index=False, **write_kwargs)
        written.append(out_path)

    return written

def main():
    # import data
    df = pd.read_csv("data/MIMIC-ED/event_level_v2.csv")

    # count NaNs
    nan_counts = df.isna().sum().sum()
    print("Number of NaNs in the DataFrame:", nan_counts)

    #change dtype of 'pain' column to numeric
    df['pain'] = pd.to_numeric(df['pain'], errors='coerce')

    # change "is_antibiotic" column to boolean then to numeric
    df['is_antibiotic'] = df['is_antibiotic'].map({'True': True, 'False': False})
    df['is_antibiotic'] = df['is_antibiotic'].astype('boolean')
    # if na, fill with False
    df['is_antibiotic'] = df['is_antibiotic'].fillna(False)
    # convert boolean to int
    df['is_antibiotic'] = df['is_antibiotic'].astype(int)

    # drop unnecessary columns
    df_dropped = df.drop(columns=["rhythm", "event_type","name","name_lower","etcdescription","first_abx_time"])

    # clean the numeric columns with nans
    df_cleaned = nan_remover(df_dropped, groupby="stay_id", time_col="charttime")
    print("Number of NaNs after cleaning:", df_cleaned.isna().sum().sum())
    df_cleaned.to_csv("data/MIMIC-ED/event_level_cleaned_v2.csv", index=False)
    print("✅ event_level_cleaned_v2.csv created (NaNs removed)")


if __name__ == "__main__":
    main()