"""Auto-generated from data_cleaning.ipynb on 2025-09-09"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import os
import glob
from typing import Iterable, Optional
import pandas as pd
import numpy as np

class DataCleaner:

    def __init__(self, **config):
        """Store arbitrary configuration for the cleaning pipeline."""
        self.config = config

    def make_time_to_sepsis_label(self, df: pd.DataFrame, time_col: str='ICULOS', label_col: str='SepsisLabel', group_col: str | None=None, horizon_hours: float=24.0, out_col: str='NewLabel') -> pd.DataFrame:
        """
    Create a time-to-sepsis label:
      - label = 1 if current sepsisLabel == 1
      - label = 0 if there is no future sepsis onset after this point
      - otherwise label = max(1 - (t_sepsis - t)/horizon_hours, 0)
    
    Supports multiple sepsis episodes per patient (e.g., 0->1, back to 0, later 0->1 again).

    Args:
        df: DataFrame containing at least [time_col, label_col], and optionally group_col.
        time_col: Name of the time variable (in hours).
        label_col: Name of the binary sepsis indicator column (0/1).
        group_col: Optional patient identifier column. If provided, labeling is done per patient.
        horizon_hours: Linear ramp window before sepsis onset (default 24).
        out_col: Name of output column to write.

    Returns:
        A DataFrame with a new float column `out_col` added.
    """

        def _label_one_patient(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values(time_col).copy()
            sepsis = g[label_col].fillna(0).astype(int)
            onset = (sepsis.shift(fill_value=0) == 0) & (sepsis == 1)
            next_onset_time = g[time_col].where(onset).bfill()
            delta_hours = (next_onset_time - g[time_col]).astype(float)
            ramp = 1.0 - delta_hours / horizon_hours
            ramp = ramp.clip(lower=0.0, upper=1.0)
            ramp[next_onset_time.isna()] = 0.0
            new_label = ramp.where(sepsis == 0, 1.0)
            g[out_col] = new_label.astype(float)
            return g.sort_index()
        if group_col is None:
            return _label_one_patient(df)
        else:
            return df.groupby(group_col, group_keys=False).apply(_label_one_patient)

    def build_training_set_labels(self, input_dir: str, output_csv: str, horizon_hours: float=24.0, make_label_fn=None) -> pd.DataFrame:
        """
    Parse all .psv files under `input_dir`, create NewLabel using `make_time_to_sepsis_label`,
    and concatenate into one CSV at `output_csv`.

    Assumes each file is a single patient's time series with columns including ICULOS and SepsisLabel.
    Adds a PatientID (derived from filename: e.g., 'p000001').

    Returns the concatenated DataFrame.
    """
        if make_label_fn is None:
            raise ValueError('Please provide `make_time_to_sepsis_label` via `make_label_fn`.')
        paths = sorted(glob.glob(os.path.join(input_dir, '*.psv')))
        if not paths:
            raise FileNotFoundError(f'No .psv files found in: {input_dir}')
        all_dfs: list[pd.DataFrame] = []
        for path in paths:
            patient_id = os.path.splitext(os.path.basename(path))[0]
            df = pd.read_csv(path, sep='|')
            if 'ICULOS' not in df.columns or 'SepsisLabel' not in df.columns:
                raise ValueError(f'File {path} missing required columns ICULOS/SepsisLabel.')
            df['ICULOS'] = pd.to_numeric(df['ICULOS'], errors='coerce')
            df['SepsisLabel'] = pd.to_numeric(df['SepsisLabel'], errors='coerce').fillna(0).astype(int)
            df = df.sort_values('ICULOS').reset_index(drop=True)
            df.insert(0, 'PatientID', patient_id)
            labeled = make_label_fn(df=df, time_col='ICULOS', label_col='SepsisLabel', group_col=None, horizon_hours=horizon_hours, out_col='NewLabel')
            if isinstance(labeled, pd.Series):
                df['NewLabel'] = labeled.values
            elif isinstance(labeled, pd.DataFrame):
                if 'NewLabel' in labeled.columns:
                    if set(df.index) == set(labeled.index) and len(labeled.columns) > 1:
                        df = labeled
                    else:
                        df['NewLabel'] = labeled['NewLabel'].values
                else:
                    df['NewLabel'] = labeled.iloc[:, 0].values
            else:
                raise TypeError('make_time_to_sepsis_label must return a pandas Series or DataFrame.')
            all_dfs.append(df)
        big_df = pd.concat(all_dfs, ignore_index=True)
        big_df.to_csv(output_csv, index=False)
        return big_df

    def impute_psv_folder(self, input_dir: str, output_dir: str, avg_series: pd.Series, exclude_cols: Optional[Iterable[str]]=('PatientID', 'ICULOS', 'SepsisLabel'), sort_by_time: bool=True, time_col: str='ICULOS') -> list[str]:
        """
    For each .psv in `input_dir`, create an imputed version in `output_dir`
    with the same filename. Imputation is column-wise and follows:
        1) forward-fill (most recent past value),
        2) THEN (only if still NaN) backward-fill (nearest future value),
        3) THEN (only if still NaN) fill with `avg_series[col]` if available (and not NaN).

    Columns excluded from imputation: by default ["PatientID", "ICULOS", "SepsisLabel"].
    If a listed exclude column doesn't exist in a file, it's ignored.

    Parameters
    ----------
    input_dir : folder containing .psv files (pipe-delimited)
    output_dir : destination folder (created if missing)
    avg_series : pd.Series whose index are column names and values are global means
    exclude_cols : columns to skip from imputation
    sort_by_time : if True and `time_col` exists, sort ascending before imputation
    time_col : time column name (default "ICULOS")

    Returns
    -------
    list[str]
        Paths of written files.
    """
        os.makedirs(output_dir, exist_ok=True)
        paths = sorted(glob.glob(os.path.join(input_dir, '*.psv')))
        if not paths:
            raise FileNotFoundError(f'No .psv files found under: {input_dir}')
        exclude_set = set(exclude_cols or [])
        written = []
        for path in paths:
            df = pd.read_csv(path, sep='|')
            if sort_by_time and time_col in df.columns:
                df = df.sort_values(time_col, kind='mergesort').reset_index(drop=True)
            cols_to_impute = [c for c in df.columns if c not in exclude_set]
            if cols_to_impute:
                df[cols_to_impute] = df[cols_to_impute].ffill()
                remaining_nan_mask = df[cols_to_impute].isna()
                if remaining_nan_mask.values.any():
                    df.loc[:, cols_to_impute] = df[cols_to_impute].where(~remaining_nan_mask, df[cols_to_impute].bfill())
                still_nan = df[cols_to_impute].isna()
                if still_nan.values.any():
                    for col in cols_to_impute:
                        if still_nan[col].any():
                            avg_val = avg_series.get(col, np.nan)
                            if pd.notna(avg_val):
                                df.loc[still_nan[col], col] = avg_val
            out_path = os.path.join(output_dir, os.path.basename(path))
            df.to_csv(out_path, sep='|', index=False)
            written.append(out_path)
        return written

    def clean(self, **kw):
        """Execute the notebook’s imperative cells. Pass an optional DataFrame as df."""
        df = pd.read_csv('../data/training_setA/p000001.psv', sep='|')
        big = self.build_training_set_labels(input_dir='../data/training_setA', output_csv='../data/agg/training_setA_labeled.csv', horizon_hours=24.0, make_label_fn=self.make_time_to_sepsis_label)
        bigB = self.build_training_set_labels(input_dir='../data/training_setB', output_csv='../data/agg/training_setB_labeled.csv', horizon_hours=24.0, make_label_fn=self.make_time_to_sepsis_label)
        df_cleaned = pd.concat([pd.read_csv('../data/agg/training_setA_labeled.csv'), pd.read_csv('../data/agg/training_setB_labeled.csv')], ignore_index=True)
        df_medians = df_cleaned.drop(columns=['PatientID', 'ICULOS', 'SepsisLabel', 'NewLabel']).median()
        imputed_paths = self.impute_psv_folder(input_dir='../data/training_setA', output_dir='../data/training_setA_imputed', avg_series=df_medians, exclude_cols=('PatientID', 'ICULOS', 'SepsisLabel'))
        imputed_pathsB = self.impute_psv_folder(input_dir='../data/training_setB', output_dir='../data/training_setB_imputed', avg_series=df_medians, exclude_cols=('PatientID', 'ICULOS', 'SepsisLabel'))
        big_imputed = self.build_training_set_labels(input_dir='../data/training_setA_imputed', output_csv='../data/agg/training_setA_imputed_labeled.csv', horizon_hours=24.0, make_label_fn=self.make_time_to_sepsis_label)
        bigB_imputed = self.build_training_set_labels(input_dir='../data/training_setB_imputed', output_csv='../data/agg/training_setB_imputed_labeled.csv', horizon_hours=24.0, make_label_fn=self.make_time_to_sepsis_label)
        df_cleaned_imputed = pd.concat([pd.read_csv('../data/agg/training_setA_imputed_labeled.csv'), pd.read_csv('../data/agg/training_setB_imputed_labeled.csv')], ignore_index=True)
        df_cleaned_imputed
        df_cleaned_imputed.to_csv('../data/cleaned_imputed_labeled.csv', index=False)
        return {'df': df, 'data': None, 'dataset': None}.get('df')