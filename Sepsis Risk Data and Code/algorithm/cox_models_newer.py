from __future__ import annotations
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import warnings

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lifelines import CoxTimeVaryingFitter
from lifelines.utils import concordance_index

# NEW: plotting backend (headless-safe)
import matplotlib
matplotlib.use("Agg")  # ensure we can save figures without a display
import matplotlib.pyplot as plt

# ----------------------------------------------------
# Config
# ----------------------------------------------------
DATA_DIR = Path("data/MIMIC-ED")
DATA_PATH = DATA_DIR / "cox_timevarying_train.csv"
OUT_DIR = Path("outputs_cox")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Column names
ID_COL = "stay_id"
START_COL = "start"
STOP_COL = "stop"
EVENT_COL = "event"

RNG = 42
TEST_SIZE = 0.2
PENALIZER = 0.1  # Increased penalizer for regularization with more features

# ----------------------------------------------------
# Utilities
# ----------------------------------------------------
def load_and_preprocess_data(data_path: Path) -> tuple[pd.DataFrame, PCA | None]:
    """Load and preprocess the time-varying data."""
    pca_model = None
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded data shape: {df.shape}")

    # Load comorbidity columns and ClinicalBERT embeddings from parquet file
    parquet_path = DATA_DIR / "cox_timevarying_with_labs_cmorbid_cbert.parquet"
    print(f"Loading comorbidity and ClinicalBERT data from {parquet_path}")
    try:
        import fastparquet
        pf = fastparquet.ParquetFile(str(parquet_path))
        # Define comorbidity columns to load
        comorbidity_cols = ['aids', 'ami', 'canc', 'cevd', 'chf', 'copd', 'dementia',
                           'diab', 'diabwc', 'hp', 'metacanc', 'mld', 'msld', 'pud',
                           'pvd', 'rend', 'rheumd', 'comorbidity_score']
        # Load comorbidity data
        cols_to_load = ['stay_id'] + comorbidity_cols
        comorbid_df = pf.to_pandas(columns=cols_to_load)
        print(f"Loaded comorbidity data shape: {comorbid_df.shape}")

        # Load ClinicalBERT embeddings
        print("Loading ClinicalBERT embeddings...")
        embedding_df = pf.to_pandas(columns=['stay_id', 'clinicalbert_emb'])
        print(f"Loaded embedding data shape: {embedding_df.shape}")

        # Process embeddings: convert list to numpy array and apply PCA
        embedding_arrays = []
        valid_stay_ids = []
        for _, row in embedding_df.iterrows():
            if row['clinicalbert_emb'] is not None and len(row['clinicalbert_emb']) == 768:
                embedding_arrays.append(row['clinicalbert_emb'])
                valid_stay_ids.append(row['stay_id'])

        if embedding_arrays:
            embedding_matrix = np.array(embedding_arrays)
            print(f"Embedding matrix shape: {embedding_matrix.shape}")

            # Apply PCA to reduce dimensionality
            n_components = min(10, embedding_matrix.shape[0], embedding_matrix.shape[1])  # Reduced to 10 components
            pca = PCA(n_components=n_components, random_state=RNG)
            embedding_pca = pca.fit_transform(embedding_matrix)
            pca_model = pca  # Store for returning

            print(f"Reduced embeddings to {n_components} components")
            print(f"Explained variance ratio: {pca.explained_variance_ratio_[:5]}")  # Show first 5

            # Create DataFrame with PCA components
            pca_cols = [f'cbert_pca_{i}' for i in range(n_components)]
            embedding_pca_df = pd.DataFrame(embedding_pca, columns=pca_cols)
            embedding_pca_df['stay_id'] = valid_stay_ids

            # Merge comorbidity and embedding data
            comorbid_df = comorbid_df.merge(embedding_pca_df, on='stay_id', how='left')
            print(f"Data shape after adding ClinicalBERT PCA: {comorbid_df.shape}")
        else:
            print("Warning: No valid ClinicalBERT embeddings found")

        # Merge comorbidity/embedding data with main dataframe
        df = df.merge(comorbid_df, on='stay_id', how='left')
        print(f"Data shape after merging comorbidities and embeddings: {df.shape}")

    except Exception as e:
        print(f"Warning: Could not load comorbidity/embedding data: {e}")
        print("Continuing with original data...")

    # Check required columns
    required_cols = {ID_COL, START_COL, STOP_COL, EVENT_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate time intervals
    if (df[STOP_COL] <= df[START_COL]).any():
        raise ValueError("Found invalid time intervals (stop <= start)")

    if not df[EVENT_COL].isin([0, 1]).all():
        raise ValueError("Event column must contain only 0 or 1")

    # Fill any NaN values with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Filled NaNs in {col} with median: {median_val}")

    return df, pca_model

def get_features(df: pd.DataFrame) -> list[str]:
    """Get feature columns excluding ID, time, and event columns."""
    exclude_cols = {ID_COL, START_COL, STOP_COL, EVENT_COL}
    features = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    # Remove features with no variance
    features = [col for col in features if df[col].std() > 0]
    return features

def split_data_by_patients(df: pd.DataFrame, test_size: float = TEST_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by patient IDs to avoid data leakage."""
    unique_ids = df[ID_COL].unique()
    np.random.seed(RNG)
    np.random.shuffle(unique_ids)

    n_test = int(len(unique_ids) * test_size)
    test_ids = set(unique_ids[:n_test])
    train_ids = set(unique_ids[n_test:])

    train_df = df[df[ID_COL].isin(train_ids)].copy()
    test_df = df[df[ID_COL].isin(test_ids)].copy()

    print(f"Train patients: {len(train_ids)}, Test patients: {len(test_ids)}")
    print(f"Train intervals: {len(train_df)}, Test intervals: {len(test_df)}")

    return train_df, test_df

def scale_features(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale features using StandardScaler fitted on training data."""
    scaler = StandardScaler()

    # Fit on training data
    train_scaled = scaler.fit_transform(train_df[features])
    test_scaled = scaler.transform(test_df[features])

    # Update dataframes
    for i, col in enumerate(features):
        train_df[col] = train_scaled[:, i]
        test_df[col] = test_scaled[:, i]

    return train_df, test_df, scaler

def calculate_concordance_index_time_varying(ctv: CoxTimeVaryingFitter, df: pd.DataFrame, features: list[str]) -> float:
    """Calculate concordance index for time-varying Cox model."""
    try:
        # Try using the model's built-in score method
        ci = ctv.score(df[[ID_COL, START_COL, STOP_COL, EVENT_COL] + features],
                      scoring_method="concordance_index")
        return ci
    except Exception as e:
        print(f"Built-in scoring failed: {e}, using manual calculation")
        # Fallback: simplified manual calculation
        # Reset index to avoid indexing issues
        df_reset = df.reset_index(drop=True)
        predictions = ctv.predict_partial_hazard(df_reset)

        event_intervals = df_reset[df_reset[EVENT_COL] == 1]
        if len(event_intervals) == 0:
            return 0.5

        concordant = 0
        total = 0

        for idx, event_row in event_intervals.iterrows():
            event_time = event_row[STOP_COL]
            event_patient = event_row[ID_COL]
            event_hazard = predictions.iloc[idx]

            # Compare with intervals from other patients that end before or at the event time
            comparable = df_reset[(df_reset[ID_COL] != event_patient) & (df_reset[STOP_COL] <= event_time)]
            if len(comparable) > 0:
                comparable_hazards = predictions.iloc[comparable.index]
                concordant += (comparable_hazards < event_hazard).sum()
                total += len(comparable)

        return concordant / total if total > 0 else 0.5

def train_time_varying_cox_model(data_path: Path) -> dict:
    """Train time-varying Cox proportional hazard model and return results."""

    # Load and preprocess data
    df, pca_model = load_and_preprocess_data(data_path)

    # Get features
    features = get_features(df)
    print(f"Using {len(features)} features: {features[:5]}...")

    # Split data
    train_df, test_df = split_data_by_patients(df)

    # Scale features
    train_df, test_df, scaler = scale_features(train_df, test_df, features)

    # Fit model
    print("Fitting CoxTimeVaryingFitter...")
    ctv = CoxTimeVaryingFitter(penalizer=PENALIZER)
    ctv.fit(train_df[[ID_COL, START_COL, STOP_COL, EVENT_COL] + features],
            id_col=ID_COL, start_col=START_COL, stop_col=STOP_COL, event_col=EVENT_COL,
            show_progress=True)

    # Calculate concordance indices
    print("Calculating concordance indices...")
    train_ci = calculate_concordance_index_time_varying(ctv, train_df, features)
    test_ci = calculate_concordance_index_time_varying(ctv, test_df, features)

    print(f"Concordance Index - Train: {train_ci:.4f}, Test: {test_ci:.4f}")
    # Feature importance
    feature_importance = ctv.summary.copy()
    feature_importance['abs_coef'] = feature_importance['coef'].abs()
    feature_importance = feature_importance.sort_values('abs_coef', ascending=False)

    print("\nTop 10 features by importance:")
    for i, (idx, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"{i+1:2d}. {idx}: coef={row['coef']:.3f}, HR={row['exp(coef)']:.3f}")

    # Save results
    feature_importance.to_csv(OUT_DIR / "feature_importance_cox_tvc_new.csv")
    with open(OUT_DIR / "cox_tvc_new.pkl", "wb") as f:
        pickle.dump(ctv, f)

    # Save scaler
    with open(OUT_DIR / "cox_tvc_scaler.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "features": features, "pca": pca_model}, f)

    return {
        "train_cindex": train_ci,
        "test_cindex": test_ci,
        "n_train_patients": train_df[ID_COL].nunique(),
        "n_test_patients": test_df[ID_COL].nunique(),
        "n_features": len(features),
        "model_path": str(OUT_DIR / "cox_tvc_new.pkl"),
        "feature_importance_path": str(OUT_DIR / "feature_importance_cox_tvc_new.csv")
    }

# ----------------------------------------------------
# Main execution
# ----------------------------------------------------
if __name__ == "__main__":
    print("Training Time-Varying Cox Proportional Hazard Model")
    print("=" * 60)

    results = train_time_varying_cox_model(DATA_PATH)

    print("\n" + "=" * 60)
    print("FINAL RESULTS:")
    print(".4f")
    print(".4f")
    print(f"Training patients: {results['n_train_patients']}")
    print(f"Test patients: {results['n_test_patients']}")
    print(f"Number of features: {results['n_features']}")
    print(f"Model saved to: {results['model_path']}")
    print(f"Feature importance saved to: {results['feature_importance_path']}")
    print("=" * 60)
