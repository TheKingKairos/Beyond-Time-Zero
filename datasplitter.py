import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_validate, GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, log_loss
import joblib
import os


# generic cleaner for same process as above
def mimic_train_test_split(Path: str="../data/MIMIC-ED/event_level_cleaned_sirs.csv",
                            test_size: float = 0.2, 
                            random_state: int = 42):
    # load cleaned data
    df_cleaned = pd.read_csv(Path)

    # group patients by "stay_id" and pick patients
    septic_patient_ids = df_cleaned[df_cleaned["sepsis_dx_any"] == 1]["stay_id"].unique()
    count_septic = len(septic_patient_ids)
    nonseptic_patient_ids = df_cleaned[df_cleaned["sepsis_dx_any"] == 0]["stay_id"].unique()
    random_nonseptic_patient_ids = np.random.choice(nonseptic_patient_ids, size=count_septic * 2, replace=False)
    random_patient_ids = np.concatenate([septic_patient_ids, random_nonseptic_patient_ids])
    np.random.shuffle(random_patient_ids)

    # filter df to only include these patients
    df_small = df_cleaned[df_cleaned["stay_id"].isin(random_patient_ids)]

    # group shuffle split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    y = df_small["sepsis_dx"]
    X = df_small.drop(columns=["sepsis_dx", "charttime", "is_antibiotic","sirs_count", "sirs_ge2","sepsis_onset_time", "is_sepsis_onset", "sepsis_dx_any"])
    train_idx, test_idx = next(gss.split(X, y, groups=df_small["stay_id"]))
    x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    x_train = x_train.drop(columns=["stay_id", "subject_id"])
    x_test = x_test.drop(columns=["stay_id", "subject_id"])

    return x_train, x_test, y_train, y_test


