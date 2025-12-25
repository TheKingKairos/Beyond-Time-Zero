# xgb_cox_grid_search.py
# Grid search for XGBoost Cox PH on GPU, using C-index for model selection.

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor

# ---- C-index (works with either scikit-survival or lifelines) ----
try:
    # Preferred: proper handling for censored data; expects risk scores (higher = worse).
    from sksurv.metrics import concordance_index_censored
    def cindex_from_risk(event, time, risk):
        return float(concordance_index_censored(event.astype(bool), time, risk)[0])
except Exception:
    # Fallback: lifelines; expects "higher = longer survival", so flip sign of risk.
    from lifelines.utils import concordance_index
    def cindex_from_risk(event, time, risk):
        return float(concordance_index(time, -risk, event_observed=event))

def run_xgb_cox_grid_search(df: pd.DataFrame, random_state: int = 42):
    """
    df: DataFrame with columns:
        - duration (float): time-to-event or time-to-censoring
        - event (int/bool): 1 if event occurred, 0 if censored
        - all other columns are numeric features
    """
    # --------- Prepare data ---------
    assert {'duration', 'event'}.issubset(df.columns), "Missing duration/event columns."
    X = df.drop(columns=['duration', 'event'])
    durations = df['duration'].to_numpy(dtype=float)
    events = df['event'].astype(int).to_numpy()

    # XGBoost Cox expects a single label vector; use +time for events, -time for censored.
    # (negative labels are treated as right-censored by XGBoost Cox)
    # Ref: XGBoost docs (and R help): survival:cox -> negative values are considered right censored.
    y_cox = np.where(events == 1, durations, -durations).astype(float)

    # --------- Estimator (GPU) ---------
    base_estimator = XGBRegressor(
        objective='survival:cox',
        eval_metric='cox-nloglik',        # appropriate for Cox training
        tree_method='gpu_hist',           # GPU training
        predictor='gpu_predictor',        # GPU inference
        # Don't set device='cuda' unless you're on XGBoost >=2.0 and prefer that API
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=random_state,
        verbosity=0
    )

    # --------- Hyperparameter grid ---------
    trials = [
    # --- depth 3 (shallower, needs more trees) ---
    {"max_depth": [3], "learning_rate": [0.05], "n_estimators": [800], "min_child_weight": [1], "subsample": [0.8], "colsample_bytree": [0.8], "reg_lambda": [1.0], "gamma": [0.0]},
    # {"max_depth": [3], "learning_rate": [0.05], "n_estimators": [800], "min_child_weight": [5], "subsample": [0.8], "colsample_bytree": [0.8], "reg_lambda": [1.0], "gamma": [0.0]},
    # {"max_depth": [3], "learning_rate": [0.10], "n_estimators": [500], "min_child_weight": [1], "subsample": [0.8], "colsample_bytree": [0.8], "reg_lambda": [1.0], "gamma": [0.0]},
    # {"max_depth": [3], "learning_rate": [0.10], "n_estimators": [500], "min_child_weight": [5], "subsample": [0.8], "colsample_bytree": [0.8], "reg_lambda": [1.0], "gamma": [0.0]},

    # # --- depth 6 (deeper, fewer trees) ---
    # {"max_depth": [6], "learning_rate": [0.05], "n_estimators": [800], "min_child_weight": [1], "subsample": [0.8], "colsample_bytree": [0.8], "reg_lambda": [1.0], "gamma": [0.0]},
    # {"max_depth": [6], "learning_rate": [0.05], "n_estimators": [800], "min_child_weight": [5], "subsample": [0.8], "colsample_bytree": [0.8], "reg_lambda": [1.0], "gamma": [0.0]},
    # {"max_depth": [6], "learning_rate": [0.10], "n_estimators": [500], "min_child_weight": [1], "subsample": [0.8], "colsample_bytree": [0.8], "reg_lambda": [1.0], "gamma": [0.0]},
    # {"max_depth": [6], "learning_rate": [0.10], "n_estimators": [500], "min_child_weight": [5], "subsample": [0.8], "colsample_bytree": [0.8], "reg_lambda": [1.0], "gamma": [0.0]},

    # # --- a quick regularization + gamma poke (depth 4 baseline) ---
    # {"max_depth": [4], "learning_rate": [0.10], "n_estimators": [500], "min_child_weight": [1], "subsample": [0.8], "colsample_bytree": [0.8], "reg_lambda": [0.0], "gamma": [0.0]},
    # {"max_depth": [4], "learning_rate": [0.10], "n_estimators": [500], "min_child_weight": [1], "subsample": [0.8], "colsample_bytree": [0.8], "reg_lambda": [5.0], "gamma": [1.0]},

    # # --- sampling extremes (to check variance/overfit) ---
    # {"max_depth": [4], "learning_rate": [0.05], "n_estimators": [800], "min_child_weight": [5], "subsample": [0.7], "colsample_bytree": [0.6], "reg_lambda": [1.0], "gamma": [0.0]},
    # {"max_depth": [4], "learning_rate": [0.05], "n_estimators": [800], "min_child_weight": [5], "subsample": [1.0], "colsample_bytree": [1.0], "reg_lambda": [1.0], "gamma": [0.0]},
    ]

    # --------- C-index scorer for GridSearchCV ---------
    # This callable gets (estimator, X_valid, y_valid_cox), where y_valid_cox encodes both
    # time and censoring via sign. We recover durations/events to compute C-index.
    def cindex_scorer(estimator, X_valid, y_valid_cox):
        risk = estimator.predict(X_valid)               # XGBoost Cox returns (log) risk scores
        time = np.abs(y_valid_cox)
        event = (y_valid_cox > 0).astype(int)
        return cindex_from_risk(event, time, risk)

    # --------- Cross-validation & Grid Search ---------
    # IMPORTANT: n_jobs=1 to avoid multiple concurrent GPU trainings.
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    gscv = GridSearchCV(
        estimator=base_estimator,
        param_grid=trials,
        scoring=cindex_scorer,
        cv=cv,
        refit=True,      # refit best params on the whole dataset using the Cox objective
        verbose=2,
        n_jobs=1
    )

    gscv.fit(X, y_cox)

    # --------- Results ---------
    best_model = gscv.best_estimator_
    print("\nBest C-index (CV mean): {:.4f}".format(gscv.best_score_))
    print("Best params:", gscv.best_params_)
    # save best params to a file if desired
    open("model_xgboost_sa/best_xgb_cox_params.txt", "w").write(str(gscv.best_params_))

    # Example: compute in-sample C-index of the refit model
    best_risk = best_model.predict(X)
    in_sample_c = cindex_from_risk(events, durations, best_risk)
    print("In-sample C-index of refit model: {:.4f}".format(in_sample_c))

    # Optional: save the model
    best_model.save_model("model_xgboost_sa/best_xgb_cox.json")

    return best_model, gscv

# ---------- Example usage ----------
if __name__ == "__main__":
    # If you already have df = pd.read_csv(...), just pass it to the function.
    df = pd.read_csv("data/MIMIC-ED/cox_static_landmark_stacked.csv")
    best_model, gscv = run_xgb_cox_grid_search(df)
    pass
