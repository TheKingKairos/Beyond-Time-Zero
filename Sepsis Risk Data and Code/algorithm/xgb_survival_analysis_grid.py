from __future__ import annotations
import os, json, pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from lifelines.utils import concordance_index

# ---------- Config ----------
DATA_DIR   = Path("data/MIMIC-ED")
DT_PATH    = DATA_DIR / "discrete_time_30min_train.csv"
OUT_DIR    = Path("outputs_discrete_xgb_grid")
PLOTS_DIR  = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "stay_id"
TBIN_COL = "t_bin"
EVENT_COL = "event"
LM_TIME_COL = "landmark_charttime"        # required for C-index construction
BIN_MINUTES = 30.0
BIN_HOURS   = BIN_MINUTES / 60.0
K_HOURS     = 10.0
K_BINS      = int(np.ceil(K_HOURS / BIN_HOURS))

RNG       = 42
TEST_SIZE = 0.2
EARLY_STOP_ROUNDS = 30
NTHREAD   = os.cpu_count()

# ---------- Utils ----------
def _xgb_gpu_kwargs():
    ver = tuple(int(x) for x in xgb.__version__.split(".")[:2])
    kw = dict(max_bin=256, nthread=NTHREAD)
    if ver >= (2, 0):
        kw.update(tree_method="hist", device="cuda")
    else:
        kw.update(tree_method="gpu_hist", gpu_id=0, predictor="gpu_predictor")
    return kw

def _coerce_numeric_inplace(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c not in df.columns:
            continue
        if not pd.api.types.is_bool_dtype(df[c]) and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if not pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype("float32")
    df[cols] = df[cols].fillna(0)

def _numeric_or_coercible_feature_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    num = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    base = [c for c in num if c not in exclude]
    cand = [c for c in df.columns if (c not in exclude) and (c not in base)]
    looks_numeric = []
    for c in cand:
        if pd.api.types.is_object_dtype(df[c]):
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().mean() >= 0.5:
                looks_numeric.append(c)
    cols = base + looks_numeric
    _coerce_numeric_inplace(df, cols)
    cols = [c for c in cols if df[c].std(ddof=0) > 0]
    return cols

def _build_snapshot_outcomes_from_pp(df_pp: pd.DataFrame, K_bins: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if LM_TIME_COL not in df_pp.columns:
        raise ValueError(f"{LM_TIME_COL} not found; enable include_landmark_cols=True in formatter.")
    latest = df_pp.groupby(ID_COL, as_index=False)[LM_TIME_COL].max().rename(columns={LM_TIME_COL:"latest_lm"})
    df2 = df_pp.merge(latest, on=ID_COL, how="inner")
    lr = df2[df2[LM_TIME_COL] == df2["latest_lm"]].copy()
    if lr.empty: return pd.DataFrame(), pd.DataFrame()

    agg = lr.groupby(ID_COL).agg(
        max_bin=(TBIN_COL, "max"),
        any_event=(EVENT_COL, "max"),
        event_bin=(EVENT_COL, lambda s: np.nan if s.sum()==0 else int(np.argmax(s.values != 0) + 1))
    ).reset_index()

    def _dur(row):
        if int(row["any_event"])==1 and not pd.isna(row["event_bin"]):
            return float(int(row["event_bin"]) * BIN_HOURS)
        return float(int(row["max_bin"]) * BIN_HOURS)

    outcomes = agg.assign(
        duration_hours=agg.apply(_dur, axis=1).astype(float),
        event=agg["any_event"].astype(int)
    )[[ID_COL, "duration_hours", "event"]]

    lr = lr.sort_values([ID_COL, TBIN_COL], kind="mergesort")
    snaps = lr.groupby(ID_COL, as_index=False).head(1).reset_index(drop=True)
    return snaps, outcomes

def _score_snaps_vectorized(model: xgb.XGBClassifier, snaps_df: pd.DataFrame,
                            base_cols: List[str], K_bins: int) -> pd.DataFrame:
    if snaps_df.empty:
        return pd.DataFrame(columns=[ID_COL, f"prob_event_within_{K_bins}bins"])
    snapped = snaps_df[[ID_COL] + base_cols].copy()
    _coerce_numeric_inplace(snapped, base_cols)
    ids = snapped[ID_COL].to_numpy()
    base_block = snapped[base_cols].to_numpy()
    X_base = np.repeat(base_block, K_bins, axis=0)
    t_bins = np.tile(np.arange(1, K_bins+1, dtype=np.int16), reps=len(snapped))
    import pandas as _pd
    G = _pd.DataFrame(X_base, columns=base_cols)
    G.insert(0, TBIN_COL, t_bins)
    _coerce_numeric_inplace(G, [TBIN_COL] + base_cols)
    proba = model.predict_proba(G[[TBIN_COL] + base_cols])[:, 1].astype("float32")
    proba = proba.reshape(len(snapped), K_bins)
    riskK = 1.0 - np.prod(1.0 - proba, axis=1)
    return pd.DataFrame({ID_COL: ids, f"prob_event_within_{K_bins}bins": riskK})

# ---------- Grid Search ----------
def run_grid_search(dt_path: Path) -> Dict:
    # load
    head_cols = pd.read_csv(dt_path, nrows=0).columns.tolist()
    parse_cols = [c for c in [LM_TIME_COL] if c in head_cols]
    df = pd.read_csv(dt_path, parse_dates=parse_cols)

    need = {ID_COL, TBIN_COL, EVENT_COL}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in PP data: {missing}")

    exclude = [ID_COL, TBIN_COL, EVENT_COL]
    if LM_TIME_COL in df.columns:
        exclude.append(LM_TIME_COL)

    base_feats = _numeric_or_coercible_feature_cols(df, exclude=exclude)
    _coerce_numeric_inplace(df, [TBIN_COL, EVENT_COL])

    # hold-out test split by stay_id
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RNG)
    idx = np.arange(len(df))
    tr_idx, te_idx = next(gss.split(idx, groups=df[ID_COL].values))
    df_tr, df_te = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()

    # param grid (keep modest for speed; expand later)
    param_grid = [
        {"max_depth": [4, 6], "learning_rate": [0.05, 0.10],
         "subsample": [0.7], "colsample_bytree":[0.7, 0.9],
         "min_child_weight":[1, 5], "reg_lambda":[1.0, 3.0], "n_estimators":[400, 800]},
    ]

    # inner CV on df_tr
    gkf = GroupKFold(n_splits=3)
    groups = df_tr[ID_COL].values

    def _fit_one(X_tr, y_tr, X_val, y_val, params) -> xgb.XGBClassifier:
        spw = (int((y_tr==0).sum()) / max(1, int((y_tr==1).sum()))) if (y_tr.sum()>0) else 1.0
        mdl = xgb.XGBClassifier(
            random_state=RNG, scale_pos_weight=spw, eval_metric="aucpr",
            **params, **_xgb_gpu_kwargs()
        )
        # early stopping on validation fold
        fitted=False
        try:
            from xgboost.callback import EarlyStopping
            es = EarlyStopping(rounds=EARLY_STOP_ROUNDS, metric_name="aucpr", save_best=True, maximize=True)
            mdl.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[es], verbose=False)
            fitted=True
        except Exception:
            pass
        if not fitted:
            try:
                mdl.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=EARLY_STOP_ROUNDS, verbose=False)
            except TypeError:
                mdl.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return mdl

    # features matrix builders
    def _xy(df_): 
        _coerce_numeric_inplace(df_, [TBIN_COL] + base_feats)
        return df_[[TBIN_COL] + base_feats], df_[EVENT_COL].astype(int).values

    best = {"score": -np.inf, "params": None, "cv_scores": None}
    for grid in param_grid:
        import itertools
        keys, vals = zip(*grid.items())
        for combo in itertools.product(*vals):
            params = dict(zip(keys, combo))
            fold_scores = []
            for fold, (tr_i, va_i) in enumerate(gkf.split(df_tr, groups=groups), 1):
                tr_fold = df_tr.iloc[tr_i].copy()
                va_fold = df_tr.iloc[va_i].copy()
                Xf_tr, yf_tr = _xy(tr_fold)
                Xf_va, yf_va = _xy(va_fold)
                mdl = _fit_one(Xf_tr, yf_tr, Xf_va, yf_va, params)
                p_va = mdl.predict_proba(Xf_va)[:,1]
                ap = average_precision_score(yf_va, p_va)
                # optionally track auc too
                try:
                    auc = roc_auc_score(yf_va, p_va) if len(np.unique(yf_va))>1 else np.nan
                except Exception:
                    auc = np.nan
                fold_scores.append({"ap": ap, "auc": auc})
            mean_ap = float(np.mean([s["ap"] for s in fold_scores]))
            print(f"Params {params} -> CV mean AP={mean_ap:.4f}")
            if mean_ap > best["score"]:
                best = {"score": mean_ap, "params": params, "cv_scores": fold_scores}

    # Refit best on full training split
    Xtr, ytr = _xy(df_tr)
    Xte, yte = _xy(df_te)
    spw = (int((ytr==0).sum()) / max(1, int((ytr==1).sum()))) if (ytr.sum()>0) else 1.0
    best_mdl = xgb.XGBClassifier(
        random_state=RNG, scale_pos_weight=spw, eval_metric="aucpr",
        **best["params"], **_xgb_gpu_kwargs()
    )
    # early stopping on test as eval_set is OK for selecting n_estimators; metrics will be computed on full test after
    try:
        from xgboost.callback import EarlyStopping
        es = EarlyStopping(rounds=EARLY_STOP_ROUNDS, metric_name="aucpr", save_best=True, maximize=True)
        best_mdl.fit(Xtr, ytr, eval_set=[(Xte, yte)], callbacks=[es], verbose=False)
    except Exception:
        try:
            best_mdl.fit(Xtr, ytr, eval_set=[(Xte, yte)], early_stopping_rounds=EARLY_STOP_ROUNDS, verbose=False)
        except TypeError:
            best_mdl.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

    # Row-level test metrics
    p_tr = best_mdl.predict_proba(Xtr)[:,1]
    p_te = best_mdl.predict_proba(Xte)[:,1]
    row_metrics = dict(
        train_ap=float(average_precision_score(ytr, p_tr)),
        test_ap=float(average_precision_score(yte, p_te)),
        train_auc=float(roc_auc_score(ytr, p_tr)) if len(np.unique(ytr))>1 else None,
        test_auc=float(roc_auc_score(yte, p_te)) if len(np.unique(yte))>1 else None,
    )

    # Patient-level C-index at K hours from latest snapshot
    snaps_all, outcomes_all = _build_snapshot_outcomes_from_pp(df, K_BINS)
    snaps_tr  = snaps_all[snaps_all[ID_COL].isin(set(df_tr[ID_COL]))].reset_index(drop=True)
    snaps_te  = snaps_all[snaps_all[ID_COL].isin(set(df_te[ID_COL]))].reset_index(drop=True)
    risk_tr = _score_snaps_vectorized(best_mdl, snaps_tr, base_cols=base_feats, K_bins=K_BINS)
    risk_te = _score_snaps_vectorized(best_mdl, snaps_te, base_cols=base_feats, K_bins=K_BINS)

    def _cindex(outcomes: pd.DataFrame, risk: pd.DataFrame) -> Tuple[float|None, int]:
        if outcomes.empty or risk.empty:
            return None, 0
        m = outcomes.merge(risk, on=ID_COL, how="inner")
        m = m[(m["duration_hours"] > 0) & m["duration_hours"].notna()]
        if m.empty: return None, 0
        c = float(concordance_index(
            event_times=m["duration_hours"].values,
            predicted_scores=m[f"prob_event_within_{K_BINS}bins"].values,
            event_observed=m["event"].values
        ))
        return c, int(len(m))

    out_tr = outcomes_all[outcomes_all[ID_COL].isin(set(snaps_tr[ID_COL]))].reset_index(drop=True)
    out_te = outcomes_all[outcomes_all[ID_COL].isin(set(snaps_te[ID_COL]))].reset_index(drop=True)
    c_train, n_train = _cindex(out_tr, risk_tr)
    c_test,  n_test  = _cindex(out_te, risk_te)

    result = {
        "best_params": best["params"],
        "cv_mean_ap": float(best["score"]),
        "cv_fold_scores": best["cv_scores"],
        "row_metrics": row_metrics,
        "cindex": {
            "train_cindex": c_train,
            "test_cindex":  c_test,
            "n_train_for_cindex": n_train,
            "n_test_for_cindex":  n_test,
            "K_hours": K_HOURS,
        },
        "features": {"order": [TBIN_COL] + base_feats, "base_features": base_feats},
    }

    # save artifacts
    best_mdl.save_model(str(OUT_DIR / "xgb_discrete_best.json"))
    with open(OUT_DIR / "xgb_discrete_best_features.pkl", "wb") as f:
        pickle.dump(result["features"], f)
    with open(OUT_DIR / "grid_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n== Grid Search Complete ==")
    print("Best params:", best["params"])
    print(f"CV mean AP: {best['score']:.4f}")
    print("Row-level (train/test) AP:", row_metrics["train_ap"], row_metrics["test_ap"])
    print("Row-level (train/test) AUC:", row_metrics["train_auc"], row_metrics["test_auc"])
    print("C-index (train/test):", c_train, c_test, "with N=", n_train, n_test)
    print(f"Artifacts saved to: {OUT_DIR}")
    return result

if __name__ == "__main__":
    run_grid_search(DT_PATH)
