import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve
)

def set_global_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)

def fit_with_early_stopping(model, X_tr, y_tr, X_va, y_va, rounds=100, metric_name="aucpr", verbose=True):
    ## early stopping
    fit_common = dict(eval_set=[(X_va, y_va)], verbose=verbose)
    try:
        from xgboost.callback import EarlyStopping
        es = EarlyStopping(rounds=rounds, metric_name=metric_name, save_best=True, maximize=True)
        model.fit(X_tr, y_tr, callbacks=[es], **fit_common)
        return
    except Exception:
        pass
    try:
        model.fit(X_tr, y_tr, early_stopping_rounds=rounds, **fit_common)
        return
    except TypeError:
        pass
    model.fit(X_tr, y_tr, **fit_common)

## sampling / data loading
def load_and_prepare_data(seed: int = 42, max_pairs: int = 2400):
    set_global_seed(seed)
    print("loading data")
    diag = pd.read_csv('data/MIMIC-ED/ed/diagnosis.csv')
    vitals = pd.read_csv('data/MIMIC-ED/ed/vitalsign.csv', parse_dates=['charttime'])
    ed = pd.read_csv('data/MIMIC-ED/ed/edstays.csv', parse_dates=['intime', 'outtime'])
    pyx = pd.read_csv('data/MIMIC-ED/ed/pyxis.csv', parse_dates=['charttime'])

    sepsis_mask = diag['icd_title'].astype(str).str.lower().str.contains('sepsis', na=False)
    sepsis_all = np.array(sorted(set(diag.loc[sepsis_mask, 'stay_id'].unique())))
    all_pids = np.array(sorted(ed['stay_id'].unique()))
    non_sepsis_all = np.setdiff1d(all_pids, sepsis_all)

    target = min(len(sepsis_all), len(non_sepsis_all), max_pairs)
    rng = np.random.default_rng(seed)
    sepsis_sample = rng.choice(sepsis_all, target, replace=False).tolist()
    non_sepsis_sample = rng.choice(non_sepsis_all, target, replace=False).tolist()
    selected = sepsis_sample + non_sepsis_sample

    print(f"total patients: {len(all_pids)}")
    print(f"Sepsis patients: {len(sepsis_all)}명")
    print(f"non-Sepsis patients: {len(non_sepsis_all)}명")

    return diag, vitals, ed, pyx, set(sepsis_sample), set(non_sepsis_sample), selected

## diagnosed time 
def calculate_sepsis_confirmation_time(sepsis_pids, pyx):
    sepsis_markers = [
        'cefepime', 'vancomycin',
        'piperacillin', 'piperacillin-tazobactam', 'pip/tazo', 'zosyn',
        'meropenem', 'imipenem', 'doripenem'
    ]
    conf = {}
    pyx_sorted = pyx.sort_values('charttime')
    for pid in sepsis_pids:
        meds = pyx_sorted[pyx_sorted['stay_id'] == pid]
        for _, m in meds.iterrows():
            name = str(m['name']).lower()
            if any(k in name for k in sepsis_markers):
                conf[pid] = m['charttime']
                break
    return conf

HMAX = 6      
BIN_HOURS = 1.0 

def _v(x, d): return x if pd.notna(x) else d

def build_person_period(diag, vitals, ed, pyx, selected_pids, sepsis_pids, seed=42):
    conf = calculate_sepsis_confirmation_time(sepsis_pids, pyx)

    rows = []
    vitals_ = vitals.sort_values(['stay_id', 'charttime'])
    for pid in selected_pids:
        edrow = ed[ed['stay_id'] == pid]
        if edrow.empty:
            continue
        intime = edrow.iloc[0]['intime']
        outtime = edrow.iloc[0]['outtime'] if 'outtime' in edrow.columns else None

        pv = vitals_[vitals_['stay_id'] == pid].reset_index(drop=True)
        if len(pv) < 2:
            continue

        if pid in conf:
            pv = pv[pv['charttime'] <= conf[pid]]

        for i in range(1, len(pv)):
            cur = pv.iloc[i]
            prev = pv.iloc[i-1]
            t = cur['charttime']
            tprev = prev['charttime']
            if pd.isna(t) or pd.isna(tprev):
                continue

            hours_from_adm = (t - intime).total_seconds()/3600.0
            time_gap = max(1e-6, (t - tprev).total_seconds()/3600.0)

            cur_temp=_v(cur['temperature'],37.0); cur_hr=_v(cur['heartrate'],70.0)
            cur_rr=_v(cur['resprate'],16.0); cur_sbp=_v(cur['sbp'],120.0); cur_dbp=_v(cur['dbp'],80.0)
            prev_temp=_v(prev['temperature'],37.0); prev_hr=_v(prev['heartrate'],70.0)
            prev_rr=_v(prev['resprate'],16.0); prev_sbp=_v(prev['sbp'],120.0); prev_dbp=_v(prev['dbp'],80.0)

            cur_sirs = int((cur_temp<36) or (cur_temp>38)) + int(cur_hr>90) + int(cur_rr>20)
            prev_sirs = int((prev_temp<36) or (prev_temp>38)) + int(prev_hr>90) + int(prev_rr>20)

            safe = lambda a,b: (a-b)/b if (b is not None and b!=0) else 0.0
            feat_base = {
                'patient_id': pid,
                'hours_from_admission': hours_from_adm,
                'time_gap': time_gap,
                'vital_check_count': i+1,
                'current_temp': cur_temp, 'current_hr': cur_hr, 'current_rr': cur_rr,
                'current_sbp': cur_sbp, 'current_dbp': cur_dbp, 'current_sirs': cur_sirs,
                'temp_change_rate': safe(cur_temp, prev_temp),
                'hr_change_rate': safe(cur_hr, prev_hr),
                'rr_change_rate': safe(cur_rr, prev_rr),
                'sbp_change_rate': safe(cur_sbp, prev_sbp),
                'dbp_change_rate': safe(cur_dbp, prev_dbp),
                'sirs_change': (cur_sirs - prev_sirs),
            }

            if pid in conf:
                rem = (conf[pid] - t).total_seconds()/3600.0
                if rem <= 0:
                    continue
                last_bin = int(min(HMAX, np.ceil(rem/BIN_HOURS)))
                event_bin = int(np.ceil(rem/BIN_HOURS))
                for k in range(1, last_bin+1):
                    y = int(k == event_bin and rem <= HMAX*BIN_HOURS)
                    rows.append({**feat_base, 't_bin': k, 'event': y})
            else:
                if pd.isna(outtime):
                    continue
                rem = (outtime - t).total_seconds()/3600.0
                if rem <= 0:
                    continue
                last_bin = int(min(HMAX, np.floor(rem/BIN_HOURS)))
                for k in range(1, last_bin+1):
                    rows.append({**feat_base, 't_bin': k, 'event': 0})

    df = pd.DataFrame(rows)
    print(f"person-period samples: {len(df)}  (event rate={df['event'].mean():.4f})")
    return df

def _make_gpu_xgb_kwargs():
    ver = tuple(int(x) for x in xgb.__version__.split('.')[:2])
    if ver >= (2, 0):
        return dict(tree_method="hist", device="cuda")
    else:
        return dict(tree_method="gpu_hist", gpu_id=0)

def train_discrete_survival_gpu(df, out_dir="outputs_discrete_survival", seed=42):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    feats = [
        't_bin',
        'hours_from_admission','time_gap','vital_check_count',
        'current_temp','current_hr','current_rr','current_sbp','current_dbp','current_sirs',
        'temp_change_rate','hr_change_rate','rr_change_rate','sbp_change_rate','dbp_change_rate','sirs_change'
    ]
    X = df[feats]; y = df['event']; groups = df['patient_id']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(gss.split(X, y, groups))
    X_tr, X_te = X.iloc[tr], X.iloc[te]; y_tr, y_te = y.iloc[tr], y.iloc[te]

    spw = max(1, int((y_tr==0).sum())) / max(1, int(y_tr.sum()))

    gpu_kwargs = _make_gpu_xgb_kwargs()
    mdl = xgb.XGBClassifier(
        n_estimators=1200, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        scale_pos_weight=spw, eval_metric="aucpr", verbosity=1,
        **gpu_kwargs
    )

    fit_with_early_stopping(mdl, X_tr, y_tr, X_te, y_te, rounds=100, metric_name="aucpr", verbose=True)

    p_tr = mdl.predict_proba(X_tr)[:,1]; p_te = mdl.predict_proba(X_te)[:,1]
    metrics = {
        'train_ap': float(average_precision_score(y_tr, p_tr)),
        'test_ap': float(average_precision_score(y_te, p_te)),
        'train_auc': float(roc_auc_score(y_tr, p_tr)),
        'test_auc': float(roc_auc_score(y_te, p_te)),
        'n_test_pos': int(y_te.sum()),
        'n_test_neg': int((y_te==0).sum()),
        'feature_importance': {k: float(v) for k,v in zip(feats, mdl.feature_importances_)}
    }

    fpr, tpr, _ = roc_curve(y_te, p_te)
    prec, rec, _ = precision_recall_curve(y_te, p_te)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={metrics['test_auc']:.3f}")
    plt.plot([0,1], [0,1], '--', linewidth=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC — Discrete Survival (test)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(Path(out_dir) / 'roc_discrete_survival.png', dpi=160, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, label=f"AP={metrics['test_ap']:.3f}")
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('PR — Discrete Survival (test)')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(Path(out_dir) / 'pr_discrete_survival.png', dpi=160, bbox_inches='tight')
    plt.close()

    mdl.save_model(str(Path(out_dir) / 'surv_discrete_gpu.json'))
    with open(Path(out_dir) / 'surv_discrete_feats.pkl', 'wb') as f:
        pickle.dump(feats, f)
    with open(Path(out_dir) / 'surv_discrete_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return mdl, feats, metrics

def predict_event_prob_within_K(mdl, feats, snapshot_df, K=6):
    out = []
    base_cols = [c for c in feats if c!='t_bin']
    need_cols = [c for c in base_cols if c in snapshot_df.columns]
    assert len(need_cols) == len(base_cols)

    for pid, g in snapshot_df.groupby('patient_id'):
        G = pd.concat([g.assign(t_bin=k) for k in range(1, K+1)], ignore_index=True)
        P = mdl.predict_proba(G[['t_bin'] + base_cols])[:,1]
        if len(P) > K:
            P = P.reshape(-1, K).mean(axis=0)
        surv = float(np.prod(1.0 - P))
        out.append({'patient_id': pid, f'prob_event_within_{K}h': 1.0 - surv})
    return pd.DataFrame(out)


def main(seed=42, out_dir="outputs_discrete_survival"):
    set_global_seed(seed)
    print("=" * 60)
    print("started")

    diag, vitals, ed, pyx, sepsis_pids, non_sepsis_pids, selected = load_and_prepare_data(seed=seed)
    df_pp = build_person_period(diag, vitals, ed, pyx, selected, sepsis_pids, seed=seed)
    mdl, feats, metrics = train_discrete_survival_gpu(df_pp, out_dir=out_dir, seed=seed)

    print("completed")
    print(f"saved to: {out_dir}")

if __name__ == "__main__":
    main()
