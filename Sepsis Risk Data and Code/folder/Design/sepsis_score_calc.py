import numpy as np
import pandas as pd
import json
import joblib

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

import os, json
from pathlib import Path
from datetime import datetime

import pickle, pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter

import xgboost as xgb

class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_layers: Tuple[int, ...],
        activation: str = "relu",
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        acts = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        if activation not in acts:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: List[nn.Module] = []
        last = in_features
        for h in hidden_layers:
            layers.append(nn.Linear(last, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(acts[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.out = nn.Linear(last, 1)

        # Kaiming init for ReLU-like, Xavier otherwise
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if activation in ("relu", "silu"):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.backbone(x)
        logits = self.out(z)  # [B, 1]
        return logits
    
required_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'rhythm_flag', 'is_white', 'is_black', 'is_asian', 'is_hispanic', 'is_other_race', 'gender_F', 'gender_M', 'arrival_transport_AMBULANCE', 'arrival_transport_HELICOPTER', 'arrival_transport_OTHER', 'arrival_transport_UNKNOWN', 'arrival_transport_WALK IN', 'lactate', 'wbc', 'time_since_adm', 'gsn_16599.0', 'gsn_43952.0', 'gsn_4490.0', 'gsn_66419.0', 'gsn_61716.0']
your_input_dim = len(required_cols)

with open("grid_runs/top5_results.json", "r") as f:
    top5 = json.load(f)

best_cfg = top5[0]["config"]

mlp_model = MLP(
    in_features=your_input_dim,
    hidden_layers=tuple(best_cfg["layers"]),
    activation=best_cfg["activation"],
    dropout=best_cfg["dropout"],
    use_batchnorm=best_cfg["batchnorm"]
)

state_dict = torch.load("grid_runs/best_model.pt", map_location="cpu")
mlp_model.load_state_dict(state_dict)
mlp_model.eval()

# Load CPU model
xgbmodel = xgb.XGBClassifier(device='cpu')  # works for XGBoost ≥ 2.0
xgbmodel.load_model("grid_runs_xgb/best_xgb.bin")

df = pd.read_json('folder/Design/src/mockData.json', lines=False)

mod_df = df.rename(columns={"temp": "temperature", "hr": "heartrate"})
required_cols = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'rhythm_flag', 'is_white', 'is_black', 'is_asian', 'is_hispanic', 'is_other_race', 'gender_F', 'gender_M', 'arrival_transport_AMBULANCE', 'arrival_transport_HELICOPTER', 'arrival_transport_OTHER', 'arrival_transport_UNKNOWN', 'arrival_transport_WALK IN', 'lactate', 'wbc', 'time_since_adm', 'gsn_16599.0', 'gsn_43952.0', 'gsn_4490.0', 'gsn_66419.0', 'gsn_61716.0']
X = mod_df[required_cols]
time_since_check = df["lastVitalTime"].values/60

with torch.no_grad():
    X_test = torch.tensor(X.to_numpy(), dtype=torch.float32)
    logits = mlp_model(X_test)
    probs = torch.sigmoid(logits)

probs = probs.squeeze().numpy()
probs = probs.round(3) * 100

xgb_probs = xgbmodel.predict_proba(X)[:, 1]
xgb_probs = xgb_probs.round(3) * 100

sepsis_scores = np.add(probs, xgb_probs) / 2
sepsis_scores = sepsis_scores.round(3)

df['sepsisScore'] = sepsis_scores

# Load model
with open("cox_models/coxph_static.pkl", "rb") as f:
    cph: CoxPHFitter = pickle.load(f)

# Load scaler + feature list
with open("cox_models/coxph_static_scaler.pkl", "rb") as f:
    payload = pickle.load(f)
scaler = payload["scaler"]
feats = payload["features"]

with open("cox_models/cox_tvc.pkl", "rb") as f:
    ctv: CoxTimeVaryingFitter = pickle.load(f)

payload_tvc = pickle.load(open("cox_models/cox_tvc_scaler.pkl", "rb"))

# # Prepare new data frame `df_new` with the same columns
# cph_X = X.copy()
# cph_X.loc[:, feats] = scaler.transform(cph_X[feats])

# # Predict (example)
# risk = cph.predict_partial_hazard(cph_X)

def prepare_static_inference_frame(raw_df: pd.DataFrame, scaler_payload: dict) -> pd.DataFrame:
    """
    Returns a *new* DataFrame where feature columns are scaled (float64) and
    non-feature columns are preserved. No in-place assignment into raw_df.
    """
    feats  = scaler_payload["features"]
    scaler = scaler_payload["scaler"]

    # ensure float for transform input
    X = scaler.transform(raw_df[feats].to_numpy(dtype="float64"))

    X_df = pd.DataFrame(X, index=raw_df.index, columns=feats)
    non_feats = raw_df.drop(columns=feats)

    return pd.concat([non_feats, X_df], axis=1)

def prepare_tvc_inference_frame(raw_df: pd.DataFrame, scaler_payload: dict) -> pd.DataFrame:
    feats  = scaler_payload["features"]
    scaler = scaler_payload["scaler"]

    X = scaler.transform(raw_df[feats].to_numpy(dtype="float64"))
    X_df = pd.DataFrame(X, index=raw_df.index, columns=feats)
    non_feats = raw_df.drop(columns=feats)

    return pd.concat([non_feats, X_df], axis=1)


df_pred_cox = prepare_static_inference_frame(X, payload)
df_pred_tvc = prepare_tvc_inference_frame(X, payload_tvc)

scores_cox_matrix = cph.predict_cumulative_hazard(df_pred_cox, times=time_since_check)

# create dict for index and time
time_since_check_dict = dict(zip(time_since_check.astype("float64").tolist(), range(df.shape[0])))
indices = list(time_since_check_dict.items())
print("Indices:", indices)


scores_cox = np.array([scores_cox_matrix.loc[i] for i in indices])

df["hazardRate"] = scores_cox

# get priority rank based on scores
df["priorityRank"] = df["hazardRate"].rank(ascending=False).astype(int)

# sort location by priority rank
df = df.sort_values(by="priorityRank")

for i in range(len(df)):
    df["trends"][i][-1]["temp"] = df.loc[i, "temp"]
    df["trends"][i][-1]["heartRate"] = df.loc[i, "hr"]
    df["trends"][i][-1]["lactate"] = df.loc[i, "lactate"]

df.to_json("folder/Design/src/mockData.json", orient="records", lines=False, indent=1)