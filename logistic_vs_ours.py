# eval_logreg_vs_ensemble_full.py

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

import torch
import torch.nn as nn
import xgboost as xgb
from typing import List, Tuple

# ---------------------------
# MLP definition (from your dashboard code)
# ---------------------------
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


# ---------------------------
# Config for ensemble model
# ---------------------------
REQUIRED_COLS = [
    "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain",
    "rhythm_flag", "is_white", "is_black", "is_asian", "is_hispanic",
    "is_other_race", "gender_F", "gender_M",
    "arrival_transport_AMBULANCE", "arrival_transport_HELICOPTER",
    "arrival_transport_OTHER", "arrival_transport_UNKNOWN",
    "arrival_transport_WALK IN",
    "lactate", "wbc", "time_since_adm",
    "gsn_16599.0", "gsn_43952.0", "gsn_4490.0", "gsn_66419.0", "gsn_61716.0",
]
INPUT_DIM = len(REQUIRED_COLS)


def load_ensemble_models():
    """Load MLP + XGB ensemble using your existing training artifacts."""
    # Load best MLP config
    with open("grid_runs/top5_results.json", "r") as f:
        top5 = json.load(f)
    best_cfg = top5[0]["config"]

    mlp_model = MLP(
        in_features=INPUT_DIM,
        hidden_layers=tuple(best_cfg["layers"]),
        activation=best_cfg["activation"],
        dropout=best_cfg["dropout"],
        use_batchnorm=best_cfg["batchnorm"],
    )

    state_dict = torch.load("grid_runs/best_model.pt", map_location="cpu")
    mlp_model.load_state_dict(state_dict)
    mlp_model.eval()

    # Load XGBoost model
    xgb_model = xgb.XGBClassifier(device="cpu")
    xgb_model.load_model("grid_runs_xgb/best_xgb.bin")

    return mlp_model, xgb_model


def summarize_at_threshold(y_true, probs, thr):
    preds = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    sens = tp / (tp + fn + 1e-9)  # recall
    spec = tn / (tn + fp + 1e-9)
    ppv = tp / (tp + fp + 1e-9)   # precision
    npv = tn / (tn + fn + 1e-9)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return dict(
        threshold=thr,
        sensitivity=sens,
        specificity=spec,
        ppv=ppv,
        npv=npv,
        accuracy=acc,
    )


def main():
    # ---------------------------
    # 1. Load data
    # ---------------------------
    df = pd.read_csv("data/MIMIC-ED/event_level_training_data.csv")

    # True labels
    y = df["is_sepsis"].to_numpy(dtype=np.int64)

    # Features for logistic regression (whatever it was trained on)
    X_lr = df.drop(columns=["is_sepsis", "stay_id"]).to_numpy(dtype=np.float32)

    # Features for ensemble model (REQUIRED_COLS)
    X_ens_df = df[REQUIRED_COLS]
    X_ens = X_ens_df.to_numpy(dtype=np.float32)

    # ---------------------------
    # 2. Load models
    # ---------------------------
    logreg_model = joblib.load("best_logistic_regression_model.joblib")
    mlp_model, xgb_model = load_ensemble_models()

    # ---------------------------
    # 3. Logistic regression predictions
    # ---------------------------
    lr_probs = logreg_model.predict_proba(X_lr)[:, 1]
    lr_preds = (lr_probs >= 0.5).astype(np.int64)

    lr_auc = roc_auc_score(y, lr_probs)
    lr_acc = accuracy_score(y, lr_preds)

    # ---------------------------
    # 4. Ensemble predictions (MLP + XGB avg)
    # ---------------------------
    with torch.no_grad():
        X_tensor = torch.tensor(X_ens, dtype=torch.float32)
        logits = mlp_model(X_tensor)
        mlp_probs = torch.sigmoid(logits).squeeze().numpy()

    xgb_probs = xgb_model.predict_proba(X_ens_df)[:, 1]
    ens_probs = (mlp_probs + xgb_probs) / 2.0
    ens_preds = (ens_probs >= 0.5).astype(np.int64)

    ens_auc = roc_auc_score(y, ens_probs)
    ens_acc = accuracy_score(y, ens_preds)

    # ==========================================================
    # (1) ROC + AUC + Accuracy (with PNG)
    # ==========================================================
    fpr_lr, tpr_lr, _ = roc_curve(y, lr_probs)
    fpr_ens, tpr_ens, _ = roc_curve(y, ens_probs)

    plt.figure()
    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {lr_auc:.3f})")
    plt.plot(fpr_ens, tpr_ens, label=f"Ensemble MLP+XGB (AUC = {ens_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Logistic vs Ensemble")
    plt.legend(loc="lower right")
    plt.tight_layout()

    roc_path = Path("logreg_vs_ensemble_roc.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # ==========================================================
    # (2) Precision–Recall Curves + AUPRC
    # ==========================================================
    prec_lr, rec_lr, _ = precision_recall_curve(y, lr_probs)
    prec_ens, rec_ens, _ = precision_recall_curve(y, ens_probs)

    auprc_lr = average_precision_score(y, lr_probs)
    auprc_ens = average_precision_score(y, ens_probs)

    plt.figure()
    plt.plot(rec_lr, prec_lr, label=f"LogReg (AUPRC = {auprc_lr:.3f})")
    plt.plot(rec_ens, prec_ens, label=f"Ensemble (AUPRC = {auprc_ens:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve – Logistic vs Ensemble")
    plt.legend(loc="lower left")
    plt.tight_layout()

    pr_path = Path("logreg_vs_ensemble_pr.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()

    # ==========================================================
    # (3) Calibration: Reliability Curves + Brier Score
    # ==========================================================
    brier_lr = brier_score_loss(y, lr_probs)
    brier_ens = brier_score_loss(y, ens_probs)

    frac_pos_lr, mean_pred_lr = calibration_curve(
        y, lr_probs, n_bins=10, strategy="quantile"
    )
    frac_pos_ens, mean_pred_ens = calibration_curve(
        y, ens_probs, n_bins=10, strategy="quantile"
    )

    plt.figure()
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.plot(mean_pred_lr, frac_pos_lr, marker="o", label="LogReg")
    plt.plot(mean_pred_ens, frac_pos_ens, marker="o", label="Ensemble")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("Calibration – Logistic vs Ensemble")
    plt.legend(loc="upper left")
    plt.tight_layout()

    calib_path = Path("logreg_vs_ensemble_calibration.png")
    plt.savefig(calib_path, dpi=300)
    plt.close()

    # ==========================================================
    # Threshold-based metrics table (sens/spec/PPV/NPV/acc)
    # ==========================================================
    thresholds = [0.1, 0.2, 0.3, 0.5]

    print("==================================================")
    print("Performance on Full Data")
    print("--------------------------------------------------")
    print(f"LogReg   – AUROC = {lr_auc:.4f}, Accuracy = {lr_acc:.4f}, "
          f"AUPRC = {auprc_lr:.4f}, Brier = {brier_lr:.4f}")
    print(f"Ensemble – AUROC = {ens_auc:.4f}, Accuracy = {ens_acc:.4f}, "
          f"AUPRC = {auprc_ens:.4f}, Brier = {brier_ens:.4f}")

    print("\nThreshold-based comparison (sens/spec/PPV/NPV/acc)")
    for thr in thresholds:
        lr_stats = summarize_at_threshold(y, lr_probs, thr)
        ens_stats = summarize_at_threshold(y, ens_probs, thr)
        print(f"\nThreshold = {thr:.2f}")
        print(
            "  LogReg  : "
            f"sens={lr_stats['sensitivity']:.3f}, "
            f"spec={lr_stats['specificity']:.3f}, "
            f"PPV={lr_stats['ppv']:.3f}, "
            f"NPV={lr_stats['npv']:.3f}, "
            f"acc={lr_stats['accuracy']:.3f}"
        )
        print(
            "  Ensemble: "
            f"sens={ens_stats['sensitivity']:.3f}, "
            f"spec={ens_stats['specificity']:.3f}, "
            f"PPV={ens_stats['ppv']:.3f}, "
            f"NPV={ens_stats['npv']:.3f}, "
            f"acc={ens_stats['accuracy']:.3f}"
        )

    print("\nSaved plots:")
    print(f"  ROC curve:         {roc_path}")
    print(f"  PR curve:          {pr_path}")
    print(f"  Calibration curve: {calib_path}")


if __name__ == "__main__":
    main()
