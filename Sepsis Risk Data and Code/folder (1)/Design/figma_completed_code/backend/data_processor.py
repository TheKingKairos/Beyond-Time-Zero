"""
Simulated data pipeline for generating Patient Sepsis dashboard metrics.

This module fabricates vitals and triage dataframes and produces a JSON payload
that mirrors what an ML-driven backend might emit. In a production setting the
random score generation would be replaced with calls to persisted models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VitalColumns:
    """Helper container listing the vital sign columns we expose downstream."""

    fields: tuple[str, ...] = (
        "temperature",
        "heart_rate",
        "respiratory_rate",
        "systolic_bp",
        "diastolic_bp",
        "spo2",
    )


def _build_mock_vitalsign_dataframe() -> pd.DataFrame:
    """Construct mock vital sign readings for at least ten unique stays."""
    base_time = pd.Timestamp("2024-03-01T08:00:00")
    vitalsign_records: List[Dict[str, Any]] = []

    for idx, stay_id in enumerate(range(1001, 1011)):
        subject_id = 500 + idx // 2
        for reading in (0, 1):
            charttime = base_time + pd.Timedelta(hours=idx * 4 + reading * 2)
            vitalsign_records.append(
                {
                    "subject_id": subject_id,
                    "stay_id": stay_id,
                    "charttime": charttime,
                    "temperature": round(36.5 + 0.2 * idx + 0.1 * reading, 1),
                    "heart_rate": 78 + idx * 2 + reading * 3,
                    "respiratory_rate": 16 + (idx % 4) + reading,
                    "systolic_bp": 108 + idx * 2,
                    "diastolic_bp": 68 + (idx % 6),
                    "spo2": 96 - (idx % 3),
                }
            )

    df_vitalsign = pd.DataFrame(vitalsign_records)
    df_vitalsign["charttime"] = pd.to_datetime(df_vitalsign["charttime"])
    return df_vitalsign


def _build_mock_triage_dataframe() -> pd.DataFrame:
    """Construct mock triage table aligned with the vitals data."""
    base_time = pd.Timestamp("2024-02-29T21:30:00")
    admission_types = ["ED", "Elective", "Urgent"]
    chief_complaints = [
        "Shortness of breath",
        "Fever and chills",
        "Abdominal pain",
        "Chest discomfort",
        "Altered mental status",
    ]

    triage_records: List[Dict[str, Any]] = []
    for idx, stay_id in enumerate(range(1001, 1011)):
        subject_id = 500 + idx // 2
        triage_records.append(
            {
                "subject_id": subject_id,
                "stay_id": stay_id,
                "triage_time": base_time + pd.Timedelta(hours=idx * 3),
                "admission_type": admission_types[idx % len(admission_types)],
                "acuity_level": ["Emergent", "Urgent", "Non-Urgent"][idx % 3],
                "chief_complaint": chief_complaints[idx % len(chief_complaints)],
            }
        )

    df_triage = pd.DataFrame(triage_records)
    df_triage["triage_time"] = pd.to_datetime(df_triage["triage_time"])
    return df_triage


df_vitalsign = _build_mock_vitalsign_dataframe()
df_triage = _build_mock_triage_dataframe()


def _numpy_to_native(value: Any) -> Any:
    """Convert NumPy scalar values to native Python types for JSON serialization."""
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def generate_final_dashboard_data(seed: Optional[int] = None) -> str:
    """
    Generate a ranked dashboard feed as a JSON string.

    Args:
        seed: When provided, ensures deterministic random score generation.

    Returns:
        A JSON-formatted string sorted by hazard rate that includes each patient's
        vitals, sepsis score, hazard rate, and derived priority rank.
    """
    rng = np.random.default_rng(seed)
    vital_fields = VitalColumns().fields

    latest_vitals = (
        df_vitalsign.sort_values("charttime")
        .groupby("stay_id", as_index=False)
        .tail(1)
    )
    latest_vitals = latest_vitals.merge(
        df_triage[["stay_id", "triage_time"]],
        on="stay_id",
        how="left",
    )

    dashboard_rows: List[Dict[str, Any]] = []
    for _, row in latest_vitals.iterrows():
        sepsis_score = int(rng.integers(65, 100))
        hazard_rate = float(np.round(rng.uniform(0.600, 0.999), 3))
        vitals_payload = {
            field: _numpy_to_native(row[field]) for field in vital_fields
        }

        dashboard_rows.append(
            {
                "subject_id": int(row["subject_id"]),
                "stay_id": int(row["stay_id"]),
                "sepsisScore": sepsis_score,
                "hazardRate": hazard_rate,
                "lastVitalTime": row["charttime"].isoformat(),
                "vitals": vitals_payload,
            }
        )

    dashboard_rows.sort(key=lambda record: record["hazardRate"], reverse=True)
    for rank, payload in enumerate(dashboard_rows, start=1):
        payload["priorityRank"] = rank

    return json.dumps(dashboard_rows, indent=2)


if __name__ == "__main__":
    print(generate_final_dashboard_data(seed=42))
