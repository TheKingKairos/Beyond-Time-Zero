import pandas as pd
import numpy as np
import re

# -------------------------------
# Load CSVs
# -------------------------------
diagnosis = pd.read_csv("data/MIMIC-ED/ed/diagnosis.csv")
edstays   = pd.read_csv("data/MIMIC-ED/ed/edstays.csv", parse_dates=["intime","outtime"])
triage    = pd.read_csv("data/MIMIC-ED/ed/triage.csv")
vitalsign = pd.read_csv("data/MIMIC-ED/ed/vitalsign.csv", parse_dates=["charttime"])
pyxis     = pd.read_csv("data/MIMIC-ED/ed/pyxis.csv", parse_dates=["charttime"])
medrecon  = pd.read_csv("data/MIMIC-ED/ed/medrecon.csv", parse_dates=["charttime"])

# -------------------------------
# Clean column names
# -------------------------------
def clean(df):
    df.columns = df.columns.str.strip().str.lower()
    return df

diagnosis, edstays, triage, vitalsign, pyxis, medrecon = map(
    clean, [diagnosis, edstays, triage, vitalsign, pyxis, medrecon]
)

# -------------------------------
# 1) Mark sepsis by ICD (stay-level)
# -------------------------------
def mark_sepsis(icd_code):
    if pd.isna(icd_code):
        return 0
    code = str(icd_code).replace(".", "").upper()
    # ICD-9
    if code.startswith("038") or code in {"99591", "99592", "78552"}:
        return 1
    # ICD-10
    if code.startswith("A40") or code.startswith("A41") or code in {"R652", "R572"}:
        return 1
    return 0

diagnosis["sepsis_dx_raw"] = diagnosis["icd_code"].apply(mark_sepsis)

sepsis_flag = (
    diagnosis.groupby(["subject_id", "stay_id"])["sepsis_dx_raw"]
    .max()
    .reset_index()
    .rename(columns={"sepsis_dx_raw": "sepsis_dx_any"})
)

# Aggregate ICD titles (optional info)
diag_agg = (
    diagnosis.groupby(["subject_id", "stay_id"])
    .agg({"icd_title": lambda x: "; ".join(sorted({str(v) for v in x if pd.notna(v)}))})
    .reset_index()
)

# -------------------------------
# 2) Rhythm → binary flag
# -------------------------------
vitalsign["rhythm_flag"] = vitalsign["rhythm"].notna().astype(int)

# -------------------------------
# 3) First antibiotic time (proxy for dx time)
# -------------------------------
ABX_MARKERS = [
    "cefepime", "vancomycin",
    "piperacillin", "piperacillin-tazobactam", "pip/tazo", "zosyn",
    "meropenem", "imipenem", "doripenem",
]
abx_pattern = "|".join(re.escape(x) for x in ABX_MARKERS)

if "name" not in pyxis.columns:
    pyxis["name"] = np.nan
pyxis["name_lower"] = pyxis["name"].astype(str).str.lower()
pyxis["is_antibiotic"] = pyxis["name_lower"].str.contains(abx_pattern, case=False, regex=True, na=False)

first_abx_time = (
    pyxis.loc[pyxis["is_antibiotic"]]
    .groupby(["subject_id", "stay_id"])["charttime"]
    .min()
    .reset_index()
    .rename(columns={"charttime": "first_abx_time"})
)

# -------------------------------
# 4) Stay-level dataset (demographics + labels)
# -------------------------------
stay_level = edstays.merge(triage, on=["subject_id", "stay_id"], how="left")
stay_level = stay_level.merge(diag_agg, on=["subject_id", "stay_id"], how="left")
stay_level = stay_level.merge(sepsis_flag, on=["subject_id", "stay_id"], how="left")
stay_level = stay_level.merge(first_abx_time, on=["subject_id", "stay_id"], how="left")

# (If age is missing but dob/anchor_year etc. exist, compute here; otherwise leave as-is.)

stay_level.to_csv("data/MIMIC-ED/stay_level_v2.csv", index=False)
print("✅ stay_level.csv created (includes demographics + sepsis_dx_any + first_abx_time)")

# -------------------------------
# 5) Event-level dataset w/ demographics and time-varying label
# -------------------------------
# Tag sources
vitalsign["event_type"] = "vitalsign"
pyxis["event_type"]     = "pyxis"
medrecon["event_type"]  = "medrecon"

# Combine events and ensure datetime
event_level = pd.concat([vitalsign, pyxis, medrecon], ignore_index=True, sort=False)
event_level["charttime"] = pd.to_datetime(event_level["charttime"], errors="coerce")

# Merge ALL stay-level info (adds race/age/etc.)
event_level = event_level.merge(
    stay_level, on=["subject_id", "stay_id"], how="left", suffixes=("", "_stay")
)

# Compute time-varying sepsis label:
#   1) stay must be ICD-confirmed (sepsis_dx_any == 1)
#   2) flips to 1 at/after first_abx_time; otherwise 0
event_level["sepsis_dx"] = np.where(
    (event_level["sepsis_dx_any"] == 1)
    & event_level["first_abx_time"].notna()
    & (event_level["charttime"] >= event_level["first_abx_time"]),
    1, 0
).astype("int8")

# ====================================================
# ONE-HOT ENCODE CATEGORICAL VARIABLES
# ====================================================
# ====================================================
# ONE-HOT ENCODE CATEGORICAL VARIABLES
# ====================================================
CATEGORICAL_COLS = ["gender", "arrival_transport"]

for col in CATEGORICAL_COLS:
    if col in event_level.columns:
        event_level[col] = event_level[col].astype("category")

# calculate race separately to combine infrequent categories
event_level["is_white"] = np.where(event_level["race"].str.contains("white", case=False, na=False), 1, 0)
event_level["is_black"] = np.where(event_level["race"].str.contains("black", case=False, na=False), 1, 0)
event_level["is_asian"] = np.where(event_level["race"].str.contains("asian", case=False, na=False), 1, 0)
event_level["is_hispanic"] = np.where(event_level["race"].str.contains("hispanic", case=False, na=False), 1, 0)
event_level["is_other_race"] = np.where(event_level["is_white"] + event_level["is_black"] + event_level["is_asian"] + event_level["is_hispanic"] == 0, 1, 0)

event_level = pd.get_dummies(event_level, columns=[c for c in CATEGORICAL_COLS if c in event_level.columns])

# replace nans in the the following rows with 0
for col in ["rhythm_flag", "sepsis_dx", "sepsis_dx_any"]:
    if col in event_level.columns:
        event_level[col] = event_level[col].fillna(0)

# Optional: cast some frequent numeric flags to small ints
for col in ["rhythm_flag", "sepsis_dx", "sepsis_dx_any"]:
    if col in event_level.columns:
        event_level[col] = event_level[col].astype("int8")

# -------------------------------
# 7) Save outputs
# -------------------------------
event_level.to_csv("data/MIMIC-ED/event_level_v2.csv", index=False)
print("✅ event_level.csv created (merged demographics + one-hot categoricals + time-varying sepsis_dx)")
