# `dashboard_update.py`

`dashboard_update.py` is a standalone utility script that updates a dashboard JSON file by attaching all required sepsis-prediction features and computing a refreshed **sepsisScore** using a saved **logistic regression model**.

This script is intentionally self-contained:  
- Loads the dashboard JSON
- Loads the feature dataset
- Aligns/creates missing columns
- Loads the trained model
- Generates sepsis scores
- Writes out an updated JSON for the UI

---

## What the Script Does

### **1. Load Dashboard JSON**
- Reads a list of records from `mockData.json`.
- Validates that every entry contains `patientId`.

### **2. Load Feature Dataset**
- Reads a cleaned dataset (`.csv` or `.parquet`) containing features for each patient/stay.
- Expects the dataset to contain `patientId` (automatically mapped from `stay_id` if present).

### **3. Normalize Column Names**
The script maps common alternate names → model feature names via `ALIAS_MAP`, e.g.:

| Raw column | Model feature |
|------------|--------------|
| `temp` | `temperature` |
| `hr`   | `heartrate` |
| `spo2` | `o2sat` |
| `white` | `is_white` |

This ensures the model always gets the exact expected feature names.

### **4. Handle Arrival Transport Encoding**
If the dataset stores transport as a single categorical column, the script automatically one-hots:

