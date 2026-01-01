# clinicalbert_encode.py

import os, json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np

CSV_PATH = "event_level_with_lac_wbc.csv"  # your existing file
TEXT_COLS = ["chiefcomplaint"]

# --- Load data
df = pd.read_csv(CSV_PATH)
col = next((c for c in TEXT_COLS if c in df.columns), None)
if col is None:
    raise ValueError(f"None of {TEXT_COLS} found in CSV columns: {df.columns.tolist()}")

texts = df[col].fillna("").astype(str).str.strip().tolist()

# --- Model / tokenizer
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- Helper: mean-pool last hidden state using attention mask
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    return masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

# --- Encode in batches
batch_size = 64
max_length = 64
all_embeds = []

with torch.no_grad():
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        emb = mean_pool(outputs.last_hidden_state, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embeds.append(emb.cpu())

embeddings = torch.vstack(all_embeds).numpy()

# --- Attach embeddings to original dataframe
df["clinicalbert_emb"] = [e.tolist() for e in embeddings]

# --- Save it BACK to the same CSV (overwrites original)
df.to_csv(CSV_PATH, index=False)

print(f"Updated original CSV with embeddings -> {CSV_PATH}")