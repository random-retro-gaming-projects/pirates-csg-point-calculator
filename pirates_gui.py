#!/usr/bin/env python3
# ship_predict_gui.py

import tkinter as tk
from tkinter import ttk
import joblib
import pandas as pd
import numpy as np
import re

import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------
# 1) Define the same parsing logic you used during training
# ---------------------------------------------------------
def parse_base_move(move_str):
    if not move_str:
        return np.nan
    parts = re.split(r'\+|\s+', move_str.strip())
    total = 0
    for p in parts:
        p = p.upper()
        if p == 'S':
            total += 1
        elif p == 'L':
            total += 2
    return total

def parse_cannons(cannon_str):
    if not cannon_str:
        return (0, np.nan, 0.0)
    parts = re.split(r'[,\s]+', cannon_str.strip())
    ranks = []
    l_count = 0
    for part in parts:
        part = part.upper().strip()
        match = re.match(r'(\d+)([SL])', part)
        if match:
            rank_val = int(match.group(1))
            range_str = match.group(2)
            ranks.append(rank_val)
            if range_str == 'L':
                l_count += 1
    if len(ranks) > 0:
        avg_rank = np.mean(ranks)
        frac_L = l_count / len(ranks)
    else:
        avg_rank = np.nan
        frac_L = 0.0
    return (len(ranks), avg_rank, frac_L)

# ---------------------------------------------------------
# 2) Load your pre-trained pipeline
# ---------------------------------------------------------

if hasattr(sys, '_MEIPASS'):
    # Running in PyInstaller bundle
    MODEL_PATH = os.path.join(sys._MEIPASS, "pirates_point_model.pkl")
else:
    # Running in normal Python environment
    MODEL_PATH = "pirates_point_model.pkl"

model_pipeline = joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# 3) Build the Tkinter GUI
# ---------------------------------------------------------
root = tk.Tk()
root.title("Pirates CSG Ship Point Cost Predictor")

# We'll create labels and entry boxes for each parameter
label_masts = ttk.Label(root, text="Masts:")
label_masts.grid(row=0, column=0, padx=5, pady=5, sticky=tk.E)
entry_masts = ttk.Entry(root)
entry_masts.grid(row=0, column=1, padx=5, pady=5)

label_cargo = ttk.Label(root, text="Cargo:")
label_cargo.grid(row=1, column=0, padx=5, pady=5, sticky=tk.E)
entry_cargo = ttk.Entry(root)
entry_cargo.grid(row=1, column=1, padx=5, pady=5)

label_base_move = ttk.Label(root, text="Base Move (e.g. 'S+L'):")
label_base_move.grid(row=2, column=0, padx=5, pady=5, sticky=tk.E)
entry_base_move = ttk.Entry(root)
entry_base_move.grid(row=2, column=1, padx=5, pady=5)

label_cannons = ttk.Label(root, text="Cannons (e.g. '3S,4L'):")
label_cannons.grid(row=3, column=0, padx=5, pady=5, sticky=tk.E)
entry_cannons = ttk.Entry(root)
entry_cannons.grid(row=3, column=1, padx=5, pady=5)

label_faction = ttk.Label(root, text="Faction:")
label_faction.grid(row=4, column=0, padx=5, pady=5, sticky=tk.E)
entry_faction = ttk.Entry(root)
entry_faction.grid(row=4, column=1, padx=5, pady=5)

label_abilities = ttk.Label(root, text="Abilities:")
label_abilities.grid(row=5, column=0, padx=5, pady=5, sticky=tk.E)
entry_abilities = ttk.Entry(root)
entry_abilities.grid(row=5, column=1, padx=5, pady=5)

# We'll show the result in a label
label_result = ttk.Label(root, text="Predicted Point Cost: N/A")
label_result.grid(row=7, column=0, columnspan=2, pady=10)

# ---------------------------------------------------------
# 4) Define the predict function for the 'Predict' button
# ---------------------------------------------------------
def predict():
    # Get values from entry fields
    try:
        masts = int(entry_masts.get().strip())
    except:
        masts = 0
    try:
        cargo = int(entry_cargo.get().strip())
    except:
        cargo = 0

    base_move_str = entry_base_move.get().strip()
    cannons_str   = entry_cannons.get().strip()
    faction_str   = entry_faction.get().strip()
    abilities_str = entry_abilities.get().strip()

    # Parse base_move/cannons the same way as training
    bm_val = parse_base_move(base_move_str)
    num_cannons, avg_rank, frac_L = parse_cannons(cannons_str)

    # Create a single-row DataFrame for the pipeline
    row = {
        "Masts": masts,
        "Cargo": cargo,
        "BaseMoveValue": bm_val,
        "NumCannons": num_cannons,
        "AvgCannonRank": avg_rank,
        "FracL": frac_L,
        "Faction": faction_str,
        "Ability": abilities_str  # Adjust if your pipeline expects "Ability" vs. "Abilities"
    }
    new_df = pd.DataFrame([row])

    # Predict
    predicted_cost = model_pipeline.predict(new_df)[0]

    # Update the label
    label_result.config(text=f"Predicted Point Cost: {predicted_cost:.2f}")

# ---------------------------------------------------------
# 5) Add a button to trigger prediction
# ---------------------------------------------------------
btn_predict = ttk.Button(root, text="Predict Cost", command=predict)
btn_predict.grid(row=6, column=0, columnspan=2, pady=5)

# ---------------------------------------------------------
# 6) Start the Tkinter main loop
# ---------------------------------------------------------
root.mainloop()
