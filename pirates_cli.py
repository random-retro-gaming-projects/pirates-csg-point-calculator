#!/usr/bin/env python3
# predict_ship.py

import argparse
import joblib
import pandas as pd
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

def parse_base_move(move_str):
    # Duplicate logic if needed
    if pd.isnull(move_str):
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
    # Duplicate logic if needed
    if pd.isnull(cannon_str):
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

def main():
    # 1) Define command-line arguments
    parser = argparse.ArgumentParser(description="Predict Pirate CSG ship's point cost.")
    parser.add_argument("--masts", type=int, required=True, help="Number of masts (int).")
    parser.add_argument("--cargo", type=int, required=True, help="Cargo (int).")
    parser.add_argument("--base_move", type=str, required=True, help="Base Move string (e.g. 'S+L').")
    parser.add_argument("--cannons", type=str, required=True, help="Cannons string (e.g. '3S,4L').")
    parser.add_argument("--faction", type=str, required=True, help="Faction name (string).")
    parser.add_argument("--abilities", type=str, required=False, default="", help="Abilities text (string).")

    args = parser.parse_args()

    # 2) Load the pre-trained pipeline
    model_pipeline = joblib.load("pirates_point_model.pkl")

    # 3) Prepare your single-row DataFrame
    bm_val = parse_base_move(args.base_move)
    num_cannons, avg_rank, frac_L = parse_cannons(args.cannons)

    row = {
        "Masts": args.masts,
        "Cargo": args.cargo,
        "BaseMoveValue": bm_val,
        "NumCannons": num_cannons,
        "AvgCannonRank": avg_rank,
        "FracL": frac_L,
        "Faction": args.faction,
        "Ability": args.abilities
    }

    new_df = pd.DataFrame([row])

    # 4) Predict
    predicted_cost = model_pipeline.predict(new_df)[0]

    # 5) Print result
    print(f"Predicted Point Cost: {predicted_cost:.2f}")

if __name__ == "__main__":
    main()
