"""
Run this script once locally to load your CSV files into SQLite.
Place your CSV files in the same folder as this script.

Usage:
    python load_data.py
"""
import sqlite3
import pandas as pd
import os

DB_PATH = "retail.db"

def clean_cols(df):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def load():
    conn = sqlite3.connect(DB_PATH)

    files = {
        "households": "400_households.csv",
        "transactions": "400_transactions.csv",
        "products": "400_products.csv",
    }

    for table, filename in files.items():
        # Try alternate filenames
        candidates = [filename, filename.replace("400_",""), filename.upper()]
        loaded = False
        for candidate in candidates:
            if os.path.exists(candidate):
                print(f"Loading {candidate} → {table}...")
                df = clean_cols(pd.read_csv(candidate))
                df.to_sql(table, conn, if_exists="replace", index=False)
                print(f"  ✓ {len(df)} rows loaded")
                loaded = True
                break
        if not loaded:
            print(f"  ⚠ Could not find file for {table} — tried: {candidates}")

    conn.close()
    print("\nDone! retail.db is ready.")

if __name__ == "__main__":
    load()
