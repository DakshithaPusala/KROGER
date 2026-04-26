"""
ML Models for Kroger Retail Analytics
======================================
This script runs the actual ML models on the Kroger dataset locally.
Results from this script are displayed in the web application dashboard.

Requirements:
    pip install pandas scikit-learn numpy matplotlib seaborn

Usage:
    python ml_models.py

Data files needed (in same folder):
    - 400_transactions.csv
    - 400_households.csv
    - 400_products.csv
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("  KROGER RETAIL ANALYTICS — ML MODEL EXECUTION")
print("=" * 65)

# ── Load Data ────────────────────────────────────────────────────────────────
print("\nLoading data...")
txn  = pd.read_csv('400_transactions.csv', encoding='latin-1')
hh   = pd.read_csv('400_households.csv',   encoding='latin-1')
prod = pd.read_csv('400_products.csv',     encoding='latin-1')

txn.columns  = [c.strip().lower().replace(' ', '_') for c in txn.columns]
hh.columns   = [c.strip().lower().replace(' ', '_') for c in hh.columns]
prod.columns = [c.strip().lower().replace(' ', '_') for c in prod.columns]

print(f"  Transactions : {len(txn):,} records")
print(f"  Households   : {len(hh):,} records")
print(f"  Products     : {len(prod):,} records")

# ─────────────────────────────────────────────────────────────────────────────
# REQ 7: BASKET ANALYSIS — Random Forest Classifier
# Question: What are the commonly purchased product combinations,
#           and how can they drive cross-selling opportunities?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  REQ 7 — BASKET ANALYSIS (Random Forest Classifier)")
print("=" * 65)

# Merge transactions with product info
df = txn.merge(prod[['product_num', 'department', 'commodity']],
               on='product_num', how='left')
df['department'] = df['department'].str.strip()
df['commodity']  = df['commodity'].str.strip()

# --- Step 1: Find top department combinations ---
print("\n[Step 1] Finding top department combinations in baskets...")
basket_depts = (df.groupby('basket_num')['department']
                  .apply(lambda x: sorted(set(x.dropna())))
                  .reset_index())
basket_depts = basket_depts[basket_depts['department'].apply(len) >= 2]

pair_counts = {}
for depts in basket_depts['department']:
    for pair in combinations(depts, 2):
        key = f"{pair[0]} + {pair[1]}"
        pair_counts[key] = pair_counts.get(key, 0) + 1

top_dept_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]
print("\n  Top Department Combinations:")
print(f"  {'Combination':<35} {'Baskets':>10}")
print(f"  {'-'*35} {'-'*10}")
for pair, count in top_dept_pairs:
    print(f"  {pair:<35} {count:>10,}")

# --- Step 2: Find top commodity combinations ---
print("\n[Step 2] Finding top commodity combinations...")
basket_comm = (df.groupby('basket_num')['commodity']
                 .apply(lambda x: sorted(set(x.dropna())))
                 .reset_index())
basket_comm = basket_comm[basket_comm['commodity'].apply(len) >= 2]

comm_pairs = {}
for comms in basket_comm['commodity']:
    for pair in combinations(comms[:6], 2):
        key = f"{pair[0]} + {pair[1]}"
        comm_pairs[key] = comm_pairs.get(key, 0) + 1

top_comm_pairs = sorted(comm_pairs.items(), key=lambda x: x[1], reverse=True)[:8]
print("\n  Top Commodity Combinations (Cross-Sell Opportunities):")
print(f"  {'Combination':<45} {'Baskets':>10}")
print(f"  {'-'*45} {'-'*10}")
for pair, count in top_comm_pairs:
    print(f"  {pair:<45} {count:>10,}")

# --- Step 3: Train Random Forest ---
print("\n[Step 3] Training Random Forest Classifier...")
print("  Target: Predict whether FOOD department appears in basket")

df_basket = df.groupby('basket_num').agg(
    has_food    = ('department', lambda x: int('FOOD' in x.values)),
    has_nonfood = ('department', lambda x: int('NON-FOOD' in x.values)),
    has_pharma  = ('department', lambda x: int('PHARMA' in x.values)),
    total_spend = ('spend', 'sum'),
    num_items   = ('product_num', 'nunique')
).reset_index()

X = df_basket[['has_nonfood', 'has_pharma', 'total_spend', 'num_items']]
y = df_basket['has_food']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(f"\n  Training samples : {len(X_train):,}")
print(f"  Testing samples  : {len(X_test):,}")
print(f"\n  Model Performance:")
print(f"  Accuracy         : {accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"  Precision        : {precision_score(y_test, y_pred)*100:.1f}%")
print(f"  Recall           : {recall_score(y_test, y_pred)*100:.1f}%")

print(f"\n  Feature Importances:")
features_rf = ['Has NonFood', 'Has Pharma', 'Total Spend', 'Num Items']
for feat, imp in zip(features_rf, rf.feature_importances_):
    bar = '|' * int(imp * 50)
    print(f"  {feat:<15} {imp*100:5.1f}%  {bar}")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Food', 'Has Food']))

# ─────────────────────────────────────────────────────────────────────────────
# REQ 8: CHURN PREDICTION — Gradient Boosting Classifier
# Question: Which customers are at risk of disengaging,
#           and how can retention strategies address this?
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  REQ 8 — CHURN PREDICTION (Gradient Boosting Classifier)")
print("=" * 65)

# --- Step 1: Build customer feature matrix ---
print("\n[Step 1] Building customer RFM feature matrix...")
cust = txn.groupby('hshd_num').agg(
    total_spend = ('spend', 'sum'),
    avg_spend   = ('spend', 'mean'),
    txn_count   = ('basket_num', 'nunique'),
    last_week   = ('week_num', 'max'),
    first_week  = ('week_num', 'min'),
    last_year   = ('year', 'max'),
    avg_units   = ('units', 'mean')
).reset_index()

cust = cust.merge(
    hh[['hshd_num', 'l', 'hh_size', 'children', 'age_range', 'income_range']],
    on='hshd_num', how='left')

# Encode features
cust['loyalty_enc']    = (cust['l'].str.strip() == 'Y').astype(int)
cust['has_children']   = (cust['children'].str.strip() != 'N').astype(int)
le = LabelEncoder()
cust['age_enc']        = le.fit_transform(cust['age_range'].fillna('Unknown').str.strip())
cust['income_enc']     = le.fit_transform(cust['income_range'].fillna('Unknown').str.strip())

# Define churn: no purchase in last 10 weeks AND not most recent year
max_week = cust['last_week'].max()
max_year = cust['last_year'].max()
cust['churned'] = ((cust['last_week'] < max_week - 10) &
                   (cust['last_year'] < max_year)).astype(int)
cust['recency'] = max_week - cust['last_week']
cust['tenure']  = cust['last_week'] - cust['first_week']

print(f"\n  Total households : {len(cust)}")
print(f"  Churned          : {cust['churned'].sum()} ({cust['churned'].mean()*100:.1f}%)")
print(f"  Active           : {(cust['churned']==0).sum()}")

# --- Step 2: Train Gradient Boosting ---
print("\n[Step 2] Training Gradient Boosting Classifier...")
features_gb = ['total_spend', 'avg_spend', 'txn_count', 'recency',
               'tenure', 'loyalty_enc', 'has_children', 'age_enc', 'income_enc']

X = cust[features_gb].fillna(0)
y = cust['churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

gb = GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

print(f"\n  Training samples : {len(X_train)}")
print(f"  Testing samples  : {len(X_test)}")
print(f"\n  Model Performance:")
print(f"  Accuracy         : {accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"  Precision        : {precision_score(y_test, y_pred, zero_division=0)*100:.1f}%")
print(f"  Recall           : {recall_score(y_test, y_pred, zero_division=0)*100:.1f}%")

print(f"\n  Feature Importances:")
for feat, imp in sorted(zip(features_gb, gb.feature_importances_),
                        key=lambda x: x[1], reverse=True):
    bar = '|' * int(imp * 50)
    print(f"  {feat:<15} {imp*100:5.1f}%  {bar}")

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Active', 'Churned'], zero_division=0))

print(f"\n  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  {'':15} Predicted Active  Predicted Churned")
print(f"  Actual Active  : {cm[0][0]:>16}  {cm[0][1]:>17}")
print(f"  Actual Churned : {cm[1][0]:>16}  {cm[1][1]:>17}")

# --- Step 3: Churn analysis by demographics ---
print("\n[Step 3] Churn analysis by demographics...")

print("\n  Churn Rate by Income Range:")
churn_income = (cust.groupby('income_range')['churned']
                    .agg(['sum','count','mean'])
                    .reset_index())
churn_income.columns = ['income_range','churned','total','rate']
churn_income['rate'] = (churn_income['rate']*100).round(1)
churn_income = churn_income.sort_values('rate', ascending=False)
print(f"  {'Income Range':<20} {'Churned':>8} {'Total':>8} {'Rate':>8}")
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
for _, row in churn_income.iterrows():
    print(f"  {str(row.income_range).strip():<20} {int(row.churned):>8} {int(row.total):>8} {row.rate:>7.1f}%")

print("\n  Churn Rate by Age Group:")
churn_age = (cust.groupby('age_range')['churned']
                 .agg(['sum','count','mean'])
                 .reset_index())
churn_age.columns = ['age_range','churned','total','rate']
churn_age['rate'] = (churn_age['rate']*100).round(1)
churn_age = churn_age.sort_values('rate', ascending=False)
print(f"  {'Age Range':<20} {'Churned':>8} {'Total':>8} {'Rate':>8}")
print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8}")
for _, row in churn_age.iterrows():
    print(f"  {str(row.age_range).strip():<20} {int(row.churned):>8} {int(row.total):>8} {row.rate:>7.1f}%")

# --- Step 4: At-risk customers ---
print("\n[Step 4] Top 10 at-risk households:")
at_risk = cust[cust['churned'] == 1].sort_values('total_spend').head(10)
print(f"\n  {'HSHD':>6} {'Last Year':>10} {'Recency':>8} {'Txns':>6} {'Total Spend':>12} {'Loyalty':>8}")
print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*6} {'-'*12} {'-'*8}")
for _, row in at_risk.iterrows():
    print(f"  {int(row.hshd_num):>6} {int(row.last_year):>10} {int(row.recency):>8} "
          f"{int(row.txn_count):>6} ${row.total_spend:>11,.2f} {str(row.l).strip():>8}")

print("\n" + "=" * 65)
print("  ML MODEL EXECUTION COMPLETE")
print("  Results above match the web application at:")
print("  https://retailapplicationkroger-ageucvhacqcpbsah.eastus2-01.azurewebsites.net/ml")
print("=" * 65)
