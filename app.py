from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

app = Flask(__name__)
app.secret_key = "kroger-retail-secret"
DB_PATH = "retail.db"

# ── DB helpers ──────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS households (
            hshd_num INTEGER, loyalty_flag TEXT, age_range TEXT,
            marital_status TEXT, income_range TEXT, homeowner_desc TEXT,
            hshd_composition TEXT, hshd_size TEXT, children TEXT
        );
        CREATE TABLE IF NOT EXISTS transactions (
            hshd_num INTEGER, basket_num TEXT, date TEXT,
            product_num TEXT, spend REAL, units INTEGER,
            store_region TEXT, week_num INTEGER, year INTEGER
        );
        CREATE TABLE IF NOT EXISTS products (
            product_num TEXT, department TEXT, commodity TEXT,
            brand_type TEXT, natural_organic_flag TEXT
        );
    """)
    conn.commit()
    conn.close()

def load_csv_to_db(filepath, table):
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    conn = get_db()
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return redirect(url_for("login"))

# Req 2 — Login page
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        return redirect(url_for("search"))
    return render_template("login.html")

# Req 3 — Sample data pull for HSHD #10
@app.route("/hshd10")
def hshd10():
    conn = get_db()
    rows = conn.execute("""
        SELECT t.hshd_num, t.basket_num, t.date, t.product_num,
               p.department, p.commodity, t.spend, t.units,
               t.store_region, t.week_num, t.year,
               h.loyalty_flag, h.age_range, h.marital_status,
               h.income_range, h.homeowner_desc, h.hshd_composition,
               h.hshd_size, h.children
        FROM transactions t
        LEFT JOIN households h ON t.hshd_num = h.hshd_num
        LEFT JOIN products p ON t.product_num = p.product_num
        WHERE t.hshd_num = 10
        ORDER BY t.hshd_num, t.basket_num, t.date, t.product_num,
                 p.department, p.commodity
    """).fetchall()
    conn.close()
    return render_template("hshd10.html", rows=rows)

# Req 4 — Interactive search by hshd_num
@app.route("/search", methods=["GET", "POST"])
def search():
    rows = []
    hshd_num = None
    if request.method == "POST":
        hshd_num = request.form.get("hshd_num", "").strip()
        if hshd_num:
            conn = get_db()
            rows = conn.execute("""
                SELECT t.hshd_num, t.basket_num, t.date, t.product_num,
                       p.department, p.commodity, t.spend, t.units,
                       t.store_region, t.week_num, t.year,
                       h.loyalty_flag, h.age_range, h.income_range,
                       h.hshd_composition, h.hshd_size, h.children
                FROM transactions t
                LEFT JOIN households h ON t.hshd_num = h.hshd_num
                LEFT JOIN products p ON t.product_num = p.product_num
                WHERE t.hshd_num = ?
                ORDER BY t.hshd_num, t.basket_num, t.date, t.product_num,
                         p.department, p.commodity
            """, (hshd_num,)).fetchall()
            conn.close()
    return render_template("search.html", rows=rows, hshd_num=hshd_num)

# Req 5 — Upload new data
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        for table, key in [("households","households"),("transactions","transactions"),("products","products")]:
            f = request.files.get(key)
            if f and f.filename:
                path = f"/tmp/{key}.csv"
                f.save(path)
                load_csv_to_db(path, table)
        flash("Data uploaded successfully!")
        return redirect(url_for("search"))
    return render_template("upload.html")

# Req 6 — Dashboard
@app.route("/dashboard")
def dashboard():
    conn = get_db()
    charts = {}

    # Spend over time
    df_time = pd.read_sql("""
        SELECT year, week_num, SUM(spend) as total_spend
        FROM transactions GROUP BY year, week_num ORDER BY year, week_num
    """, conn)
    if not df_time.empty:
        fig = px.line(df_time, x="week_num", y="total_spend", color="year",
                      title="Weekly Spend Over Time",
                      labels={"week_num":"Week","total_spend":"Total Spend ($)"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e0e0e0")
        charts["spend_time"] = fig.to_html(full_html=False)

    # Spend by department
    df_dept = pd.read_sql("""
        SELECT p.department, SUM(t.spend) as total_spend
        FROM transactions t JOIN products p ON t.product_num = p.product_num
        WHERE p.department IS NOT NULL
        GROUP BY p.department ORDER BY total_spend DESC LIMIT 10
    """, conn)
    if not df_dept.empty:
        fig2 = px.bar(df_dept, x="department", y="total_spend",
                      title="Top 10 Departments by Spend",
                      labels={"department":"Department","total_spend":"Total Spend ($)"})
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e0e0e0")
        charts["dept_spend"] = fig2.to_html(full_html=False)

    # Loyalty vs spend
    df_loyal = pd.read_sql("""
        SELECT h.loyalty_flag, SUM(t.spend) as total_spend, COUNT(*) as txn_count
        FROM transactions t JOIN households h ON t.hshd_num = h.hshd_num
        WHERE h.loyalty_flag IS NOT NULL
        GROUP BY h.loyalty_flag
    """, conn)
    if not df_loyal.empty:
        fig3 = px.bar(df_loyal, x="loyalty_flag", y="total_spend",
                      title="Loyalty Members vs Non-Members: Total Spend",
                      labels={"loyalty_flag":"Loyalty Flag","total_spend":"Total Spend ($)"})
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e0e0e0")
        charts["loyalty"] = fig3.to_html(full_html=False)

    # Brand preference
    df_brand = pd.read_sql("""
        SELECT p.brand_type, SUM(t.spend) as total_spend
        FROM transactions t JOIN products p ON t.product_num = p.product_num
        WHERE p.brand_type IS NOT NULL
        GROUP BY p.brand_type
    """, conn)
    if not df_brand.empty:
        fig4 = px.pie(df_brand, names="brand_type", values="total_spend",
                      title="Private vs National Brand Spend")
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0")
        charts["brand"] = fig4.to_html(full_html=False)

    conn.close()
    return render_template("dashboard.html", charts=charts)

# Req 7 — ML: Basket Analysis + Req 8 — Churn Prediction
@app.route("/ml")
def ml():
    conn = get_db()
    results = {}

    # ── Req 7: Basket Analysis with Random Forest ──
    df = pd.read_sql("""
        SELECT t.hshd_num, t.basket_num, p.department, p.commodity,
               p.brand_type, t.spend, t.units
        FROM transactions t
        JOIN products p ON t.product_num = p.product_num
        WHERE p.department IS NOT NULL
    """, conn)

    if not df.empty and len(df) > 100:
        le = LabelEncoder()
        df["dept_enc"] = le.fit_transform(df["department"].fillna("Unknown"))
        df["brand_enc"] = le.fit_transform(df["brand_type"].fillna("Unknown"))
        X = df[["dept_enc","brand_enc","spend","units"]].fillna(0)
        y = df["dept_enc"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        importances = dict(zip(["Department","Brand","Spend","Units"],
                               [round(i*100,2) for i in rf.feature_importances_]))
        results["basket_importances"] = importances
        results["basket_score"] = round(rf.score(X_test, y_test)*100, 2)

        # Top co-purchased departments
        basket_dept = df.groupby(["basket_num","department"]).size().reset_index(name="count")
        top_combos = basket_dept.groupby("department")["count"].sum().sort_values(ascending=False).head(8)
        fig_basket = px.bar(x=top_combos.index, y=top_combos.values,
                            title="Most Frequently Purchased Departments",
                            labels={"x":"Department","y":"Frequency"})
        fig_basket.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 font_color="#e0e0e0")
        results["basket_chart"] = fig_basket.to_html(full_html=False)

    # ── Req 8: Churn Prediction with Gradient Boosting ──
    df_txn = pd.read_sql("""
        SELECT hshd_num, SUM(spend) as total_spend,
               COUNT(*) as txn_count, MAX(year) as last_year,
               MAX(week_num) as last_week, AVG(spend) as avg_spend
        FROM transactions GROUP BY hshd_num
    """, conn)
    df_hh = pd.read_sql("SELECT hshd_num, loyalty_flag FROM households", conn)

    if not df_txn.empty and not df_hh.empty:
        df_churn = df_txn.merge(df_hh, on="hshd_num", how="left")
        max_week = df_churn["last_week"].max()
        df_churn["churned"] = ((df_churn["last_week"] < max_week - 10) &
                               (df_churn["last_year"] == df_churn["last_year"].min())).astype(int)
        df_churn["loyalty_enc"] = (df_churn["loyalty_flag"] == "Y").astype(int)
        features = ["total_spend","txn_count","avg_spend","loyalty_enc"]
        X = df_churn[features].fillna(0)
        y = df_churn["churned"]

        if y.nunique() > 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
            gb.fit(X_train, y_train)
            y_pred = gb.predict(X_test)
            results["churn_accuracy"] = round((y_pred == y_test).mean()*100, 2)
            results["churn_count"] = int(df_churn["churned"].sum())
            results["total_hh"] = int(len(df_churn))
            results["churn_rate"] = round(df_churn["churned"].mean()*100, 2)

            # Churn by loyalty
            churn_loyal = df_churn.groupby("loyalty_flag")["churned"].mean().reset_index()
            churn_loyal.columns = ["Loyalty","Churn Rate"]
            churn_loyal["Churn Rate"] = (churn_loyal["Churn Rate"]*100).round(2)
            fig_churn = px.bar(churn_loyal, x="Loyalty", y="Churn Rate",
                               title="Churn Rate by Loyalty Status (%)",
                               labels={"Loyalty":"Loyalty Flag","Churn Rate":"Churn Rate (%)"})
            fig_churn.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font_color="#e0e0e0")
            results["churn_chart"] = fig_churn.to_html(full_html=False)

    conn.close()
    return render_template("ml.html", results=results)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
