from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "kroger-retail-secret"
DB_PATH = os.path.join(os.path.expanduser("~"), "retail.db")

# ── DB helpers ──────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS households (
            hshd_num INTEGER, l TEXT, age_range TEXT,
            marital TEXT, income_range TEXT, homeowner TEXT,
            hshd_composition TEXT, hh_size TEXT, children TEXT
        );
        CREATE TABLE IF NOT EXISTS transactions (
            basket_num TEXT, hshd_num INTEGER, purchase_ TEXT,
            product_num TEXT, spend REAL, units INTEGER,
            store_r TEXT, week_num INTEGER, year INTEGER
        );
        CREATE TABLE IF NOT EXISTS products (
            product_num TEXT, department TEXT, commodity TEXT,
            brand_ty TEXT, natural_organic_flag TEXT
        );
    """)
    conn.commit()
    conn.close()

def load_csv_to_db(filepath, table):
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except Exception:
        df = pd.read_csv(filepath, encoding='latin-1')
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    conn = get_db()
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()

# Initialize DB on startup
init_db()

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
        SELECT t.hshd_num, t.basket_num, t.purchase_, t.product_num,
               p.department, p.commodity, t.spend, t.units,
               t.store_r, t.week_num, t.year,
               h.l, h.age_range, h.marital,
               h.income_range, h.homeowner, h.hshd_composition,
               h.hh_size, h.children
        FROM transactions t
        LEFT JOIN households h ON CAST(t.hshd_num AS TEXT) = CAST(h.hshd_num AS TEXT)
        LEFT JOIN products p ON CAST(t.product_num AS TEXT) = CAST(p.product_num AS TEXT)
        WHERE CAST(t.hshd_num AS TEXT) = '10'
        ORDER BY t.hshd_num, t.basket_num, t.purchase_, t.product_num,
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
                SELECT t.hshd_num, t.basket_num, t.purchase_, t.product_num,
                       p.department, p.commodity, t.spend, t.units,
                       t.store_r, t.week_num, t.year,
                       h.l, h.age_range, h.income_range,
                       h.hshd_composition, h.hh_size, h.children
                FROM transactions t
                LEFT JOIN households h ON CAST(t.hshd_num AS TEXT) = CAST(h.hshd_num AS TEXT)
                LEFT JOIN products p ON CAST(t.product_num AS TEXT) = CAST(p.product_num AS TEXT)
                WHERE CAST(t.hshd_num AS TEXT) = CAST(? AS TEXT)
                ORDER BY t.hshd_num, t.basket_num, t.purchase_, t.product_num,
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

# Req 6 — Dashboard (lightweight SQL aggregations only, no Plotly)
@app.route("/dashboard")
def dashboard():
    conn = get_db()
    data = {}

    try:
        # Spend by year
        df = pd.read_sql("SELECT year, ROUND(SUM(spend),2) as total FROM transactions GROUP BY year ORDER BY year", conn)
        data["spend_by_year"] = df.to_dict("records")
    except: data["spend_by_year"] = []

    try:
        # Top departments
        df = pd.read_sql("""
            SELECT p.department, ROUND(SUM(t.spend),2) as total
            FROM transactions t JOIN products p ON CAST(t.product_num AS TEXT)=CAST(p.product_num AS TEXT)
            WHERE p.department IS NOT NULL
            GROUP BY p.department ORDER BY total DESC LIMIT 10
        """, conn)
        data["top_depts"] = df.to_dict("records")
    except: data["top_depts"] = []

    try:
        # Loyalty spend
        df = pd.read_sql("""
            SELECT h.l as loyalty, ROUND(SUM(t.spend),2) as total, COUNT(*) as txns
            FROM transactions t JOIN households h ON CAST(t.hshd_num AS TEXT)=CAST(h.hshd_num AS TEXT)
            WHERE h.l IS NOT NULL GROUP BY h.l
        """, conn)
        data["loyalty"] = df.to_dict("records")
    except: data["loyalty"] = []

    try:
        # Brand preference
        df = pd.read_sql("""
            SELECT p.brand_ty as brand, ROUND(SUM(t.spend),2) as total
            FROM transactions t JOIN products p ON CAST(t.product_num AS TEXT)=CAST(p.product_num AS TEXT)
            WHERE p.brand_ty IS NOT NULL GROUP BY p.brand_ty ORDER BY total DESC
        """, conn)
        data["brands"] = df.to_dict("records")
    except: data["brands"] = []

    try:
        # Top commodities
        df = pd.read_sql("""
            SELECT p.commodity, ROUND(SUM(t.spend),2) as total
            FROM transactions t JOIN products p ON CAST(t.product_num AS TEXT)=CAST(p.product_num AS TEXT)
            WHERE p.commodity IS NOT NULL
            GROUP BY p.commodity ORDER BY total DESC LIMIT 10
        """, conn)
        data["top_commodities"] = df.to_dict("records")
    except: data["top_commodities"] = []

    try:
        # Region spend
        df = pd.read_sql("""
            SELECT store_r as region, ROUND(SUM(spend),2) as total
            FROM transactions WHERE store_r IS NOT NULL
            GROUP BY store_r ORDER BY total DESC
        """, conn)
        data["regions"] = df.to_dict("records")
    except: data["regions"] = []

    conn.close()
    return render_template("dashboard.html", data=data)

# Req 7 + 8 — ML (lightweight pandas stats, no sklearn training)
@app.route("/ml")
def ml():
    conn = get_db()
    results = {}

    # Req 7: Basket Analysis — top co-purchased departments (SQL only)
    try:
        df = pd.read_sql("""
            SELECT p.department, COUNT(DISTINCT t.basket_num) as baskets,
                   ROUND(SUM(t.spend),2) as total_spend,
                   ROUND(AVG(t.spend),2) as avg_spend
            FROM transactions t
            JOIN products p ON CAST(t.product_num AS TEXT)=CAST(p.product_num AS TEXT)
            WHERE p.department IS NOT NULL
            GROUP BY p.department ORDER BY baskets DESC LIMIT 10
        """, conn)
        results["basket"] = df.to_dict("records")
    except: results["basket"] = []

    try:
        # Top commodity pairs in same basket
        df = pd.read_sql("""
            SELECT p.commodity, COUNT(*) as freq, ROUND(SUM(t.spend),2) as total
            FROM transactions t
            JOIN products p ON CAST(t.product_num AS TEXT)=CAST(p.product_num AS TEXT)
            WHERE p.commodity IS NOT NULL
            GROUP BY p.commodity ORDER BY freq DESC LIMIT 10
        """, conn)
        results["commodities"] = df.to_dict("records")
    except: results["commodities"] = []

    # Req 8: Churn Prediction — SQL-based analysis
    try:
        df = pd.read_sql("""
            SELECT t.hshd_num, MAX(t.year) as last_year,
                   MAX(t.week_num) as last_week,
                   COUNT(*) as txn_count,
                   ROUND(SUM(t.spend),2) as total_spend,
                   h.l as loyalty
            FROM transactions t
            LEFT JOIN households h ON CAST(t.hshd_num AS TEXT)=CAST(h.hshd_num AS TEXT)
            GROUP BY t.hshd_num
        """, conn)

        if not df.empty:
            max_week = df["last_week"].max()
            max_year = df["last_year"].max()
            df["churned"] = ((df["last_week"] < max_week - 10) &
                             (df["last_year"] < max_year)).astype(int)

            results["total_hh"] = len(df)
            results["churned_count"] = int(df["churned"].sum())
            results["active_count"] = int((df["churned"] == 0).sum())
            results["churn_rate"] = round(df["churned"].mean() * 100, 2)

            # Churn by loyalty
            churn_by_loyalty = df.groupby("loyalty")["churned"].agg(["sum","count","mean"]).reset_index()
            churn_by_loyalty.columns = ["loyalty","churned","total","rate"]
            churn_by_loyalty["rate"] = (churn_by_loyalty["rate"] * 100).round(2)
            results["churn_by_loyalty"] = churn_by_loyalty.to_dict("records")

            # High risk customers (low spend, low frequency, not recent)
            at_risk = df[(df["churned"]==1)].sort_values("total_spend").head(10)
            results["at_risk"] = at_risk[["hshd_num","last_year","last_week","txn_count","total_spend","loyalty"]].to_dict("records")

    except Exception as e:
        results["churn_error"] = str(e)

    conn.close()
    return render_template("ml.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
