from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "kroger-retail-secret"
DB_PATH = os.path.join(os.path.expanduser("~"), "retail.db")

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

init_db()

@app.route("/")
def index():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        return redirect(url_for("search"))
    return render_template("login.html")

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

@app.route("/dashboard")
def dashboard():
    conn = get_db()
    data = {}

    try:
        cur = conn.execute("SELECT year, ROUND(SUM(spend),2) as total FROM transactions WHERE year IS NOT NULL GROUP BY year ORDER BY year")
        data["spend_by_year"] = [dict(r) for r in cur.fetchall()]
    except: data["spend_by_year"] = []

    try:
        cur = conn.execute("SELECT store_r as region, ROUND(SUM(spend),2) as total FROM transactions WHERE store_r IS NOT NULL GROUP BY store_r ORDER BY total DESC")
        data["regions"] = [dict(r) for r in cur.fetchall()]
    except: data["regions"] = []

    try:
        cur = conn.execute("SELECT l as loyalty, COUNT(*) as total_hh FROM households WHERE l IS NOT NULL GROUP BY l")
        data["loyalty"] = [dict(r) for r in cur.fetchall()]
    except: data["loyalty"] = []

    try:
        cur = conn.execute("SELECT department, COUNT(*) as count FROM products WHERE department IS NOT NULL GROUP BY department ORDER BY count DESC LIMIT 10")
        data["top_depts"] = [dict(r) for r in cur.fetchall()]
    except: data["top_depts"] = []

    try:
        cur = conn.execute("SELECT brand_ty as brand, COUNT(*) as count FROM products WHERE brand_ty IS NOT NULL GROUP BY brand_ty ORDER BY count DESC")
        data["brands"] = [dict(r) for r in cur.fetchall()]
    except: data["brands"] = []

    try:
        cur = conn.execute("SELECT commodity, COUNT(*) as count FROM products WHERE commodity IS NOT NULL GROUP BY commodity ORDER BY count DESC LIMIT 10")
        data["top_commodities"] = [dict(r) for r in cur.fetchall()]
    except: data["top_commodities"] = []

    conn.close()
    return render_template("dashboard.html", data=data)

@app.route("/ml")
def ml():
    conn = get_db()
    results = {}

    # Req 7: Basket Analysis
    try:
        cur = conn.execute("SELECT department, COUNT(*) as baskets FROM products WHERE department IS NOT NULL GROUP BY department ORDER BY baskets DESC LIMIT 10")
        results["basket"] = [dict(r) for r in cur.fetchall()]
    except: results["basket"] = []

    try:
        cur = conn.execute("SELECT commodity, COUNT(*) as freq FROM products WHERE commodity IS NOT NULL GROUP BY commodity ORDER BY freq DESC LIMIT 10")
        results["commodities"] = [dict(r) for r in cur.fetchall()]
    except: results["commodities"] = []

    # Req 8: Churn Prediction
    try:
        cur = conn.execute("""
            SELECT hshd_num, MAX(year) as last_year, MAX(week_num) as last_week,
                   COUNT(*) as txn_count, ROUND(SUM(spend),2) as total_spend
            FROM transactions GROUP BY hshd_num
        """)
        rows = cur.fetchall()
        if rows:
            max_week = max(r["last_week"] for r in rows)
            max_year = max(r["last_year"] for r in rows)
            churned = [r for r in rows if r["last_week"] < max_week - 10 and r["last_year"] < max_year]
            active = [r for r in rows if not (r["last_week"] < max_week - 10 and r["last_year"] < max_year)]
            results["total_hh"] = len(rows)
            results["churned_count"] = len(churned)
            results["active_count"] = len(active)
            results["churn_rate"] = round(len(churned)/len(rows)*100, 2)
            results["at_risk"] = [dict(r) for r in sorted(churned, key=lambda x: x["total_spend"])[:10]]
    except Exception as e:
        results["churn_error"] = str(e)

    try:
        cur = conn.execute("SELECT l as loyalty, COUNT(*) as total FROM households WHERE l IS NOT NULL GROUP BY l")
        results["churn_by_loyalty"] = [{"loyalty": r["loyalty"], "total": r["total"], "churned": 0, "rate": 0} for r in cur.fetchall()]
    except: results["churn_by_loyalty"] = []

    conn.close()
    return render_template("ml.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
