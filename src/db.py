"""Prediction logging store. PostgreSQL via DATABASE_URL in compose; SQLite locally."""
import os, sqlite3, time

SQLITE_PATH = os.environ.get("APP_DB", "data/app.db")

def get_conn():
    if os.environ.get("DATABASE_URL"):
        import psycopg2
        return psycopg2.connect(os.environ["DATABASE_URL"])
    return sqlite3.connect(SQLITE_PATH)

def run_migrations(path=None):
    if path is None:
        path = "migrations/001_init.sql" if os.environ.get("DATABASE_URL") else "migrations/001_init.sqlite.sql"
    with open(path) as f:
        sql = f.read()
    conn = get_conn(); cur = conn.cursor()
    for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
        cur.execute(stmt)
    conn.commit(); conn.close()

def log_prediction(filename, top_class, probability):
    conn = get_conn(); cur = conn.cursor()
    q = ("INSERT INTO prediction_log (ts, filename, top_class, probability) VALUES (%s, %s, %s, %s)"
         if os.environ.get("DATABASE_URL") else
         "INSERT INTO prediction_log (ts, filename, top_class, probability) VALUES (?, ?, ?, ?)")
    cur.execute(q, (int(time.time()), filename, top_class, probability))
    conn.commit(); conn.close()

def history(limit=50):
    conn = get_conn(); cur = conn.cursor()
    q = ("SELECT ts, filename, top_class, probability FROM prediction_log ORDER BY id DESC LIMIT %s"
         if os.environ.get("DATABASE_URL") else
         "SELECT ts, filename, top_class, probability FROM prediction_log ORDER BY id DESC LIMIT ?")
    cur.execute(q, (limit,))
    rows = cur.fetchall(); conn.close()
    return [{"ts": r[0], "filename": r[1], "top_class": r[2], "probability": r[3]} for r in rows]
