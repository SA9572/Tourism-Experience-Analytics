import os
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = Path(__file__).parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
# On Vercel, use /tmp for SQLite (writable); otherwise use project root
DB_PATH = Path("/tmp/app.db") if os.environ.get("VERCEL") else BASE_DIR / "app.db"


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            visit_year INTEGER,
            visit_month INTEGER,
            continent TEXT,
            attraction_type TEXT,
            predicted_mode TEXT,
            predicted_rating REAL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    conn.commit()
    conn.close()

# Load data and models once at startup
try:
    master_df = pd.read_parquet(PROCESSED_DATA_DIR / "master.parquet")
    rating_model = joblib.load(MODELS_DIR / "rating_regressor.joblib")
    visitmode_model = joblib.load(MODELS_DIR / "visitmode_classifier.joblib")
    recommender: Dict[str, Any] = joblib.load(MODELS_DIR / "recommender.joblib")
    print("✓ Models and data loaded successfully")
except Exception as e:
    print(f"⚠ Error loading models/data: {e}")
    print("⚠ Please ensure train_models.py has been run first")
    raise


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Replicate the feature selection from train_models.py."""
    candidate_features: List[str] = []
    candidate_features += [c for c in ["VisitYear", "VisitMonth"] if c in df.columns]
    candidate_features += [c for c in ["ContinentId", "RegionId", "CountryId", "CityId"] if c in df.columns]
    candidate_features += [c for c in ["AttractionTypeId", "AttractionCityId"] if c in df.columns]
    candidate_features += [c for c in ["UserCountry_Country", "UserRegion_Region", "UserContinent_Continent"] if c in df.columns]
    candidate_features += [c for c in ["AttractionType", "Attraction"] if c in df.columns]
    candidate_features += [
        c for c in [
            "User_avg_rating", "User_num_visits", "User_num_unique_attractions",
            "Attr_avg_rating", "Attr_num_ratings", "Attr_num_unique_users",
        ] if c in df.columns
    ]
    return candidate_features


FEATURE_COLUMNS = select_feature_columns(master_df)


def build_input_row(form_data) -> pd.DataFrame:
    """
    Build a single-row DataFrame with the same feature columns
    used for training.
    """
    row: Dict[str, Any] = {}
    for col in FEATURE_COLUMNS:
        # Form field names are the same as column names
        if col in ["VisitYear", "VisitMonth"]:
            value = form_data.get(col)
            row[col] = int(value) if value not in (None, "", "null") else np.nan
        else:
            row[col] = form_data.get(col) or np.nan
    return pd.DataFrame([row])


def recommend_for_user(user_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Use the pre-computed item-item similarity matrix to recommend attractions.
    """
    user_to_idx = recommender["user_id_to_idx"]
    item_to_idx = recommender["item_id_to_idx"]
    idx_to_item_id = recommender["idx_to_item_id"]
    ratings_matrix = recommender["ratings_matrix"]
    item_similarity = recommender["item_similarity"]

    if user_id not in user_to_idx:
        # Fallback: top attractions by average rating
        top_items = (
            master_df.groupby("AttractionId")["Rating"]
            .mean()
            .sort_values(ascending=False)
            .head(top_k)
            .index.tolist()
        )
    else:
        u_idx = user_to_idx[user_id]
        user_ratings = ratings_matrix[u_idx]

        # Predicted score for an item = similarity-weighted sum of ratings
        scores = item_similarity.dot(user_ratings)

        # Do not recommend already-rated items
        rated_indices = np.where(user_ratings > 0)[0]
        scores[rated_indices] = -np.inf

        top_indices = np.argsort(scores)[::-1][:top_k]
        top_items = [idx_to_item_id[int(i)] for i in top_indices if scores[int(i)] > -np.inf]

    # Attach names and basic info
    recs = (
        master_df[["AttractionId", "Attraction", "AttractionType", "Rating"]]
        .drop_duplicates(subset=["AttractionId"])
    )
    recommendations = (
        recs[recs["AttractionId"].isin(top_items)]
        .sort_values(by="Rating", ascending=False)
        .head(top_k)
        .to_dict(orient="records")
    )
    return recommendations


def build_eda_summaries() -> Dict[str, Any]:
    """Pre-compute simple EDA summaries for charts."""
    summaries: Dict[str, Any] = {}

    if "UserContinent_Continent" in master_df.columns:
        continent_counts = (
            master_df["UserContinent_Continent"]
            .value_counts()
            .reset_index(name="count")
        )
        continent_counts.columns = ["continent", "count"]
        summaries["continent_labels"] = continent_counts["continent"].tolist()
        summaries["continent_counts"] = continent_counts["count"].tolist()

    if "AttractionType" in master_df.columns:
        type_counts = (
            master_df["AttractionType"]
            .value_counts()
            .reset_index(name="count")
        )
        type_counts.columns = ["type", "count"]
        summaries["type_labels"] = type_counts["type"].tolist()
        summaries["type_counts"] = type_counts["count"].tolist()

    if "Rating" in master_df.columns and "Attraction" in master_df.columns:
        top_attractions = (
            master_df.groupby("Attraction")["Rating"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        summaries["top_attraction_labels"] = top_attractions["Attraction"].tolist()
        summaries["top_attraction_ratings"] = top_attractions["Rating"].round(2).tolist()

    return summaries


eda_summaries = build_eda_summaries()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-production")

# Initialize database
try:
    init_db()
    print("✓ Database initialized")
except Exception as e:
    print(f"⚠ Database initialization warning: {e}")


def current_user() -> Optional[sqlite3.Row]:
    user_id = session.get("user_id")
    if not user_id:
        return None
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cur.fetchone()
    conn.close()
    return user


@app.route("/", methods=["GET", "POST"])
def index():
    user = current_user()
    if user is None:
        return redirect(url_for("login"))

    prediction_result = None
    recommendations: List[Dict[str, Any]] = []

    if request.method == "POST":
        if request.form.get("action") == "clear":
            return redirect(url_for("index"))

        X_input = build_input_row(request.form)
        visit_mode = visitmode_model.predict(X_input)[0]
        predicted_rating = float(rating_model.predict(X_input)[0])

        prediction_result = {
            "visit_mode": str(visit_mode),
            "predicted_rating": round(predicted_rating, 2),
        }

        user_id_val = request.form.get("UserId")
        if user_id_val:
            try:
                user_id_int = int(user_id_val)
                recommendations = recommend_for_user(user_id_int, top_k=5)
            except ValueError:
                recommendations = []

        if user is not None:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO search_history (
                    user_id, visit_year, visit_month, continent,
                    attraction_type, predicted_mode, predicted_rating
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user["id"],
                    int(request.form.get("VisitYear")) if request.form.get("VisitYear") else None,
                    int(request.form.get("VisitMonth")) if request.form.get("VisitMonth") else None,
                    request.form.get("UserContinent_Continent") or None,
                    request.form.get("AttractionType") or None,
                    str(visit_mode),
                    float(prediction_result["predicted_rating"]),
                ),
            )
            conn.commit()
            conn.close()

    continents = sorted(
        master_df["UserContinent_Continent"].dropna().unique().tolist()
    ) if "UserContinent_Continent" in master_df.columns else []
    attraction_types = sorted(
        master_df["AttractionType"].dropna().unique().tolist()
    ) if "AttractionType" in master_df.columns else []
    years = sorted(master_df["VisitYear"].dropna().unique().tolist()) if "VisitYear" in master_df.columns else []
    months = sorted(master_df["VisitMonth"].dropna().unique().tolist()) if "VisitMonth" in master_df.columns else []

    return render_template(
        "index.html",
        prediction=prediction_result,
        recommendations=recommendations,
        continents=continents,
        attraction_types=attraction_types,
        years=years,
        months=months,
        eda=eda_summaries,
        feature_columns=FEATURE_COLUMNS,
        user=user,
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    error: Optional[str] = None
    message: Optional[str] = None
    if request.method == "GET" and request.args.get("created") == "1":
        message = "Account created successfully. Please log in."

    if request.method == "POST":
        action = request.form.get("action")
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        if not email or not password:
            error = "Email and password are required."
            return render_template(
                "register.html" if action == "register" else "login.html",
                error=error,
                message=message if action != "register" else None,
            )
        else:
            conn = get_db()
            cur = conn.cursor()
            if action == "register":
                try:
                    cur.execute(
                        "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                        (email, generate_password_hash(password)),
                    )
                    conn.commit()
                    conn.close()
                    return redirect(url_for("login", created=1))
                except sqlite3.IntegrityError:
                    error = "An account with this email already exists."
                    conn.close()
                    return render_template("register.html", error=error)
            else:
                cur.execute("SELECT * FROM users WHERE email = ?", (email,))
                u = cur.fetchone()
                if u is None or not check_password_hash(u["password_hash"], password):
                    error = "Invalid email or password."
                    conn.close()
                    return render_template("login.html", error=error, message=message)
                conn.close()
                session["user_id"] = u["id"]
                session["user_email"] = email
                return redirect(url_for("index"))

    return render_template("login.html", error=error, message=message)


@app.route("/register", methods=["GET"])
def register():
    """Show registration page (form posts to login with action=register)."""
    return render_template("register.html", error=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/history")
def history():
    user = current_user()
    if user is None:
        return redirect(url_for("login"))
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT visit_year, visit_month, continent, attraction_type,
               predicted_mode, predicted_rating, created_at
        FROM search_history WHERE user_id = ?
        ORDER BY created_at DESC LIMIT 100
        """,
        (user["id"],),
    )
    rows = cur.fetchall()
    conn.close()
    return render_template("history.html", user=user, history=rows)


if __name__ == "__main__":
    # Local development
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)