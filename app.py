from pathlib import Path
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Load data and models once at startup
master_df = pd.read_parquet(PROCESSED_DATA_DIR / "master.parquet")
rating_model = joblib.load(MODELS_DIR / "rating_regressor.joblib")
visitmode_model = joblib.load(MODELS_DIR / "visitmode_classifier.joblib")
recommender: Dict[str, Any] = joblib.load(MODELS_DIR / "recommender.joblib")


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Replicate the feature selection from train_models.py."""
    candidate_features: List[str] = []
    candidate_features += [c for c in ["VisitYear", "VisitMonth"] if c in df.columns]
    candidate_features += [c for c in ["ContinentId", "RegionId", "CountryId", "CityId"] if c in df.columns]
    candidate_features += [c for c in ["AttractionTypeId", "AttractionCityId"] if c in df.columns]
    candidate_features += [c for c in ["UserCountry_Country", "UserRegion_Region", "UserContinent_Continent"] if c in df.columns]
    candidate_features += [c for c in ["AttractionType", "Attraction"] if c in df.columns]
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


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    recommendations: List[Dict[str, Any]] = []

    if request.method == "POST":
        # Build feature row from form
        X_input = build_input_row(request.form)

        # Predict visit mode and expected rating
        visit_mode = visitmode_model.predict(X_input)[0]
        predicted_rating = float(rating_model.predict(X_input)[0])

        prediction_result = {
            "visit_mode": str(visit_mode),
            "predicted_rating": round(predicted_rating, 2),
        }

        # User ID for personalization (optional)
        user_id_val = request.form.get("UserId")
        if user_id_val:
            try:
                user_id_int = int(user_id_val)
                recommendations = recommend_for_user(user_id_int, top_k=5)
            except ValueError:
                recommendations = []

    # Dropdown options (built from master_df to keep them realistic)
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
    )


if __name__ == "__main__":
    # Local development
    app.run(host="0.0.0.0", port=5000, debug=True)
