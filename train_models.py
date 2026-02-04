import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
import joblib


BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"


def load_tables() -> dict:
    """Load all Excel tables from the raw data directory."""
    tables = {}
    tables["transaction"] = pd.read_excel(RAW_DATA_DIR / "Transaction.xlsx")
    tables["user"] = pd.read_excel(RAW_DATA_DIR / "User.xlsx")
    # Prefer Updated_Item if present, otherwise fall back to Item
    item_path = RAW_DATA_DIR / "Updated_Item.xlsx"
    if not item_path.exists():
        item_path = RAW_DATA_DIR / "Item.xlsx"
    tables["item"] = pd.read_excel(item_path)
    tables["mode"] = pd.read_excel(RAW_DATA_DIR / "Mode.xlsx")
    tables["city"] = pd.read_excel(RAW_DATA_DIR / "City.xlsx")
    tables["country"] = pd.read_excel(RAW_DATA_DIR / "Country.xlsx")
    tables["region"] = pd.read_excel(RAW_DATA_DIR / "Region.xlsx")
    tables["continent"] = pd.read_excel(RAW_DATA_DIR / "Continent.xlsx")
    tables["type"] = pd.read_excel(RAW_DATA_DIR / "Type.xlsx")
    return tables


def build_master_dataset(tables: dict) -> pd.DataFrame:
    """Join transaction, user, item and lookup tables into a single master dataset."""
    tx = tables["transaction"].copy()
    user = tables["user"].copy()
    item = tables["item"].copy()
    mode = tables["mode"].copy()
    city = tables["city"].copy()
    country = tables["country"].copy()
    region = tables["region"].copy()
    continent = tables["continent"].copy()
    atype = tables["type"].copy()

    # Standardise column names to avoid case issues
    def lower_cols(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip() for c in df.columns]
        return df

    tx = lower_cols(tx)
    user = lower_cols(user)
    item = lower_cols(item)
    mode = lower_cols(mode)
    city = lower_cols(city)
    country = lower_cols(country)
    region = lower_cols(region)
    continent = lower_cols(continent)
    atype = lower_cols(atype)

    # Merge user demographics
    master = tx.merge(user, on="UserId", how="left")

    # Merge attraction info
    master = master.merge(item, on="AttractionId", how="left")

    # Join attraction type
    if "AttractionTypeId" in master.columns and "AttractionTypeId" in atype.columns:
        master = master.merge(atype, on="AttractionTypeId", how="left", suffixes=("", "_Type"))

    # Join visit mode labels if transaction uses an ID
    if "VisitModeId" in master.columns and "VisitModeId" in mode.columns:
        master = master.merge(mode, on="VisitModeId", how="left", suffixes=("", "_Mode"))

    # Join city / country / region / continent for attraction or user if possible
    if "AttractionCityId" in master.columns and "CityId" in city.columns:
        master = master.merge(city.add_prefix("AttractionCity_"), left_on="AttractionCityId", right_on="AttractionCity_CityId", how="left")

    # Join region / country / continent (for user side) – these may already be numeric IDs
    if "CountryId" in master.columns and "CountryId" in country.columns:
        master = master.merge(country.add_prefix("UserCountry_"), left_on="CountryId", right_on="UserCountry_CountryId", how="left")

    if "RegionId" in master.columns and "RegionId" in region.columns:
        master = master.merge(region.add_prefix("UserRegion_"), left_on="RegionId", right_on="UserRegion_RegionId", how="left")

    if "ContinentId" in master.columns and "ContinentId" in continent.columns:
        master = master.merge(continent.add_prefix("UserContinent_"), left_on="ContinentId", right_on="UserContinent_ContinentId", how="left")

    # Basic cleaning: drop completely duplicate rows
    master = master.drop_duplicates()

    # Drop rows with missing essential fields
    essential_cols = [c for c in ["UserId", "AttractionId", "Rating"] if c in master.columns]
    master = master.dropna(subset=essential_cols)

    return master


def build_feature_sets(master: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create separate targets for regression (rating) and classification (visit mode)."""
    # Determine the visit mode target column
    if "VisitMode" in master.columns:
        y_class = master["VisitMode"]
    elif "VisitMode_Mode" in master.columns:
        y_class = master["VisitMode_Mode"]
    elif "VisitModeId" in master.columns:
        y_class = master["VisitModeId"]
    else:
        raise ValueError("No visit mode column found (expected VisitMode or VisitModeId).")

    if "Rating" not in master.columns:
        raise ValueError("Rating column not found in master dataset.")

    y_reg = master["Rating"]

    # Candidate feature columns
    candidate_features: List[str] = []
    candidate_features += [c for c in ["VisitYear", "VisitMonth"] if c in master.columns]
    candidate_features += [c for c in ["ContinentId", "RegionId", "CountryId", "CityId"] if c in master.columns]
    candidate_features += [c for c in ["AttractionTypeId", "AttractionCityId"] if c in master.columns]
    candidate_features += [c for c in ["UserCountry_Country", "UserRegion_Region", "UserContinent_Continent"] if c in master.columns]
    candidate_features += [c for c in ["AttractionType", "Attraction"] if c in master.columns]

    X = master[candidate_features].copy()

    return X, y_reg, y_class


def train_regression_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Pipeline, dict]:
    """Train a regression model to predict ratings."""
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_cols),
            ("num", numeric_pipeline, numeric_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }
    return pipe, metrics


def train_classification_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Pipeline, dict]:
    """Train a classifier to predict visit mode."""
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_cols),
            ("num", numeric_pipeline, numeric_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }
    return pipe, metrics


def build_recommender(master: pd.DataFrame) -> dict:
    """
    Build a simple user-item collaborative filtering recommender
    based on cosine similarity between items.
    """
    ratings = master[["UserId", "AttractionId", "Rating"]].copy()

    # Create integer indices
    user_ids = ratings["UserId"].unique()
    item_ids = ratings["AttractionId"].unique()

    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    idx_to_user_id = {i: uid for uid, i in user_id_to_idx.items()}
    item_id_to_idx = {iid: i for i, iid in enumerate(item_ids)}
    idx_to_item_id = {i: iid for iid, i in item_id_to_idx.items()}

    user_count = len(user_ids)
    item_count = len(item_ids)
    matrix = np.zeros((user_count, item_count), dtype=np.float32)

    for _, row in ratings.iterrows():
        ui = user_id_to_idx[row["UserId"]]
        ii = item_id_to_idx[row["AttractionId"]]
        matrix[ui, ii] = float(row["Rating"])

    # Compute item-item similarity
    # Add a tiny value to avoid division by zero if a column is all zeros
    item_norms = np.linalg.norm(matrix, axis=0)
    item_norms[item_norms == 0] = 1e-6
    normalized_items = matrix / item_norms
    similarity = cosine_similarity(normalized_items.T)

    recommender = {
        "user_ids": user_ids,
        "item_ids": item_ids,
        "user_id_to_idx": user_id_to_idx,
        "idx_to_user_id": idx_to_user_id,
        "item_id_to_idx": item_id_to_idx,
        "idx_to_item_id": idx_to_item_id,
        "ratings_matrix": matrix,
        "item_similarity": similarity,
    }
    return recommender


def save_artifacts(
    master: pd.DataFrame,
    reg_model: Pipeline,
    reg_metrics: dict,
    clf_model: Pipeline,
    clf_metrics: dict,
    recommender: dict,
) -> None:
    """Persist processed data, models, metrics, and recommender structures."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    master.to_parquet(PROCESSED_DATA_DIR / "master.parquet", index=False)

    joblib.dump(reg_model, MODELS_DIR / "rating_regressor.joblib")
    joblib.dump(clf_model, MODELS_DIR / "visitmode_classifier.joblib")
    joblib.dump(recommender, MODELS_DIR / "recommender.joblib")

    metrics = {
        "regression": reg_metrics,
        "classification": clf_metrics,
    }
    metrics_df = pd.json_normalize(metrics)
    metrics_df.to_csv(PROCESSED_DATA_DIR / "model_metrics.csv", index=False)


def main():
    print("Loading tables...")
    tables = load_tables()

    print("Building master dataset...")
    master = build_master_dataset(tables)
    print(f"Master dataset shape: {master.shape}")

    print("Building feature sets...")
    X, y_reg, y_class = build_feature_sets(master)
    print(f"Feature matrix shape: {X.shape}")

    print("Training regression model (rating prediction)...")
    reg_model, reg_metrics = train_regression_model(X, y_reg)
    print(f"Regression metrics: {reg_metrics}")

    print("Training classification model (visit mode prediction)...")
    clf_model, clf_metrics = train_classification_model(X, y_class)
    print(f"Classification metrics: {clf_metrics}")

    print("Building recommender system...")
    recommender = build_recommender(master)

    print("Saving artifacts...")
    save_artifacts(master, reg_model, reg_metrics, clf_model, clf_metrics, recommender)

    print("All done.")


if __name__ == "__main__":
    main()

