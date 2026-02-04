## Tourism Experience Analytics: Classification, Prediction & Recommendation

This project uses real tourism data (transactions, users, attractions, geography) to:
- **Predict attraction ratings** (regression)
- **Predict visit mode** such as Business / Family / Couples (classification)
- **Recommend attractions** based on past behaviour (collaborative filtering)
- **Visualise tourism insights** in a modern web UI

The stack is **Python + scikit-learn for ML** and **Flask + HTML/CSS/JS** for the web app, deployable on **Render**.

### 1. Project Structure

- `data/raw/` – original Excel files (`Transaction.xlsx`, `User.xlsx`, `Item.xlsx`, `Mode.xlsx`, `City.xlsx`, `Country.xlsx`, `Region.xlsx`, `Continent.xlsx`, `Type.xlsx`, `Updated_Item.xlsx`)
- `data/processed/master.parquet` – cleaned, joined dataset used for modelling
- `models/`
  - `rating_regressor.joblib` – regression model to predict ratings
  - `visitmode_classifier.joblib` – classification model to predict visit mode
  - `recommender.joblib` – user–item collaborative filtering structures
- `train_models.py` – end‑to‑end data preparation + model training script
- `app.py` – Flask app exposing predictions, recommendations, and charts
- `templates/index.html` – main HTML template
- `static/css/style.css` – modern dark UI styling
- `static/js/main.js` – charts with Chart.js
- `requirements.txt` – Python dependencies

### 2. Data Pipeline & Models

`train_models.py` performs:
- **Load** all Excel tables from `data/raw/`
- **Join**: transactions ↔ users ↔ items ↔ type ↔ city / country / region / continent
- **Cleaning**:
  - standardise column names
  - drop duplicates
  - drop rows missing key fields (`UserId`, `AttractionId`, `Rating`)
- **Feature engineering**:
  - use visit year/month, geography IDs, attraction type/city and user geography names as features
  - encode categoricals with `OneHotEncoder`
  - scale numeric features with `StandardScaler`
  - impute missing values (`SimpleImputer` – most_frequent for categoricals, median for numerics)
- **Regression**: `RandomForestRegressor` to predict `Rating`
- **Classification**: `RandomForestClassifier` to predict `VisitMode`
- **Recommendation**:
  - build a **user–item rating matrix**
  - compute **item–item cosine similarity**
  - use this for top‑N attraction recommendations
- **Persist**:
  - `data/processed/master.parquet`
  - models in `models/`
  - simple metrics in `data/processed/model_metrics.csv`

Run the full pipeline:

```bash
pip install -r requirements.txt
python train_models.py
```

### 3. Web Application (Flask + HTML/CSS/JS)

`app.py`:
- Loads `master.parquet` and all trained models at startup
- Defines a single route `/` that:
  - shows a **traveller profile form** (year, month, continent, attraction type, user ID)
  - on **POST**, builds the same feature vector used in training
  - returns:
    - **Predicted visit mode**
    - **Predicted rating** for the given profile
    - **Top‑K recommended attractions** for the given `UserId` (if it exists in the data)
- Pre‑computes basic **EDA summaries** for:
  - users by continent
  - attractions by type
  - top‑rated attractions

`templates/index.html`:
- Responsive, card‑based layout
- Form bound directly to the feature columns
- Sections for **prediction results**, **recommendation list**, and **charts**

`static/js/main.js` + Chart.js:
- Renders bar, doughnut, and horizontal bar charts using EDA summaries passed from Flask.

### 4. Local Development

```bash
python -m venv .venv
.venv\Scripts\activate   # on Windows
# or source .venv/bin/activate on macOS/Linux

pip install -r requirements.txt
python train_models.py   # prepares data + trains models
python app.py            # start Flask dev server
```

Open `http://localhost:5000` in your browser.

### 5. Deploying on Render

1. Push this project to a Git repository (e.g. GitHub).
2. In Render:
   - Create a **New Web Service**.
   - Connect the repo.
   - Set **Runtime** to Python 3.x.
   - **Build command**:
     ```bash
     pip install -r requirements.txt
     python train_models.py
     ```
   - **Start command**:
     ```bash
     gunicorn app:app
     ```
3. Make sure `data/raw/*.xlsx` are included in the repo so `train_models.py` can run in the Render environment.

Render will then host the Flask app, exposing the same UI and functionality as your local environment.

### 6. How This Meets the Project Objectives

- **Regression**: predict attraction ratings for a given user profile.
- **Classification**: predict visit mode (Business / Family / Couples / Friends / etc.).
- **Recommendation**: item‑based collaborative filtering using real ratings.
- **EDA & Visualisation**: charts for user geography, attraction types, and top attractions.
- **Tech stack**: Python ML + Flask backend + HTML/CSS/JS frontend, deployable on Render.

You can extend this further with more engineered features, additional charts, or a hybrid recommender that mixes content‑based similarity with the current collaborative approach.

