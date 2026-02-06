# Deploy Tourism Experience Analytics (Streamlit) – Frontend & Backend Together

This project can run as a **single Streamlit app** (frontend + backend in one). You can deploy it to **Streamlit Community Cloud** or **Render**.

## What’s included in the Streamlit app

- **Same backend**: Uses the same `app.py` data, models (`rating_regressor.joblib`, `visitmode_classifier.joblib`, `recommender.joblib`), and SQLite DB.
- **Same features**: Login / register, predictions, recommendations, search history, EDA charts.
- **One process**: Run `streamlit run streamlit_app.py` – no separate frontend or backend server.

## Run locally

```bash
# From project root (after training models)
pip install -r requirements.txt
python train_models.py
streamlit run streamlit_app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Deploy on Streamlit Community Cloud

1. **Push to GitHub**  
   Ensure the repo includes:
   - `streamlit_app.py`
   - `app.py`
   - `train_models.py`
   - `requirements.txt`
   - `data/raw/*.xlsx`
   - `templates/` and `static/` (optional; Streamlit app doesn’t use them, but keep if you also use Flask)

2. **Go to [share.streamlit.io](https://share.streamlit.io)**  
   Sign in with GitHub.

3. **New app**
   - **Repository**: `your-username/your-repo`
   - **Branch**: `main` (or your default)
   - **Main file path**: `streamlit_app.py`
   - **App URL**: e.g. `tourism-experience-analytics`

4. **Advanced settings**
   - **Python version**: 3.11 (or 3.10).
   - Streamlit Cloud will run:
     - `pip install -r requirements.txt`
   - You **must have models and processed data** in the repo, or the app will fail on import. So either:
     - Run `python train_models.py` locally and commit `data/processed/` and `models/`, **or**
     - Add a **Build command** in the Streamlit Cloud UI (if available) to run `python train_models.py` before starting the app.  
     (Streamlit Cloud does not always support a custom build step; if not, commit the trained artifacts.)

5. **Deploy**  
   Streamlit will build and host the app. One URL = frontend + backend.

**Note:** Streamlit Community Cloud has resource limits. If the app is heavy (large models/data), use Render instead.

---

## Deploy on Render (Streamlit as web service)

1. **New Web Service** on [Render](https://dashboard.render.com), connect your GitHub repo.

2. **Settings**
   - **Build command**:
     ```bash
     pip install -r requirements.txt && python train_models.py
     ```
   - **Start command**:
     ```bash
     streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
     ```
   - **Environment**: Add `PORT` if needed (Render usually sets it).

3. **Deploy**  
   One service runs the full Streamlit app (frontend + backend).

---

## Summary

| Item        | Detail                                                                 |
|------------|------------------------------------------------------------------------|
| **App file** | `streamlit_app.py`                                                     |
| **Backend**  | Same as Flask: `app.py` (models, DB, helpers)                         |
| **Run**      | `streamlit run streamlit_app.py`                                      |
| **Deploy**   | Streamlit Community Cloud or Render; single app = frontend + backend. |

You can keep both the **Flask** app (`app.py` + templates) and the **Streamlit** app (`streamlit_app.py`) in the same repo and choose which one to run or deploy.
