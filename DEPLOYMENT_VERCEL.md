# Deploy Tourism Experience Analytics on Vercel

This guide walks you through deploying the Flask app to Vercel.

## Prerequisites

- GitHub account
- Project pushed to a GitHub repository
- [Vercel account](https://vercel.com/signup)
- All Excel files in `data/raw/` committed to the repo

## Important: Vercel vs This Project

- **Serverless**: The app runs as a serverless function. Cold starts may add 1–2 seconds to the first request.
- **Bundle size**: Python dependencies (pandas, scikit-learn, etc.) are large. The deployment must stay under **250 MB** (uncompressed). If the build fails with size errors, consider using Render or a VPS instead.
- **SQLite**: The app uses SQLite. On Vercel, the database is stored in `/tmp`, so **data is not persistent** across deployments or function instances. For persistent users and history, use a hosted database (e.g. Vercel Postgres, Supabase) and change the app to use it.
- **Models**: `train_models.py` runs at **build time**. Trained models are included in the deployment. The build may take several minutes.

## Step-by-Step Deployment

### 1. Prepare the repository

Ensure these are in the repo:

- `app.py`
- `train_models.py`
- `vercel.json`
- `requirements.txt`
- `data/raw/*.xlsx`
- `templates/`
- `static/`

### 2. Deploy with Vercel Dashboard

1. Go to [vercel.com/new](https://vercel.com/new).
2. **Import** your GitHub repository.
3. **Configure**:
   - **Framework Preset**: Vercel should detect Flask (or leave as “Other”).
   - **Build Command**: (optional; already in `vercel.json`)
     ```bash
     pip install -r requirements.txt && python train_models.py
     ```
   - **Install Command**: `pip install -r requirements.txt` (or leave default).
4. Click **Deploy**.

Vercel will install dependencies, run `train_models.py`, and deploy the Flask app.

### 3. Deploy with Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# From project root
cd "F:\projects\Tourism Experience Analytics"
vercel login
vercel
```

Follow the prompts and deploy. For production:

```bash
vercel --prod
```

### 4. Environment variables (optional)

In **Project → Settings → Environment Variables** add:

- **FLASK_SECRET_KEY**: A long random string (e.g. from `openssl rand -hex 32`).

If you don’t set it, the app falls back to a default (less secure for production).

### 5. After deployment

- Open the URL Vercel gives you (e.g. `https://your-project.vercel.app`).
- You should be redirected to the login page.
- Create an account and test predictions.

Note: On Vercel, SQLite is in `/tmp`, so accounts and history are not kept across deploys or different function instances. For a real product, switch to a hosted database.

## vercel.json

The project includes a `vercel.json` that:

- Sets the **build command** to install dependencies and run `train_models.py`.
- Uses the **Flask** framework so Vercel can run the app as a serverless function.

You can override **Build Command** or **Install Command** in the Vercel project settings; those override `vercel.json`.

## Static files (CSS/JS)

Flask serves files from the `static/` folder. If you prefer to serve them from Vercel’s CDN:

1. Create a `public` directory.
2. Copy contents of `static/` into `public/` (e.g. `public/css/`, `public/js/`).
3. In your templates, reference `/css/style.css` and `/js/main.js` instead of `url_for('static', ...)`.

## Troubleshooting

### Build fails: “Module not found” or dependency errors

- Ensure `requirements.txt` lists every import used in `app.py` and `train_models.py`.
- In Vercel, check the **Build Logs** for the exact error.

### Build fails: bundle too large

- Python + pandas + scikit-learn can exceed 250 MB.
- Options: remove unused dependencies, use a lighter stack, or deploy to Render/Railway instead.

### “Models not found” or similar at runtime

- `train_models.py` must run during the build and write to `data/processed/` and `models/`.
- Ensure all `data/raw/*.xlsx` files are committed so the build can run successfully.
- Check build logs to confirm `train_models.py` completed without errors.

### Login / database issues

- On Vercel, the DB is in `/tmp` and is not persistent. Use a hosted database for production.

### Cold starts

- First request after idle can be slow. Consider keeping the app warm or moving to a long-running server (e.g. Render) if latency is critical.

## Summary

| Item              | Notes                                                |
|-------------------|------------------------------------------------------|
| **Build**         | `pip install -r requirements.txt && python train_models.py` |
| **Start**         | Handled by Vercel (Flask as serverless function)     |
| **Database**      | SQLite in `/tmp` (not persistent)                     |
| **Static files**  | Served by Flask from `static/`                       |
| **Limits**        | 250 MB bundle; consider Render if you hit limits   |

For a production setup with persistent data and no strict serverless limits, **Render** is a better fit for this app. Use Vercel when you want quick, serverless deployment and accept cold starts and non-persistent SQLite.
