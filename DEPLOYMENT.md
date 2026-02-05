# Deployment Guide for Render

This guide will help you deploy the Tourism Experience Analytics application to Render.

## Prerequisites

1. A GitHub account
2. Your project pushed to a GitHub repository
3. A Render account (sign up at https://render.com)

## Step-by-Step Deployment

### 1. Prepare Your Repository

Make sure your project is pushed to GitHub with all necessary files:
- ✅ All Python files (`app.py`, `train_models.py`)
- ✅ `requirements.txt`
- ✅ `render.yaml` (optional but recommended)
- ✅ All data files in `data/raw/` (Excel files)
- ✅ Templates in `templates/`
- ✅ Static files in `static/`

### 2. Deploy Using render.yaml (Recommended)

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Click "New +"** → **"Blueprint"**
3. **Connect your GitHub repository**
4. **Render will automatically detect `render.yaml`** and configure the service
5. **Click "Apply"** to deploy

The `render.yaml` file will:
- Install dependencies
- Run `train_models.py` to train models
- Start the app with gunicorn
- Set up environment variables automatically

### 3. Deploy Manually (Alternative)

If you prefer manual setup:

1. **Go to Render Dashboard** → **"New +"** → **"Web Service"**
2. **Connect your GitHub repository**
3. **Configure the service**:
   - **Name**: `tourism-experience-analytics` (or your choice)
   - **Runtime**: `Python 3`
   - **Region**: Choose closest to your users
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: Leave empty (or specify if your app is in a subfolder)
   - **Environment**: `Python 3`
   - **Build Command**:
     ```bash
     pip install -r requirements.txt && python train_models.py
     ```
   - **Start Command**:
     ```bash
     gunicorn app:app
     ```
4. **Environment Variables** (optional but recommended):
   - `FLASK_SECRET_KEY`: Generate a secure random string (Render can auto-generate)
5. **Click "Create Web Service"**

### 4. Important Notes

#### Build Process
- The build command runs `train_models.py` which trains your ML models
- This may take 5-15 minutes depending on your dataset size
- Make sure all Excel files are in `data/raw/` directory

#### Database
- The app uses SQLite (`app.db`) which is stored in the filesystem
- **Note**: On Render's free tier, the filesystem is ephemeral (resets on deploy)
- For production, consider using Render PostgreSQL or another persistent database

#### Model Files
- Models are generated during build (`models/*.joblib`)
- These are stored in the filesystem
- If you want models to persist, commit them to Git or use persistent storage

#### Port Configuration
- Render automatically sets the `PORT` environment variable
- The app uses `gunicorn` which reads `PORT` automatically
- No manual port configuration needed

### 5. Post-Deployment

1. **Check Build Logs**: Ensure `train_models.py` completed successfully
2. **Check Runtime Logs**: Verify the app started without errors
3. **Test the Application**: Visit your Render URL (e.g., `https://tourism-experience-analytics.onrender.com`)
4. **Test Login**: Create an account and test predictions

### 6. Troubleshooting

#### Build Fails
- Check that all Excel files exist in `data/raw/`
- Verify `requirements.txt` has all dependencies
- Check build logs for specific error messages

#### App Crashes on Start
- Check runtime logs for errors
- Verify models were created during build
- Ensure database initialization succeeded

#### Models Not Found
- Make sure `train_models.py` ran successfully during build
- Check that `models/` directory exists and contains `.joblib` files

#### Database Issues
- SQLite database is created automatically on first run
- If you see database errors, check file permissions

### 7. Updating Your Deployment

1. **Push changes to GitHub**
2. **Render will automatically detect and redeploy**
3. **Note**: Models will be retrained on each deploy (if build command includes `train_models.py`)

### 8. Optional: Persistent Database

For production, consider migrating to PostgreSQL:

1. **Create a PostgreSQL database** in Render
2. **Update `app.py`** to use PostgreSQL instead of SQLite
3. **Set connection string** as environment variable

## Support

If you encounter issues:
1. Check Render logs (Build Logs and Runtime Logs)
2. Verify all files are committed to Git
3. Test locally first: `python train_models.py && python app.py`

## Render Free Tier Limitations

- **Sleeps after 15 minutes** of inactivity (wakes on next request)
- **Ephemeral filesystem** (resets on deploy)
- **512 MB RAM** limit
- **100 GB bandwidth** per month

For production use, consider upgrading to a paid plan.
