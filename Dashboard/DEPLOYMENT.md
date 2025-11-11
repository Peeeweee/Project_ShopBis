# 🚀 ShopBis Dashboard Deployment Guide

This guide will help you deploy the ShopBis Analytics Dashboard to the cloud.

---

## 📋 Table of Contents

1. [Streamlit Community Cloud (Recommended)](#streamlit-community-cloud-recommended-)
2. [Render](#render)
3. [Railway](#railway)
4. [Heroku](#heroku)
5. [Troubleshooting](#troubleshooting)

---

## Streamlit Community Cloud (Recommended) ⭐

**Best for:** Free hosting specifically optimized for Streamlit apps

### Prerequisites
- GitHub account
- Your code pushed to a GitHub repository

### Steps

#### 1. Push Your Code to GitHub

If you haven't already, initialize git and push to GitHub:

```bash
# Navigate to the Dashboard directory
cd c:\Users\Paulo\Documents\Project_ShopBis

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit changes
git commit -m "Deploy ShopBis Dashboard"

# Set main branch
git branch -M main

# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

#### 2. Deploy on Streamlit Community Cloud

1. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)

2. Click **"Sign in"** with your GitHub account

3. Click **"New app"** button

4. Fill in the deployment form:
   - **Repository:** Select your GitHub repository
   - **Branch:** `main` (or your default branch)
   - **Main file path:** `Dashboard/app.py`
   - **App URL:** Choose your custom URL (e.g., `shopbis-analytics`)

5. Click **"Deploy!"**

6. Wait 2-3 minutes for deployment

7. Your app will be live at: `https://YOUR-APP-NAME.streamlit.app`

### Features
✅ **Free forever** with generous limits
✅ **Auto-deploy** on git push
✅ **Built-in SSL** (HTTPS)
✅ **No configuration** needed
✅ **Community support**

### Limits (Free Tier)
- 1 GB RAM per app
- Unlimited public apps
- Community support only

---

## Render

**Best for:** Free tier with more control and custom domains

### Prerequisites
- Render account
- GitHub account

### Steps

#### 1. Create Procfile

Create a file named `Procfile` in the Dashboard directory:

```bash
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

#### 2. Push to GitHub

```bash
git add Procfile
git commit -m "Add Procfile for Render deployment"
git push
```

#### 3. Deploy on Render

1. Go to [Render.com](https://render.com) and sign in

2. Click **"New +"** → **"Web Service"**

3. Connect your GitHub repository

4. Configure the service:
   - **Name:** `shopbis-dashboard`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

5. Click **"Create Web Service"**

6. Wait for deployment (3-5 minutes)

7. Your app will be live at: `https://shopbis-dashboard.onrender.com`

### Features
✅ Free tier available (with limits)
✅ Custom domains (paid plans)
✅ Auto-deploy on git push
✅ Built-in SSL
✅ More control than Streamlit Cloud

### Limits (Free Tier)
- 512 MB RAM
- Spins down after 15 minutes of inactivity
- Slower cold starts

---

## Railway

**Best for:** Simple deployment with great developer experience

### Prerequisites
- Railway account
- GitHub account

### Steps

#### 1. Push to GitHub (if not done)

```bash
git add .
git commit -m "Deploy to Railway"
git push
```

#### 2. Deploy on Railway

1. Go to [Railway.app](https://railway.app) and sign in

2. Click **"New Project"**

3. Select **"Deploy from GitHub repo"**

4. Select your repository

5. Railway will auto-detect Python and install dependencies

6. Click **"Add variables"** and add:
   - `PORT`: `8501`

7. In **Settings** → **Domains**, generate a domain

8. Your app will be live at: `https://shopbis-dashboard.up.railway.app`

### Features
✅ $5 free credit per month
✅ Easy deployment
✅ Great developer experience
✅ Automatic HTTPS
✅ Good performance

### Limits (Free Tier)
- $5/month credit (500 hours)
- No credit card required

---

## Heroku

**Best for:** Enterprise deployments (paid)

### Prerequisites
- Heroku account
- Heroku CLI installed
- GitHub account

### Steps

#### 1. Create Heroku Config Files

Create `Procfile` in Dashboard directory:
```
web: sh setup.sh && streamlit run app.py
```

Create `setup.sh` in Dashboard directory:
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

#### 2. Deploy to Heroku

```bash
# Login to Heroku
heroku login

# Create Heroku app
heroku create shopbis-dashboard

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# Open app
heroku open
```

### Features
✅ Enterprise-grade hosting
✅ Great documentation
✅ Add-ons ecosystem
✅ Good performance
✅ Excellent support

### Costs
⚠️ **No free tier** anymore
- Basic plans start at $5-7/month
- Production plans $25+/month

---

## ⚙️ Environment Variables

If you need to use environment variables (for API keys, secrets, etc.):

### Streamlit Community Cloud
1. Go to your app settings
2. Click "Secrets"
3. Add your secrets in TOML format:
```toml
API_KEY = "your-api-key"
DATABASE_URL = "your-database-url"
```

### Render / Railway / Heroku
Add environment variables in the dashboard under "Environment Variables" or "Config Vars"

---

## 📊 Monitoring

### Streamlit Community Cloud
- View logs in the app menu
- Check resource usage in settings

### Render
- View logs in the dashboard
- Monitor resource usage

### Railway
- View logs in real-time
- Monitor deployments

### Heroku
```bash
# View logs
heroku logs --tail

# Check app status
heroku ps
```

---

## 🐛 Troubleshooting

### App Won't Start

**Check logs:**
- Streamlit Cloud: Click "Manage app" → "Logs"
- Render: Dashboard → Logs tab
- Railway: Dashboard → Deployments
- Heroku: `heroku logs --tail`

**Common issues:**
1. **Missing dependencies**: Check `requirements.txt`
2. **Wrong file path**: Ensure main file path is correct
3. **Data file not found**: Make sure `data/shopping_behavior_cleaned.csv` exists
4. **Port issues**: Ensure PORT variable is set correctly

### App is Slow

**Solutions:**
1. Use `@st.cache_data` for data loading (already implemented)
2. Reduce data size if possible
3. Upgrade to paid tier for more resources

### Module Not Found Error

**Fix:**
```bash
# Update requirements.txt with exact versions
pip freeze > requirements.txt

# Commit and push
git add requirements.txt
git commit -m "Update dependencies"
git push
```

### Data File Issues

**Fix:**
- Ensure `shopping_behavior_cleaned.csv` is in the `data/` folder
- Check file path in code: `Path(__file__).parent.parent / "data" / "shopping_behavior_cleaned.csv"`
- Verify file is committed to git (not in `.gitignore`)

---

## 🔄 Updating Your Deployed App

All platforms support auto-deployment from git:

```bash
# Make your changes locally

# Test locally first
streamlit run app.py

# Commit changes
git add .
git commit -m "Update dashboard features"

# Push to trigger auto-deployment
git push
```

The app will automatically redeploy with your changes!

---

## 📈 Recommended Deployment

For **ShopBis Analytics Dashboard**, we recommend:

### 🥇 **Streamlit Community Cloud** (Best Choice)

**Why:**
- ✅ Free forever
- ✅ Built specifically for Streamlit
- ✅ Zero configuration
- ✅ Auto-deploy on push
- ✅ Perfect for this project

**Steps:**
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy in 2 minutes

**Perfect for:**
- Portfolio projects
- Demos
- Educational projects
- Small teams

---

## 🎉 After Deployment

Once your app is deployed:

1. **Share the URL** with your team or on your resume/portfolio

2. **Monitor usage** through the platform dashboard

3. **Keep improving** - push updates anytime

4. **Add to README** - Update the main README with your live URL

---

## 📝 Deployment Checklist

Before deploying, ensure:

- [ ] All code is committed to git
- [ ] `requirements.txt` includes all dependencies
- [ ] Data files are in the correct location
- [ ] `.gitignore` excludes unnecessary files
- [ ] App runs successfully locally (`streamlit run app.py`)
- [ ] GitHub repository is public (or private with proper access)
- [ ] Main file path is correct (`Dashboard/app.py`)
- [ ] Sensitive data is removed (use environment variables for secrets)

---

## 🆘 Getting Help

If you encounter issues:

1. **Check the logs** on your deployment platform
2. **Search the issue** on Streamlit forums or Stack Overflow
3. **Read the docs**:
   - [Streamlit Deployment Docs](https://docs.streamlit.io/streamlit-community-cloud)
   - [Render Docs](https://render.com/docs)
   - [Railway Docs](https://docs.railway.app)
4. **Open an issue** on GitHub

---

## 🎊 Success!

Your ShopBis Analytics Dashboard is now live on the web! 🚀

Share it with the world and showcase your data science & ML skills!

---

**Developed by Kent Paulo Delgado**

*Happy Deploying! 🎉*
