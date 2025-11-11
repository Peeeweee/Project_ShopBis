# ShopBis Dashboard 🛍️

A comprehensive Streamlit dashboard for shopping behavior analysis with ML-powered predictions and customer segmentation.

## 📊 Features

### 4 Interactive Pages:

1. **🏠 Overview** - Dataset snapshot with key metrics
   - Total customers, average spending, demographics
   - Interactive filters (gender, age, location, income)
   - Top 5 categories, location analysis
   - Gender and age distributions

2. **👥 Segmentation** - Customer clustering analysis
   - K-Means clustering visualization
   - Configurable features and cluster count
   - 2D and 3D scatter plots
   - Cluster profiles and insights
   - Automatic segment interpretation

3. **🔮 Prediction** - ML-powered category prediction
   - Single prediction interface
   - Batch prediction from CSV
   - Model performance metrics
   - Feature importance analysis
   - Uses trained Random Forest model

4. **📦 Product Insights** - Product trends and analytics
   - Top-selling categories and products
   - Performance by gender and age group
   - Seasonal trends analysis
   - Review ratings distribution
   - Discount and promotion impact
   - Payment method preferences

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Navigate to the Dashboard directory:
```bash
cd Dashboard
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Dashboard

1. Make sure you're in the Dashboard directory

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. The dashboard will open in your default browser at `http://localhost:8501`

### Alternative: Run with Python
```bash
python -m streamlit run app.py
```

## 🌐 Deployment to the Cloud

### Deploy to Streamlit Community Cloud (Recommended) ⭐

**Why Streamlit Cloud?**
- ✅ Free forever with generous limits
- ✅ Zero configuration needed
- ✅ Auto-deploy on git push
- ✅ Built-in HTTPS/SSL
- ✅ Optimized for Streamlit apps

**Quick Deploy Steps:**

1. **Push your code to GitHub:**
```bash
git add .
git commit -m "Deploy ShopBis Dashboard"
git push
```

2. **Go to [streamlit.io/cloud](https://streamlit.io/cloud)** and sign in with GitHub

3. **Click "New app"** and configure:
   - **Repository:** Your GitHub repo
   - **Branch:** `main`
   - **Main file path:** `Dashboard/app.py`

4. **Click "Deploy"** - Your app will be live in 2-3 minutes at `https://your-app-name.streamlit.app`

### Alternative Deployment Options

For detailed deployment instructions including **Render**, **Railway**, and **Heroku**, see the comprehensive [DEPLOYMENT.md](DEPLOYMENT.md) guide.

**Note:** Vercel is NOT recommended for Streamlit apps as it's designed for static sites and serverless functions, not persistent Python web applications.

## 📁 Project Structure

```
Dashboard/
│
├── app.py                          # Main dashboard page
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
└── pages/
    ├── 1_🏠_Overview.py           # Overview page
    ├── 2_👥_Segmentation.py        # Segmentation page
    ├── 3_🔮_Prediction.py          # Prediction page
    └── 4_📦_Product_Insights.py    # Product insights page
```

## 📊 Data Requirements

The dashboard expects the following data files in the parent directory:

- `../data/shopping_behavior_cleaned.csv` - Cleaned dataset
- `../Model/random_forest_model.joblib` - Trained ML model
- `../Model/label_encoders.joblib` - Label encoders
- `../Model/le_target.joblib` - Target encoder

Make sure you've run the data cleaning and model training scripts before using the dashboard.

## 🎯 Usage Tips

### Overview Page
- Use the sidebar filters to drill down into specific customer segments
- Compare filtered metrics against overall statistics
- Download filtered data for further analysis

### Segmentation Page
- Experiment with different features for clustering
- Try different numbers of clusters (2-6)
- Use 3D visualization for deeper insights
- Export clustered data for marketing campaigns

### Prediction Page
- Fill in customer details for single predictions
- Upload CSV files for batch predictions
- Review feature importance to understand model decisions
- Check model performance visualizations

### Product Insights Page
- Identify top-performing products and categories
- Analyze seasonal trends for inventory planning
- Understand customer preferences by demographics
- Export summaries for reporting

## 🔧 Configuration

### Customizing the Dashboard

You can customize various aspects:

1. **Colors**: Modify color schemes in plotly charts
2. **Metrics**: Add or remove KPIs in each page
3. **Filters**: Adjust filter options in the sidebar
4. **Charts**: Add new visualizations using plotly

### Adding New Features

1. Create a new page in `pages/` directory
2. Follow the naming convention: `N_emoji_PageName.py`
3. Use the same data loading pattern with `@st.cache_data`
4. Maintain consistent styling with other pages

## 📈 Performance

- Uses Streamlit's caching for optimal performance
- Data is loaded once and cached
- Model predictions are efficient with joblib
- Handles 3,900+ records smoothly

## 🐛 Troubleshooting

### Dashboard won't start
- Check if all dependencies are installed: `pip install -r requirements.txt`
- Verify Python version: `python --version` (should be 3.8+)

### Model not found error
- Run the model training script first: `python ../Model/category_prediction_model.py`
- Check if model files exist in the Model directory

### Data not loading
- Verify the data file path: `../data/shopping_behavior_cleaned.csv`
- Run the cleaning script if needed: `python ../Clean/clean_dataset.py`

### Charts not displaying
- Clear Streamlit cache: In the browser, click ⋮ menu → Clear Cache
- Restart the dashboard

## 🔄 Updates

To update the dashboard with new data:

1. Update the dataset in `data/shopping_behavior_cleaned.csv`
2. Retrain the model if needed: `python Model/category_prediction_model.py`
3. Restart the dashboard (it will reload the new data)

## 📚 Technologies Used

- **Streamlit** - Dashboard framework
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning & clustering
- **NumPy** - Numerical computations

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

To add new features or improvements:

1. Create a new branch
2. Add your changes
3. Test thoroughly
4. Submit a pull request

## 📧 Support

For issues or questions:
- Check the troubleshooting section above
- Review Streamlit documentation: https://docs.streamlit.io
- Open an issue on the project repository

---

**Happy Analyzing! 📊✨**

**Developed by Kent Paulo Delgado**

Built with ❤️ using Streamlit, Scikit-learn, Plotly & Python
