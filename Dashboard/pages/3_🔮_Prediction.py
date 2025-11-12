"""
Purchase Behavior Prediction Page - ShopBis Dashboard
======================================================
AI-powered prediction: Will this customer buy again?
Uses Random Forest ML model with 97.82% accuracy
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Purchase Behavior Prediction - ShopBis",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply professional theme
professional_css = Path(__file__).parent.parent / "shared_styles.txt"
if professional_css.exists():
    st.markdown(f"<style>{professional_css.read_text()}</style>", unsafe_allow_html=True)

# Additional page-specific styles
st.markdown("""
<style>
    /* Hide Streamlit Branding and Header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden !important;}

    /* Hide Streamlit's default header toolbar */
    [data-testid="stHeader"] {
        display: none !important;
    }

    /* Hide the top decoration bar */
    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* Remove default Streamlit padding */
    .main .block-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Remove top app container padding */
    .stApp > header {
        display: none !important;
    }

    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 1rem;
        padding: 2.5rem;
        border: 2px solid #E9ECEF;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
        margin: 1.5rem 0;
    }

    /* Result Card - Success (Yes) */
    .result-card-yes {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        border-radius: 1rem;
        padding: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
        margin: 2rem 0;
    }

    /* Result Card - Warning (No) */
    .result-card-no {
        background: linear-gradient(135deg, #DD0303 0%, #dc3545 100%);
        border-radius: 1rem;
        padding: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(221, 3, 3, 0.4);
        margin: 2rem 0;
    }

    /* Section Header */
    .section-header-custom {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 1rem;
        border-bottom: 3px solid #FA812F;
    }

    .section-header-custom h2 {
        margin: 0;
        color: #2C3E50;
        font-size: 1.8rem;
        font-weight: 700;
    }

    .section-header-custom span {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Add spacer at the very top
st.markdown("""
<div style="height: 1.5rem; background: #F8F9FA;"></div>
""", unsafe_allow_html=True)

# Top Navigation
st.markdown("""
<div class="top-nav">
    <div class="top-nav-logo">
        <span class="emoji">🛍️</span>
        <h1>Shop_Bis</h1>
    </div>
    <div class="top-nav-links">
        <a href="/" target="_self" class="nav-link">
            <span>🏠</span> Home & Overview
        </a>
        <a href="/Segmentation" target="_self" class="nav-link">
            <span>👥</span> Segmentation
        </a>
        <a href="/Prediction" target="_self" class="nav-link active">
            <span>🔮</span> Prediction
        </a>
        <a href="/Product_Insights" target="_self" class="nav-link">
            <span>📦</span> Product Insights
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div style="background: white; padding: 2.5rem 2rem; margin: 1rem -2rem 2rem -2rem; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);">
    <div style="text-align: center;">
        <h1 style="font-size: 3rem; font-weight: 800; color: #2C3E50; margin-bottom: 0.5rem; letter-spacing: -1px;">
            🔮 Purchase Behavior Prediction
        </h1>
        <p style="font-size: 1.1rem; color: #6C757D; max-width: 700px; margin: 0 auto;">
            Predict if a customer will make repeat purchases using AI-powered Random Forest model
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Load the model and encoders
@st.cache_resource
def load_model():
    """Load the trained purchase behavior model and encoders"""
    try:
        model_dir = Path(__file__).parent.parent.parent / "Model" / "saved_models"

        model = joblib.load(model_dir / "purchase_behavior_model.joblib")
        label_encoders = joblib.load(model_dir / "behavior_label_encoders.joblib")
        target_encoder = joblib.load(model_dir / "behavior_target_encoder.joblib")

        return model, label_encoders, target_encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, label_encoders, target_encoder = load_model()

if model is None:
    st.error("⚠️ Model not found! Please train the model first by running: `python Model/purchase_behavior_model.py`")
    st.stop()

# Load data for reference
@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent.parent / "data" / "shopping_behavior_cleaned.csv"
    return pd.read_csv(data_path)

df = load_data()

# Section: Make a Prediction
st.markdown("""
<div class="section-header-custom">
    <span>🎯</span>
    <h2>Make a Prediction</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #FEF3E2 0%, #ffffff 100%); padding: 1.5rem; border-radius: 0.75rem; border-left: 4px solid #FA812F; margin-bottom: 2rem;">
    <p style="color: #495057; margin: 0; line-height: 1.6;">
        Enter customer and purchase details below to predict the likelihood of repeat purchases.
        Our AI model analyzes <strong>10 key factors</strong> to provide a confidence score.
    </p>
</div>
""", unsafe_allow_html=True)

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem;">
        <h3 style="color: #2C3E50; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
            <span>👤</span> Customer Information
        </h3>
    </div>
    """, unsafe_allow_html=True)

    # Customer Demographics
    age = st.slider("Age", min_value=18, max_value=70, value=35, help="Customer's age")
    gender = st.selectbox("Gender", options=sorted(df['Gender'].unique()), help="Customer's gender")

with col2:
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem;">
        <h3 style="color: #2C3E50; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
            <span>🛒</span> Purchase Information
        </h3>
    </div>
    """, unsafe_allow_html=True)

    # Purchase Details
    category = st.selectbox("Product Category", options=sorted(df['Category'].unique()),
                           help="Type of product purchased")
    season = st.selectbox("Season", options=sorted(df['Season'].unique()),
                         help="Season of purchase")

# Additional Features
st.markdown("""
<div style="background: white; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem; margin-top: 1rem;">
    <h3 style="color: #2C3E50; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
        <span>📊</span> Additional Details
    </h3>
</div>
""", unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)

with col3:
    review_rating = st.slider("Review Rating", min_value=1.0, max_value=5.0, value=3.5, step=0.1,
                             help="Customer's review rating (1-5 stars)")
    size = st.selectbox("Product Size", options=sorted(df['Size'].unique()),
                       help="Size of the product")

with col4:
    shipping_type = st.selectbox("Shipping Type", options=sorted(df['Shipping Type'].unique()),
                                help="Shipping method chosen")
    purchase_amount = st.number_input("Purchase Amount (USD)", min_value=20, max_value=100, value=50,
                                     help="Total purchase amount in USD")

with col5:
    discount_applied = st.selectbox("Discount Applied", options=['Yes', 'No'],
                                   help="Was a discount applied?")
    promo_code = st.selectbox("Promo Code Used", options=['Yes', 'No'],
                             help="Did customer use a promo code?")

# Predict Button
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮 Predict Purchase Behavior", use_container_width=True, type="primary"):
    # Prepare input data
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Category': [category],
        'Season': [season],
        'Review Rating': [review_rating],
        'Size': [size],
        'Shipping Type': [shipping_type],
        'Purchase Amount (USD)': [purchase_amount],
        'Discount Applied': [discount_applied],
        'Promo Code Used': [promo_code]
    })

    # Encode categorical features
    for col in input_data.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            try:
                input_data[col] = label_encoders[col].transform(input_data[col])
            except ValueError:
                st.error(f"⚠️ Unknown value for {col}. Please select a valid option.")
                st.stop()

    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # Get prediction label
    prediction_label = target_encoder.inverse_transform([prediction])[0]
    confidence = prediction_proba[prediction] * 100

    # Show balloons for success
    if prediction_label == 'Yes':
        st.balloons()

    # Display result
    st.markdown("<br>", unsafe_allow_html=True)

    if prediction_label == 'Yes':
        st.markdown(f"""
        <div class="result-card-yes">
            <div style="font-size: 3rem; margin-bottom: 1rem;">✅</div>
            <h2 style="margin: 0 0 1rem 0; font-size: 2.5rem; font-weight: 800;">YES - Likely to Buy Again!</h2>
            <p style="font-size: 1.2rem; margin: 0.5rem 0; opacity: 0.95;">
                This customer shows strong indicators of repeat purchase behavior
            </p>
            <div style="font-size: 3rem; font-weight: 800; margin: 1rem 0;">{confidence:.1f}%</div>
            <p style="font-size: 1rem; margin: 0; opacity: 0.9;">Confidence Score</p>
        </div>
        """, unsafe_allow_html=True)

        # Insights
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 0.75rem; box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);">
            <h3 style="color: #28a745; margin-top: 0;">💡 Business Insights</h3>
            <ul style="color: #495057; line-height: 2;">
                <li><strong>High retention potential</strong> - Consider loyalty programs</li>
                <li><strong>Target for upselling</strong> - Recommend premium products</li>
                <li><strong>Email marketing</strong> - Send personalized offers</li>
                <li><strong>Request reviews</strong> - Leverage positive sentiment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="result-card-no">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
            <h2 style="margin: 0 0 1rem 0; font-size: 2.5rem; font-weight: 800;">NO - May Not Return</h2>
            <p style="font-size: 1.2rem; margin: 0.5rem 0; opacity: 0.95;">
                This customer shows low indicators of repeat purchase behavior
            </p>
            <div style="font-size: 3rem; font-weight: 800; margin: 1rem 0;">{confidence:.1f}%</div>
            <p style="font-size: 1rem; margin: 0; opacity: 0.9;">Confidence Score</p>
        </div>
        """, unsafe_allow_html=True)

        # Insights
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 0.75rem; box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);">
            <h3 style="color: #dc3545; margin-top: 0;">💡 Retention Strategies</h3>
            <ul style="color: #495057; line-height: 2;">
                <li><strong>Win-back campaign</strong> - Send special discount offers</li>
                <li><strong>Feedback request</strong> - Understand dissatisfaction</li>
                <li><strong>Product recommendations</strong> - Suggest alternatives</li>
                <li><strong>Customer service</strong> - Proactive support outreach</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Detailed Probability Breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header-custom">
        <span>📊</span>
        <h2>Prediction Breakdown</h2>
    </div>
    """, unsafe_allow_html=True)

    col_yes, col_no = st.columns(2)

    with col_yes:
        yes_prob = prediction_proba[1] * 100 if len(prediction_proba) > 1 else 0
        st.metric(
            label="💚 Probability: Will Buy Again (Yes)",
            value=f"{yes_prob:.1f}%",
            delta="High Confidence" if yes_prob >= 70 else ("Medium" if yes_prob >= 50 else "Low")
        )

    with col_no:
        no_prob = prediction_proba[0] * 100 if len(prediction_proba) > 0 else 0
        st.metric(
            label="❌ Probability: Won't Buy Again (No)",
            value=f"{no_prob:.1f}%",
            delta="High Risk" if no_prob >= 70 else ("Medium" if no_prob >= 50 else "Low"),
            delta_color="inverse"
        )

# Model Information Section
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="section-header-custom">
    <span>🤖</span>
    <h2>Model Information</h2>
</div>
""", unsafe_allow_html=True)

col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 0.75rem; box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05); text-align: center;">
        <div style="font-size: 2.5rem; font-weight: 700; color: #FA812F;">97.82%</div>
        <p style="color: #6C757D; margin: 0.5rem 0 0 0;">Model Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 0.75rem; box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05); text-align: center;">
        <div style="font-size: 2.5rem; font-weight: 700; color: #FA812F;">10</div>
        <p style="color: #6C757D; margin: 0.5rem 0 0 0;">Features Used</p>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 0.75rem; box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05); text-align: center;">
        <div style="font-size: 2.5rem; font-weight: 700; color: #FA812F;">200</div>
        <p style="color: #6C757D; margin: 0.5rem 0 0 0;">Decision Trees</p>
    </div>
    """, unsafe_allow_html=True)

# Feature Importance
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="background: white; padding: 2rem; border-radius: 0.75rem; box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);">
    <h3 style="color: #2C3E50; margin-top: 0;">🎯 Top 5 Most Important Features</h3>
    <ol style="color: #495057; line-height: 2; font-size: 1rem;">
        <li><strong>Purchase Amount (USD)</strong> - 23.41% importance</li>
        <li><strong>Age</strong> - 21.69% importance</li>
        <li><strong>Review Rating</strong> - 16.79% importance</li>
        <li><strong>Shipping Type</strong> - 10.25% importance</li>
        <li><strong>Season</strong> - 7.42% importance</li>
    </ol>
    <p style="color: #6C757D; margin: 1rem 0 0 0; font-size: 0.9rem;">
        <em>These features have the strongest influence on predicting repeat purchase behavior.</em>
    </p>
</div>
""", unsafe_allow_html=True)

# Batch Prediction Section
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="section-header-custom">
    <span>📤</span>
    <h2>Batch Prediction</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #FEF3E2 0%, #ffffff 100%); padding: 1.5rem; border-radius: 0.75rem; border-left: 4px solid #FA812F; margin-bottom: 1.5rem;">
    <p style="color: #495057; margin: 0; line-height: 1.6;">
        Upload a CSV file with multiple customer records to get predictions in bulk.
        The file should contain columns: <code>Age, Gender, Category, Season, Review Rating, Size, Shipping Type, Purchase Amount (USD), Discount Applied, Promo Code Used</code>
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload CSV File", type=['csv'], help="Upload a CSV file with customer data")

if uploaded_file is not None:
    try:
        # Read uploaded file
        batch_df = pd.read_csv(uploaded_file)

        st.success(f"✅ File uploaded successfully! Found {len(batch_df)} records.")

        # Show preview
        with st.expander("👁️ Preview Data", expanded=True):
            st.dataframe(batch_df.head(10), use_container_width=True)

        if st.button("🚀 Run Batch Predictions", use_container_width=True):
            # Prepare data
            batch_input = batch_df.copy()

            # Encode categorical features
            for col in batch_input.select_dtypes(include=['object']).columns:
                if col in label_encoders:
                    batch_input[col] = label_encoders[col].transform(batch_input[col])

            # Make predictions
            with st.spinner("🔮 Generating predictions..."):
                predictions = model.predict(batch_input)
                predictions_proba = model.predict_proba(batch_input)

                # Add results to original dataframe
                batch_df['Prediction'] = target_encoder.inverse_transform(predictions)
                batch_df['Confidence'] = [proba[pred] * 100 for pred, proba in zip(predictions, predictions_proba)]
                batch_df['Will_Buy_Again_Probability'] = predictions_proba[:, 1] * 100

            st.success("✅ Predictions complete!")

            # Summary metrics
            col_b1, col_b2, col_b3 = st.columns(3)

            with col_b1:
                yes_count = (batch_df['Prediction'] == 'Yes').sum()
                st.metric("💚 Will Buy Again", f"{yes_count}", f"{yes_count/len(batch_df)*100:.1f}%")

            with col_b2:
                no_count = (batch_df['Prediction'] == 'No').sum()
                st.metric("❌ Won't Buy Again", f"{no_count}", f"{no_count/len(batch_df)*100:.1f}%")

            with col_b3:
                avg_confidence = batch_df['Confidence'].mean()
                st.metric("📊 Avg Confidence", f"{avg_confidence:.1f}%")

            # Show results
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📋 Prediction Results")
            st.dataframe(batch_df, use_container_width=True)

            # Download results
            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name="purchase_behavior_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #1a252f 0%, #2C3E50 100%); border-radius: 0.75rem; color: white;">
    <p style="color: #cbd5e1; font-size: 0.9rem; margin: 0;">
        <strong>ShopBis Purchase Behavior Prediction</strong> | Powered by Random Forest ML (97.82% Accuracy)
    </p>
    <p style="color: #FAB12F; font-size: 0.9rem; margin-top: 0.5rem; font-weight: 600;">
        Developed by Kent Paulo Delgado
    </p>
</div>
""", unsafe_allow_html=True)
