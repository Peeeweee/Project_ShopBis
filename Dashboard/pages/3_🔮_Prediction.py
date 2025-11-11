"""
Prediction Page - ShopBis Dashboard
====================================
ML-powered category prediction using the trained Random Forest model
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(page_title="Prediction - ShopBis", page_icon="🔮", layout="wide", initial_sidebar_state="collapsed")

# Apply professional theme
professional_css = Path(__file__).parent.parent / "shared_styles.txt"
if professional_css.exists():
    st.markdown(f"<style>{professional_css.read_text()}</style>", unsafe_allow_html=True)

# Additional page-specific styles
st.markdown("""
<style>
    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 1rem;
        padding: 2.5rem;
        border: 2px solid #E9ECEF;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
        margin: 1.5rem 0;
    }

    /* Input Section */
    .input-section {
        background: white;
        border-radius: 1rem;
        padding: 2rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #E9ECEF;
        margin-bottom: 1.5rem;
    }

    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, #FA812F 0%, #DD0303 100%);
        border-radius: 1rem;
        padding: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 6px 20px rgba(250, 129, 47, 0.4);
        margin: 2rem 0;
    }

    /* Section Header */
    .section-header-pred {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 1rem;
        border-bottom: 3px solid #FA812F;
    }

    .section-header-pred h2 {
        margin: 0;
        color: #2C3E50;
        font-weight: 700;
    }

    .section-header-pred .icon {
        font-size: 2rem;
        filter: drop-shadow(0 2px 4px rgba(250, 129, 47, 0.3));
    }
</style>
""", unsafe_allow_html=True)

# Top Navigation Bar
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
<div style="text-align: center; padding: 6.5rem 0 1.5rem 0;">
    <h1 style="font-size: 3rem; font-weight: 800; color: #2C3E50; margin-bottom: 0.5rem; letter-spacing: -1px;">
        🔮 AI Category Prediction
    </h1>
    <p style="font-size: 1.1rem; color: #6C757D; max-width: 700px; margin: 0 auto;">
        Predict product categories using advanced Random Forest machine learning
    </p>
</div>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_resource
def load_model_artifacts():
    model_path = Path(__file__).parent.parent.parent / "Model"
    try:
        model = joblib.load(model_path / "random_forest_model.joblib")
        label_encoders = joblib.load(model_path / "label_encoders.joblib")
        le_target = joblib.load(model_path / "le_target.joblib")
        return model, label_encoders, le_target
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent.parent / "data" / "shopping_behavior_cleaned.csv"
    return pd.read_csv(data_path)

model, label_encoders, le_target = load_model_artifacts()
df = load_data()

if model is None:
    st.error("⚠️ Model not found! Please train the model first by running Model/category_prediction_model.py")
    st.stop()

st.markdown("---")

# Model Overview
st.markdown("""
<div class="section-header-pred">
    <span class="icon">🤖</span>
    <h2>Model Overview</h2>
</div>
""", unsafe_allow_html=True)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric(
        label="🔬 Algorithm",
        value="Random Forest",
        delta="Ensemble Method"
    )

with metric_col2:
    st.metric(
        label="🎯 Target",
        value="Category",
        delta="4 classes"
    )

with metric_col3:
    st.metric(
        label="📊 Features",
        value="7",
        delta="Input variables"
    )

with metric_col4:
    st.metric(
        label="📈 Accuracy",
        value="99%",
        delta="High performance"
    )

st.markdown("---")

# Prediction Interface
st.markdown("""
<div class="section-header-pred">
    <span class="icon">🎯</span>
    <h2>Make a Prediction</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #FEF3E2 0%, #ffffff 100%); padding: 1.5rem; border-radius: 0.75rem; border-left: 4px solid #FA812F; margin-bottom: 2rem;">
    <p style="margin: 0; color: #2C3E50; font-weight: 500;">
        Enter customer and product details below to predict the product category. Our AI model will analyze the inputs and provide a confidence score.
    </p>
</div>
""", unsafe_allow_html=True)

# Input Form
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("#### 👤 Customer Information")

    gender = st.selectbox(
        "Gender",
        options=df["Gender"].unique().tolist(),
        help="Select customer gender",
        key="gender_input"
    )

    age = st.slider(
        "Age",
        min_value=int(df["Age"].min()),
        max_value=int(df["Age"].max()),
        value=int(df["Age"].mean()),
        help="Customer age",
        key="age_input"
    )

    review_rating = st.slider(
        "Expected Review Rating",
        min_value=float(df["Review Rating"].min()),
        max_value=float(df["Review Rating"].max()),
        value=float(df["Review Rating"].mean()),
        step=0.1,
        help="Predicted product rating",
        key="rating_input"
    )

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("#### 📦 Product Information")

    item_purchased = st.selectbox(
        "Item Name",
        options=sorted(df["Item Purchased"].unique().tolist()),
        help="Select product item",
        key="item_input"
    )

    color = st.selectbox(
        "Product Color",
        options=sorted(df["Color"].unique().tolist()),
        help="Select product color",
        key="color_input"
    )

    size = st.selectbox(
        "Product Size",
        options=df["Size"].unique().tolist(),
        help="Select product size",
        key="size_input"
    )

    purchase_amount = st.number_input(
        "Purchase Amount (USD)",
        min_value=int(df["Purchase Amount (USD)"].min()),
        max_value=int(df["Purchase Amount (USD)"].max()),
        value=int(df["Purchase Amount (USD)"].mean()),
        help="Transaction amount",
        key="amount_input"
    )

    st.markdown('</div>', unsafe_allow_html=True)

# Prediction Button
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
    # Prepare input data
    input_data = {
        'Item Purchased': item_purchased,
        'Color': color,
        'Size': size,
        'Gender': gender,
        'Age': age,
        'Purchase Amount (USD)': purchase_amount,
        'Review Rating': review_rating
    }

    # Create DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    try:
        for feature in ['Item Purchased', 'Color', 'Size', 'Gender']:
            if feature in label_encoders:
                le = label_encoders[feature]
                try:
                    input_df[feature] = le.transform(input_df[feature])
                except ValueError:
                    st.warning(f"⚠️ '{input_data[feature]}' is a new value for {feature}. Using fallback.")
                    input_df[feature] = le.transform([le.classes_[0]])

        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        # Get predicted category
        predicted_category = le_target.inverse_transform([prediction])[0]
        confidence = prediction_proba[prediction] * 100

        # Display Results with Animation
        st.balloons()

        # Result Header
        st.markdown(f"""
        <div class="result-card">
            <h2 style="color: white; margin: 0 0 1rem 0; font-size: 2.5rem;">
                🎯 Prediction Complete!
            </h2>
            <div style="font-size: 3rem; font-weight: 800; margin: 1rem 0;">
                {predicted_category}
            </div>
            <div style="font-size: 1.5rem; opacity: 0.95; margin-top: 1rem;">
                Confidence: {confidence:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Detailed Results
        st.markdown("---")

        # Results Metrics
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric(
                label="🎯 Predicted Category",
                value=predicted_category,
                delta="AI Prediction"
            )

        with res_col2:
            st.metric(
                label="📈 Confidence Score",
                value=f"{confidence:.2f}%",
                delta="Model Certainty"
            )

        with res_col3:
            reliability = "High" if confidence >= 80 else ("Medium" if confidence >= 60 else "Low")
            st.metric(
                label="🎲 Reliability",
                value=reliability,
                delta=f"Based on {confidence:.0f}%"
            )

        # Probability Distribution
        st.markdown("""
        <div class="section-header-pred">
            <span class="icon">📊</span>
            <h2>Probability Distribution</h2>
        </div>
        """, unsafe_allow_html=True)

        categories = le_target.classes_
        probabilities = prediction_proba * 100

        # Create enhanced bar chart
        fig_proba = go.Figure(data=[
            go.Bar(
                x=categories,
                y=probabilities,
                marker_color=['#DD0303' if i == prediction else '#FA812F' for i in range(len(categories))],
                text=[f"{p:.2f}%" for p in probabilities],
                textposition='outside',
                marker=dict(
                    line=dict(color='white', width=2)
                )
            )
        ])

        fig_proba.update_layout(
            title="Prediction Confidence Across All Categories",
            xaxis_title="Product Category",
            yaxis_title="Probability (%)",
            showlegend=False,
            height=450,
            font=dict(family="Inter", color="#2C3E50"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#F8F9FA',
            xaxis=dict(gridcolor='#E9ECEF'),
            yaxis=dict(gridcolor='#E9ECEF')
        )

        st.plotly_chart(fig_proba, use_container_width=True)

        # Input Summary
        st.markdown("---")

        st.markdown("""
        <div class="section-header-pred">
            <span class="icon">📝</span>
            <h2>Input Summary</h2>
        </div>
        """, unsafe_allow_html=True)

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #E9ECEF;">
                <h4 style="color: #2C3E50; margin-top: 0;">Customer Details</h4>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"**Gender:** {gender}")
            st.markdown(f"**Age:** {age} years")
            st.markdown(f"**Review Rating:** {review_rating:.1f} ⭐")

        with summary_col2:
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #E9ECEF;">
                <h4 style="color: #2C3E50; margin-top: 0;">Product Details</h4>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"**Item:** {item_purchased}")
            st.markdown(f"**Color:** {color}")
            st.markdown(f"**Size:** {size}")
            st.markdown(f"**Price:** ${purchase_amount}")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

st.markdown("---")

# Feature Importance
st.markdown("""
<div class="section-header-pred">
    <span class="icon">🎯</span>
    <h2>Feature Importance Analysis</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #FEF3E2 0%, #ffffff 100%); padding: 1.5rem; border-radius: 0.75rem; margin-bottom: 2rem;">
    <p style="margin: 0; color: #2C3E50;">
        This chart shows which features have the most influence on the model's predictions. Higher importance means the feature plays a larger role in determining the category.
    </p>
</div>
""", unsafe_allow_html=True)

# Get feature importance from model
feature_names = ['Item Purchased', 'Color', 'Size', 'Gender', 'Age', 'Purchase Amount', 'Review Rating']
feature_importance = model.feature_importances_

# Create dataframe
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=True)  # Sort for horizontal bar

# Enhanced importance chart
fig_importance = go.Figure(data=[
    go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker=dict(
            color=importance_df['Importance'],
            colorscale=[[0, '#FEF3E2'], [0.5, '#FAB12F'], [0.75, '#FA812F'], [1, '#DD0303']],
            line=dict(color='white', width=1.5)
        ),
        text=[f"{val:.3f}" for val in importance_df['Importance']],
        textposition='outside'
    )
])

fig_importance.update_layout(
    title="Feature Impact on Category Predictions",
    xaxis_title="Importance Score",
    yaxis_title="",
    showlegend=False,
    height=450,
    font=dict(family="Inter", color="#2C3E50"),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#F8F9FA',
    xaxis=dict(gridcolor='#E9ECEF'),
    yaxis=dict(gridcolor='#E9ECEF')
)

st.plotly_chart(fig_importance, use_container_width=True)

st.markdown("---")

# Batch Prediction
st.markdown("""
<div class="section-header-pred">
    <span class="icon">📦</span>
    <h2>Batch Predictions</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: linear-gradient(135deg, #FEF3E2 0%, #ffffff 100%); padding: 1.5rem; border-radius: 0.75rem; border-left: 4px solid #FA812F; margin-bottom: 2rem;">
    <h4 style="color: #2C3E50; margin-top: 0;">Upload CSV for Bulk Predictions</h4>
    <p style="margin: 0; color: #495057;">
        Upload a CSV file containing multiple rows of customer and product data. The model will generate predictions for all rows simultaneously.
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📎 Choose a CSV file",
    type="csv",
    help="Upload a CSV with columns: Item Purchased, Color, Size, Gender, Age, Purchase Amount (USD), Review Rating"
)

if uploaded_file is not None:
    try:
        # Read uploaded file
        batch_df = pd.read_csv(uploaded_file)

        st.success(f"✅ File loaded successfully! {len(batch_df)} rows detected.")

        with st.expander("👁️ Preview Uploaded Data", expanded=True):
            st.dataframe(batch_df.head(10), use_container_width=True)

        if st.button("🔮 Run Batch Predictions", type="primary", use_container_width=True):
            # Prepare data
            required_features = ['Item Purchased', 'Color', 'Size', 'Gender', 'Age', 'Purchase Amount (USD)', 'Review Rating']

            # Check if all required features are present
            missing_features = set(required_features) - set(batch_df.columns)
            if missing_features:
                st.error(f"⚠️ Missing required columns: {', '.join(missing_features)}")
            else:
                with st.spinner('🔄 Processing predictions...'):
                    # Encode categorical features
                    batch_encoded = batch_df.copy()
                    for feature in ['Item Purchased', 'Color', 'Size', 'Gender']:
                        if feature in label_encoders:
                            le = label_encoders[feature]
                            # Handle unseen labels
                            batch_encoded[feature] = batch_encoded[feature].apply(
                                lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0]
                            )

                    # Make predictions
                    predictions = model.predict(batch_encoded[required_features])
                    predicted_categories = le_target.inverse_transform(predictions)

                    # Add predictions to dataframe
                    result_df = batch_df.copy()
                    result_df['Predicted_Category'] = predicted_categories

                    st.success(f"✅ Batch predictions complete! {len(result_df)} rows processed.")

                    # Show results
                    st.markdown("#### 📊 Prediction Results")
                    st.dataframe(result_df, use_container_width=True, height=400)

                    # Category distribution
                    category_dist = result_df['Predicted_Category'].value_counts()
                    fig_batch = px.pie(
                        values=category_dist.values,
                        names=category_dist.index,
                        title="Distribution of Predicted Categories",
                        color_discrete_sequence=["#FA812F", "#FAB12F", "#DD0303", "#FEF3E2"],
                        hole=0.4
                    )
                    fig_batch.update_layout(
                        height=350,
                        font=dict(family="Inter", color="#2C3E50")
                    )
                    st.plotly_chart(fig_batch, use_container_width=True)

                    # Download button
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=result_df.to_csv(index=False),
                        file_name="shopbis_batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

st.markdown("---")

# Model Information
with st.expander("ℹ️ About the Prediction Model", expanded=False):
    st.markdown("""
    ### 🤖 Random Forest Classifier

    **Training Data**: 3,900 customer purchase records

    **Input Features** (7 total):
    - 📦 **Item Purchased**: Product name
    - 🎨 **Color**: Product color
    - 📏 **Size**: Product size (S, M, L, XL)
    - 👤 **Gender**: Customer gender
    - 🎂 **Age**: Customer age in years
    - 💰 **Purchase Amount**: Transaction amount in USD
    - ⭐ **Review Rating**: Customer rating (1-5 stars)

    **Output**: Product **Category** prediction
    - Clothing
    - Footwear
    - Accessories
    - Outerwear

    **Model Performance**:
    - Accuracy: ~99%
    - Algorithm: Random Forest (Ensemble of 200 decision trees)
    - Training Method: Supervised learning with label encoding

    ### 📚 How to Use

    **Single Prediction:**
    1. Fill in customer and product information
    2. Click "Generate Prediction"
    3. View predicted category and confidence score

    **Batch Prediction:**
    1. Prepare CSV with required columns
    2. Upload file using the file uploader
    3. Click "Run Batch Predictions"
    4. Download results with predictions

    ### 🎯 Interpreting Results

    - **Predicted Category**: Most likely product category
    - **Confidence**: Model certainty (higher = more confident)
    - **Probability Distribution**: Shows likelihood for all categories
    - **Feature Importance**: Which inputs most influence predictions
    """)

# Footer
st.markdown("""
<div style="margin-top: 3rem; padding: 2rem; background: linear-gradient(135deg, #2C3E50 0%, #1a252f 100%); border-radius: 1rem; text-align: center; color: white;">
    <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">
        Powered by Random Forest ML Algorithm | Scikit-learn & Streamlit
    </p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; color: #FAB12F; font-weight: 600;">
        Developed by Kent Paulo Delgado
    </p>
</div>
""", unsafe_allow_html=True)
