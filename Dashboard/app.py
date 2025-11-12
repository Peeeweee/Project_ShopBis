"""
ShopBis Dashboard - Main Application
=====================================
A comprehensive Streamlit dashboard for shopping behavior analysis
with 4 pages: Overview, Segmentation, Prediction, and Product Insights
"""

import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="ShopBis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Advanced Custom CSS with Animations and Modern Design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }

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

    /* Professional Cool-Toned Background */
    .stApp {
        background: #F8F9FA;
    }

    /* Remove default Streamlit padding */
    .main .block-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Remove top padding from app */
    section[data-testid="stAppViewContainer"] > .main {
        padding-top: 0 !important;
    }

    /* Remove top app container padding */
    .stApp > header {
        display: none !important;
    }

    /* Main Header - Professional */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        padding: 2rem 0 0.5rem 0;
        margin-bottom: 0.5rem;
        color: #2C3E50;
        letter-spacing: -0.5px;
    }

    /* Subtitle - Professional */
    .subtitle {
        font-size: 1rem;
        color: #6C757D;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Section Headers - Professional */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2C3E50;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E9ECEF;
    }

    /* Professional Card Design */
    .glass-card {
        background: white;
        border-radius: 0.75rem;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid #E9ECEF;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }

    /* Professional Feature Cards */
    .feature-card {
        background: white;
        border-radius: 0.75rem;
        padding: 1.75rem;
        margin: 1rem 0;
        border: 1px solid #E9ECEF;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
        border-color: #DEE2E6;
    }

    .feature-card h4 {
        color: #2C3E50;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }

    .feature-card p {
        color: #6C757D;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    .feature-card ul {
        list-style: none;
        padding-left: 0;
    }

    .feature-card ul li {
        color: #495057;
        font-size: 0.9rem;
        padding: 0.4rem 0;
        padding-left: 1.5rem;
        position: relative;
    }

    .feature-card ul li::before {
        content: '✓';
        position: absolute;
        left: 0;
        color: #FA812F;
        font-weight: bold;
        font-size: 1rem;
    }

    /* Professional Info Box */
    .info-box {
        background: white;
        padding: 1.75rem;
        border-radius: 0.75rem;
        border-left: 4px solid #FA812F;
        margin: 1.5rem 0;
        border: 1px solid #E9ECEF;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: all 0.3s ease;
    }

    .info-box:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }

    .info-box strong {
        color: #2C3E50;
        font-size: 1.05rem;
        font-weight: 600;
    }

    /* Professional Metric Cards */
    .stMetric {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        border: 1px solid #E9ECEF;
        transition: all 0.3s ease;
    }

    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }

    /* Metric Value Styling - Orange Accent */
    .stMetric label {
        color: #6C757D !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: #FA812F !important;
        font-weight: 700 !important;
    }

    /* Hide Sidebar Completely */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Hide sidebar toggle button */
    button[kind="header"] {
        display: none;
    }

    /* Enhanced Professional Navigation Bar */
    .top-nav {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        background: linear-gradient(135deg, #1a252f 0%, #2C3E50 100%);
        padding: 1rem 3rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 3px solid #FA812F;
    }

    .top-nav-logo {
        display: flex;
        align-items: center;
        gap: 1rem;
        color: white;
        padding: 0.5rem 1rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.75rem;
        transition: all 0.3s ease;
    }

    .top-nav-logo:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
    }

    .top-nav-logo h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        background: linear-gradient(135deg, #ffffff 0%, #FAB12F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .top-nav-logo .emoji {
        font-size: 2rem;
        filter: drop-shadow(0 2px 4px rgba(250, 129, 47, 0.3));
    }

    .top-nav-links {
        display: flex;
        gap: 0.75rem;
        align-items: center;
        background: rgba(0, 0, 0, 0.2);
        padding: 0.5rem;
        border-radius: 0.75rem;
    }

    .nav-link {
        color: #CBD5E0;
        text-decoration: none;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        background: transparent;
        border: 2px solid transparent;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }

    .nav-link::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s ease;
    }

    .nav-link:hover::before {
        left: 100%;
    }

    .nav-link:hover {
        background: rgba(255, 255, 255, 0.1);
        color: white;
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.2);
    }

    .nav-link.active {
        background: linear-gradient(135deg, #FA812F 0%, #DD0303 100%);
        color: white;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(250, 129, 47, 0.4);
        border-color: #FAB12F;
    }

    .nav-link.active:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 6px 20px rgba(250, 129, 47, 0.5);
    }

    .nav-link span {
        font-size: 1.1rem;
        filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.3));
    }

    /* Professional Button Styling */
    .stButton button {
        background: #FA812F;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.65rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .stButton button:hover {
        background: #DD0303;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }

    /* Professional Divider */
    hr {
        margin: 2.5rem 0;
        border: none;
        height: 1px;
        background: #E9ECEF;
    }

    /* Professional Stats Badge */
    .stat-badge {
        display: inline-block;
        background: #FA812F;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 0.375rem;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.25rem;
        transition: all 0.2s ease;
    }

    .stat-badge:hover {
        background: #DD0303;
    }

    /* Professional Progress Bar */
    .stProgress > div > div {
        background: #FA812F;
    }

    /* Professional Alert Styling */
    .stAlert {
        border-radius: 0.5rem !important;
        border-left: 4px solid #FA812F !important;
        background: white !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    }

    /* Professional Download Button */
    .stDownloadButton button {
        background: #FA812F !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        padding: 0.65rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }

    .stDownloadButton button:hover {
        background: #DD0303 !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
    }

    /* Professional Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        background: white;
        border: 1px solid #E9ECEF;
        color: #6C757D;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: #FA812F;
        color: white !important;
        border-color: #FA812F;
    }
</style>
""", unsafe_allow_html=True)

# Add spacer at the very top
st.markdown("""
<div style="height: 1.5rem; background: #F8F9FA;"></div>
""", unsafe_allow_html=True)

# Top Navigation Bar
st.markdown("""
<div class="top-nav">
    <div class="top-nav-logo">
        <span class="emoji">🛍️</span>
        <h1>Shop_Bis</h1>
    </div>
    <div class="top-nav-links">
        <a href="/" target="_self" class="nav-link active">
            <span>🏠</span> Home & Overview
        </a>
        <a href="/Segmentation" target="_self" class="nav-link">
            <span>👥</span> Segmentation
        </a>
        <a href="/Prediction" target="_self" class="nav-link">
            <span>🔮</span> Prediction
        </a>
        <a href="/Product_Insights" target="_self" class="nav-link">
            <span>📦</span> Product Insights
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# Hero Section with Animated Header
st.markdown("""
<div style="background: white; padding: 2.5rem 2rem; margin: 1rem -2rem 2rem -2rem; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);">
    <div style="text-align: center;">
        <h1 style="font-size: 3.5rem; font-weight: 800; color: #2C3E50; margin-bottom: 1rem; letter-spacing: -1px;">
            ShopBis Analytics Dashboard
        </h1>
        <p style="font-size: 1.2rem; color: #6C757D; max-width: 800px; margin: 0 auto; line-height: 1.8;">
            Comprehensive Shopping Behavior Analysis & Insights Platform
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# About This Dashboard Section
st.markdown("""
<div style="background: linear-gradient(135deg, #FEF3E2 0%, #ffffff 100%); padding: 2.5rem; border-radius: 1rem; border-left: 4px solid #FA812F; margin: 2rem 0 3rem 0; box-shadow: 0 4px 16px rgba(0,0,0,0.06);">
    <h2 style="color: #2C3E50; margin-top: 0;">
        📊 About This Dashboard
    </h2>
    <p style="color: #495057; margin: 1rem 0; line-height: 1.8; font-size: 1.05rem;">
        <strong>ShopBis Analytics Dashboard</strong> is a comprehensive, AI-powered analytics platform designed to transform raw shopping behavior data into actionable business intelligence. Built with modern data science techniques and professional UI/UX design, this dashboard empowers businesses to make data-driven decisions.
    </p>
</div>
""", unsafe_allow_html=True)

# What Can You Do Section
st.markdown("### 🎯 What Can You Do?")
col1, col2 = st.columns(2, gap="medium")

with col1:
    st.markdown("""
    **🏠 Explore Real-Time Insights**
    View live metrics, KPIs, and visualizations from 3,900+ customer records including demographics, purchase patterns, and behavioral trends.

    **🔮 Predict Behavior**
    Leverage our 97.82% accurate Random Forest model to predict if customers will make repeat purchases and identify retention strategies.
    """)

with col2:
    st.markdown("""
    **👥 Segment Customers**
    Use K-Means clustering to discover natural customer groups, understand their characteristics, and receive AI-powered marketing recommendations.

    **📦 Analyze Products**
    Deep-dive into product performance, seasonal trends, category analytics, and customer preferences across demographics.
    """)

# ML Stats Section
st.markdown("### 🤖 Powered by Machine Learning")
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 0.75rem; border: 1px solid #E9ECEF;">
        <div style="font-size: 2.5rem; font-weight: 700; color: #FA812F;">Random Forest</div>
        <p style="color: #6C757D; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Purchase Behavior<br>Prediction Algorithm</p>
        <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #E9ECEF;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #FA812F;">97.82%</div>
            <p style="color: #6C757D; margin: 0; font-size: 0.85rem;">Accuracy</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 0.75rem; border: 1px solid #E9ECEF;">
        <div style="font-size: 2.5rem; font-weight: 700; color: #FA812F;">K-Means</div>
        <p style="color: #6C757D; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Customer Segmentation<br>Algorithm</p>
        <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #E9ECEF;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #FA812F;">2-6 Clusters</div>
            <p style="color: #6C757D; margin: 0; font-size: 0.85rem;">Configurable</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 0.75rem; border: 1px solid #E9ECEF;">
        <div style="font-size: 2.5rem; font-weight: 700; color: #FA812F;">10+</div>
        <p style="color: #6C757D; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Interactive<br>Visualizations</p>
        <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid #E9ECEF;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #FA812F;">4 Pages</div>
            <p style="color: #6C757D; margin: 0; font-size: 0.85rem;">Analytics Modules</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<p style="color: #6C757D; margin: 2rem 0 0 0; font-size: 0.95rem; text-align: center;">
    <em>Navigate through the pages using the top navigation bar to explore different analytics modules</em>
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Load data for overview metrics
@st.cache_data
def load_data():
    from pathlib import Path
    import pandas as pd
    data_path = Path(__file__).parent.parent / "data" / "shopping_behavior_cleaned.csv"
    return pd.read_csv(data_path)

df = load_data()

# Key Metrics Section
st.markdown("""
<div style="text-align: center; margin-bottom: 1rem;">
    <h2 style="color: #2C3E50; font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem;">
        Real-Time Dataset Insights
    </h2>
    <p style="color: #6C757D; font-size: 1rem;">
        Live metrics from your shopping behavior data
    </p>
</div>
""", unsafe_allow_html=True)

# Top KPIs in 4 columns
col1, col2, col3, col4 = st.columns(4, gap="large")

with col1:
    st.metric(
        label="📊 Total Customers",
        value=f"{len(df):,}",
        delta="Active Dataset"
    )

with col2:
    avg_purchase = df["Purchase Amount (USD)"].mean()
    st.metric(
        label="💰 Avg Purchase",
        value=f"${avg_purchase:.2f}",
        delta=f"±${df['Purchase Amount (USD)'].std():.2f}"
    )

with col3:
    avg_rating = df["Review Rating"].mean()
    st.metric(
        label="⭐ Avg Rating",
        value=f"{avg_rating:.2f}/5.0",
        delta="Customer Satisfaction"
    )

with col4:
    categories = df["Category"].nunique()
    st.metric(
        label="📦 Categories",
        value=f"{categories}",
        delta="Product Lines"
    )

# Quick Visual Insights
st.markdown('<h2 class="section-header">Quick Visual Insights</h2>', unsafe_allow_html=True)

viz_col1, viz_col2, viz_col3 = st.columns(3, gap="large")

with viz_col1:
    # Gender Distribution
    import plotly.express as px
    gender_counts = df["Gender"].value_counts()
    fig_gender = px.pie(
        values=gender_counts.values,
        names=gender_counts.index,
        title="Customer Gender Distribution",
        color_discrete_sequence=["#FA812F", "#DD0303"]
    )
    fig_gender.update_layout(
        height=300,
        margin=dict(t=40, b=0, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color="#2C3E50")
    )
    st.plotly_chart(fig_gender, use_container_width=True)

with viz_col2:
    # Top Categories
    category_sales = df.groupby("Category")["Purchase Amount (USD)"].sum().sort_values(ascending=False)
    fig_cat = px.bar(
        x=category_sales.index,
        y=category_sales.values,
        title="Revenue by Category",
        labels={"x": "Category", "y": "Total Revenue ($)"},
        color=category_sales.values,
        color_continuous_scale=["#FEF3E2", "#FAB12F", "#FA812F", "#DD0303"]
    )
    fig_cat.update_layout(
        height=300,
        margin=dict(t=40, b=40, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color="#2C3E50"),
        showlegend=False
    )
    st.plotly_chart(fig_cat, use_container_width=True)

with viz_col3:
    # Age Distribution
    fig_age = px.histogram(
        df,
        x="Age",
        title="Customer Age Distribution",
        nbins=20,
        color_discrete_sequence=["#FA812F"]
    )
    fig_age.update_layout(
        height=300,
        margin=dict(t=40, b=40, l=0, r=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color="#2C3E50"),
        showlegend=False,
        xaxis_title="Age",
        yaxis_title="Count"
    )
    st.plotly_chart(fig_age, use_container_width=True)

# Dashboard Pages Section
st.markdown('<h2 class="section-header">Explore Advanced Analytics</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="feature-card">
        <h4>📊 Overview</h4>
        <p>Comprehensive snapshot of your dataset with real-time insights and interactive controls.</p>
        <ul>
            <li>8 Dynamic Key Performance Indicators</li>
            <li>Advanced filtering system (gender, age, location, income)</li>
            <li>Interactive demographic visualizations</li>
            <li>One-click data export to CSV</li>
            <li>Real-time metric updates</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">
        <h4>🎯 Segmentation</h4>
        <p>Discover natural customer groups using advanced machine learning clustering algorithms.</p>
        <ul>
            <li>K-Means clustering with 2-6 customizable clusters</li>
            <li>Interactive 2D and 3D scatter visualizations</li>
            <li>Automated customer profile generation</li>
            <li>AI-powered segment insights and recommendations</li>
            <li>Export clustered data for campaigns</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h4>🔮 Purchase Behavior Prediction</h4>
        <p>Predict if customers will make repeat purchases using AI with 97.82% accuracy.</p>
        <ul>
            <li>Single customer prediction with confidence scores</li>
            <li>Batch CSV upload for bulk behavior analysis</li>
            <li>Actionable business insights & retention strategies</li>
            <li>Feature importance visualization (10 factors)</li>
            <li>Random Forest model with 200 decision trees</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">
        <h4>📈 Product Insights</h4>
        <p>Deep dive into product performance across multiple business dimensions.</p>
        <ul>
            <li>Category-level performance analytics</li>
            <li>Demographic segmentation analysis</li>
            <li>Seasonal trend identification</li>
            <li>Discount & promotion effectiveness tracking</li>
            <li>Comprehensive export options</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Dataset Information Section
st.markdown('<h2 class="section-header">Dataset Overview</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    st.metric(
        label="Total Customers",
        value="3,900",
        delta="100% Complete",
        delta_color="normal"
    )

with col2:
    st.metric(
        label="Features",
        value="18",
        delta="Multi-dimensional",
        delta_color="normal"
    )

with col3:
    st.metric(
        label="Categories",
        value="4",
        delta="Balanced",
        delta_color="normal"
    )

with col4:
    st.metric(
        label="Avg Rating",
        value="3.75/5.0",
        delta="+0.25",
        delta_color="normal"
    )

st.markdown("""
<div class="info-box">
    <strong>📋 Dataset Composition</strong><br><br>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; color: #495057;">
        <div>
            <strong style="color: #FA812F;">Demographics</strong><br>
            <span style="font-size: 0.9rem; color: #6C757D;">Customer ID, Age, Gender, Location</span>
        </div>
        <div>
            <strong style="color: #FA812F;">Product Details</strong><br>
            <span style="font-size: 0.9rem; color: #6C757D;">Item, Category, Size, Color, Season</span>
        </div>
        <div>
            <strong style="color: #FA812F;">Purchase Behavior</strong><br>
            <span style="font-size: 0.9rem; color: #6C757D;">Amount, Previous Purchases, Frequency</span>
        </div>
        <div>
            <strong style="color: #FA812F;">Engagement Metrics</strong><br>
            <span style="font-size: 0.9rem; color: #6C757D;">Rating, Subscription, Shipping, Discounts, Payment</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Machine Learning Section
st.markdown('<h2 class="section-header">🤖 Model Information</h2>', unsafe_allow_html=True)

# Metrics for the model
col1, col2, col3 = st.columns(3, gap="large")
with col1:
    st.metric("Model Accuracy", "97.82%")
with col2:
    st.metric("Features Used", "10")
with col3:
    st.metric("Decision Trees", "200")

st.markdown("---")

# Top 5 Features
st.markdown('<h3 style="color: #2C3E50; font-weight: 600; font-size: 1.5rem; margin-bottom: 1rem;">🎯 Top 5 Most Important Features</h3>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <p><strong>Purchase Amount (USD)</strong> - 23.41% importance</p>
    <p><strong>Age</strong> - 21.69% importance</p>
    <p><strong>Review Rating</strong> - 16.79% importance</p>
    <p><strong>Shipping Type</strong> - 10.25% importance</p>
    <p><strong>Season</strong> - 7.42% importance</p>
    <br>
    <p style="font-size: 0.9rem; color: #6C757D;">
        These features have the strongest influence on predicting repeat purchase behavior.
    </p>
</div>
""", unsafe_allow_html=True)

# Getting Started Section
st.markdown('<h2 class="section-header">Get Started</h2>', unsafe_allow_html=True)

# Dataset Source Section
st.markdown("""
<div style="background: linear-gradient(135deg, #FEF3E2 0%, #ffffff 100%); padding: 2rem; border-radius: 1rem; border-left: 4px solid #FA812F; margin: 2rem 0; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);">
    <h3 style="color: #2C3E50; margin-top: 0; display: flex; align-items: center; gap: 0.75rem;">
        <span style="font-size: 1.5rem;">📊</span>
        Dataset Source
    </h3>
    <p style="color: #495057; margin: 1rem 0; line-height: 1.6;">
        This dashboard analyzes the <strong>Shopping Behavior Dataset</strong> from Kaggle, containing 3,900+ customer purchase records with detailed demographics, product information, and behavioral metrics.
    </p>
    <a href="https://www.kaggle.com/datasets/kainatjamil12/shopping-behaviour" target="_blank" style="text-decoration: none;">
        <button style="
            background: linear-gradient(135deg, #FA812F 0%, #DD0303 100%);
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.75rem 2rem;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(250, 129, 47, 0.3);
            transition: all 0.3s ease;
            font-family: 'Inter', sans-serif;
        " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 16px rgba(250, 129, 47, 0.4)';" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(250, 129, 47, 0.3)';">
            🔗 Access Dataset on Kaggle
        </button>
    </a>
</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Ready to Explore?")
    st.write("""
    Select a page from the sidebar navigation to begin your data exploration journey.
    Each page offers unique insights and interactive features.
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tips Grid
    tip_col1, tip_col2 = st.columns(2, gap="medium")

    with tip_col1:
        st.info("💡 **Pro Tip**: Start with Overview to understand the dataset distribution")
        st.success("🔮 **Power User**: Test Prediction with edge cases for model validation")

    with tip_col2:
        st.warning("🎯 **Best Practice**: Use Segmentation to identify customer patterns")
        st.error("📊 **Analytics**: Export insights from Product Insights for reporting")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem;">
    <p style="color: #94a3b8; font-size: 0.9rem;">
        <strong>ShopBis Analytics Dashboard</strong> v1.0.0 |
        Data Source: shopping_behavior_cleaned.csv |
        Last Updated: 2025
    </p>
    <p style="color: #cbd5e1; font-size: 0.85rem; margin-top: 0.5rem;">
        Built with Streamlit, Scikit-learn, Plotly & Python
    </p>
    <p style="color: #FAB12F; font-size: 0.9rem; margin-top: 1rem; font-weight: 600;">
        Developed by Kent Paulo Delgado
    </p>
</div>
""", unsafe_allow_html=True)
