"""
Product Insights Page - ShopBis Dashboard
==========================================
Product-level trends and performance analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(page_title="Product Insights - ShopBis", page_icon="📦", layout="wide", initial_sidebar_state="collapsed")

# Apply professional theme
professional_css = Path(__file__).parent.parent / "shared_styles.txt"
if professional_css.exists():
    st.markdown(f"<style>{professional_css.read_text()}</style>", unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .stApp { background: #F8F9FA; }
        [data-testid="stSidebar"] { display: none; }
        button[kind="header"] { display: none; }
        .top-nav { position: sticky; top: 0; z-index: 1000; background: #2C3E50; padding: 0.75rem 2rem; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08); margin: -1rem -2rem 2rem -2rem; display: flex; align-items: center; justify-content: space-between; }
        .top-nav-logo { display: flex; align-items: center; gap: 0.75rem; color: white; }
        .top-nav-logo h1 { margin: 0; font-size: 1.4rem; font-weight: 600; }
        .top-nav-logo .emoji { font-size: 1.8rem; }
        .top-nav-links { display: flex; gap: 0.5rem; align-items: center; }
        .nav-link { color: #E9ECEF; text-decoration: none; padding: 0.6rem 1.25rem; border-radius: 0.5rem; font-weight: 500; font-size: 0.9rem; transition: all 0.2s ease; background: transparent; display: flex; align-items: center; gap: 0.5rem; cursor: pointer; }
        .nav-link:hover { background: rgba(255, 255, 255, 0.1); color: white; }
        .nav-link.active { background: #FA812F; color: white; font-weight: 600; }
        .stMetric { background: white; padding: 1.5rem; border-radius: 0.75rem; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04); border: 1px solid #E9ECEF; transition: all 0.3s ease; }
        .stMetric:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); }
        .stMetric label { color: #6C757D !important; font-weight: 500 !important; font-size: 0.9rem !important; }
        .stMetric [data-testid="stMetricValue"] { color: #FA812F !important; font-weight: 700 !important; }
        .stButton button, .stDownloadButton button { background: #FA812F !important; color: white !important; border: none !important; border-radius: 0.5rem !important; padding: 0.65rem 1.5rem !important; font-weight: 600 !important; transition: all 0.2s ease !important; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important; }
        .stButton button:hover, .stDownloadButton button:hover { background: #DD0303 !important; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important; }
        h1, h2, h3, h4, h5, h6 { color: #2C3E50 !important; }
        p, div, span { color: #495057; }
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
        <a href="/Prediction" target="_self" class="nav-link">
            <span>🔮</span> Prediction
        </a>
        <a href="/Product_Insights" target="_self" class="nav-link active">
            <span>📦</span> Product Insights
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# Title
st.title("📦 Product Insights")
st.markdown("### Analyze product trends and performance metrics")

# Load data
@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent.parent / "data" / "shopping_behavior_cleaned.csv"
    return pd.read_csv(data_path)

df = load_data()

st.markdown("---")

# Top-level KPIs
st.subheader("📊 Product Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_products = df["Item Purchased"].nunique()
    st.metric("Total Unique Items", f"{total_products:,}")

with col2:
    total_categories = df["Category"].nunique()
    st.metric("Product Categories", f"{total_categories}")

with col3:
    total_revenue = df["Purchase Amount (USD)"].sum()
    st.metric("Total Revenue", f"${total_revenue:,.0f}")

with col4:
    avg_price = df["Purchase Amount (USD)"].mean()
    st.metric("Avg Product Price", f"${avg_price:.2f}")

with col5:
    avg_rating = df["Review Rating"].mean()
    st.metric("Avg Product Rating", f"{avg_rating:.2f} ⭐")

st.markdown("---")

# Top-Selling Categories
st.subheader("🏆 Top-Selling Categories")

col1, col2 = st.columns([2, 1])

with col1:
    # Category sales
    category_sales = df.groupby("Category").agg({
        "Customer ID": "count",
        "Purchase Amount (USD)": ["sum", "mean"],
        "Review Rating": "mean"
    }).round(2)

    category_sales.columns = ["Total Sales", "Total Revenue ($)", "Avg Price ($)", "Avg Rating"]
    category_sales = category_sales.sort_values("Total Sales", ascending=False).reset_index()

    fig_category = px.bar(
        category_sales,
        x="Category",
        y="Total Sales",
        title="Sales Volume by Category",
        color="Total Sales",
        color_continuous_scale=["#FEF3E2", "#FAB12F", "#FA812F", "#DD0303"],
        text="Total Sales"
    )
    fig_category.update_traces(textposition='outside')
    fig_category.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_category, use_container_width=True)

with col2:
    st.markdown("**Category Performance**")
    st.dataframe(category_sales, use_container_width=True, hide_index=True)

st.markdown("---")

# Product Performance by Gender
st.subheader("👥 Product Performance by Gender")

col1, col2 = st.columns(2)

with col1:
    # Category by gender
    gender_category = df.groupby(["Gender", "Category"]).size().reset_index(name="Count")

    fig_gender_cat = px.bar(
        gender_category,
        x="Category",
        y="Count",
        color="Gender",
        barmode="group",
        title="Category Purchases by Gender",
        color_discrete_map={"Male": "#FA812F", "Female": "#DD0303"}
    )
    fig_gender_cat.update_layout(height=400)
    st.plotly_chart(fig_gender_cat, use_container_width=True)

with col2:
    # Average spending by gender and category
    gender_spending = df.groupby(["Gender", "Category"])["Purchase Amount (USD)"].mean().reset_index()

    fig_gender_spend = px.bar(
        gender_spending,
        x="Category",
        y="Purchase Amount (USD)",
        color="Gender",
        barmode="group",
        title="Average Spending by Gender & Category",
        color_discrete_map={"Male": "#FA812F", "Female": "#DD0303"}
    )
    fig_gender_spend.update_layout(height=400)
    st.plotly_chart(fig_gender_spend, use_container_width=True)

st.markdown("---")

# Product Performance by Age Group
st.subheader("🎂 Product Performance by Age Group")

# Create age groups
df_with_age_group = df.copy()
df_with_age_group["Age Group"] = pd.cut(
    df_with_age_group["Age"],
    bins=[0, 25, 35, 45, 55, 100],
    labels=["18-25", "26-35", "36-45", "46-55", "56+"]
)

col1, col2 = st.columns(2)

with col1:
    # Category by age group
    age_category = df_with_age_group.groupby(["Age Group", "Category"]).size().reset_index(name="Count")

    fig_age_cat = px.bar(
        age_category,
        x="Age Group",
        y="Count",
        color="Category",
        title="Category Purchases by Age Group",
        barmode="stack"
    )
    fig_age_cat.update_layout(height=400)
    st.plotly_chart(fig_age_cat, use_container_width=True)

with col2:
    # Average spending by age group
    age_spending = df_with_age_group.groupby("Age Group")["Purchase Amount (USD)"].mean().reset_index()

    fig_age_spend = px.line(
        age_spending,
        x="Age Group",
        y="Purchase Amount (USD)",
        title="Average Spending by Age Group",
        markers=True,
        line_shape="spline"
    )
    fig_age_spend.update_traces(marker=dict(size=10), line=dict(width=3))
    fig_age_spend.update_layout(height=400)
    st.plotly_chart(fig_age_spend, use_container_width=True)

st.markdown("---")

# Seasonal Trends
st.subheader("🌤️ Seasonal Product Trends")

col1, col2 = st.columns(2)

with col1:
    # Category by season
    season_category = df.groupby(["Season", "Category"]).size().reset_index(name="Count")

    fig_season = px.bar(
        season_category,
        x="Season",
        y="Count",
        color="Category",
        title="Product Categories by Season",
        barmode="group"
    )
    fig_season.update_layout(height=400)
    st.plotly_chart(fig_season, use_container_width=True)

with col2:
    # Revenue by season
    season_revenue = df.groupby("Season").agg({
        "Purchase Amount (USD)": "sum",
        "Customer ID": "count"
    }).reset_index()
    season_revenue.columns = ["Season", "Revenue", "Count"]

    fig_season_rev = px.bar(
        season_revenue,
        x="Season",
        y="Revenue",
        title="Total Revenue by Season",
        color="Season",
        color_discrete_map={
            "Spring": "#90EE90",
            "Summer": "#FFD700",
            "Fall": "#FFA500",
            "Winter": "#4169E1"
        },
        text="Revenue"
    )
    fig_season_rev.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig_season_rev.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_season_rev, use_container_width=True)

st.markdown("---")

# Review Ratings Analysis
st.subheader("⭐ Product Ratings Analysis")

col1, col2 = st.columns(2)

with col1:
    # Review ratings by category
    fig_rating_cat = px.box(
        df,
        x="Category",
        y="Review Rating",
        title="Review Ratings Distribution by Category",
        color="Category"
    )
    fig_rating_cat.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_rating_cat, use_container_width=True)

with col2:
    # Rating distribution
    fig_rating_dist = px.histogram(
        df,
        x="Review Rating",
        nbins=20,
        title="Overall Rating Distribution",
        color_discrete_sequence=["#FA812F"]
    )
    fig_rating_dist.update_layout(
        xaxis_title="Review Rating",
        yaxis_title="Count",
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig_rating_dist, use_container_width=True)

st.markdown("---")

# Top Products
st.subheader("🏅 Top Performing Products")

# Top 10 products by sales volume
top_products = df.groupby("Item Purchased").agg({
    "Customer ID": "count",
    "Purchase Amount (USD)": ["mean", "sum"],
    "Review Rating": "mean"
}).round(2)

top_products.columns = ["Sales Count", "Avg Price ($)", "Total Revenue ($)", "Avg Rating"]
top_products = top_products.sort_values("Sales Count", ascending=False).head(10).reset_index()

st.dataframe(top_products, use_container_width=True, hide_index=True)

# Visualize top 10 products
fig_top_products = px.bar(
    top_products,
    x="Sales Count",
    y="Item Purchased",
    orientation='h',
    title="Top 10 Best-Selling Products",
    color="Avg Rating",
    color_continuous_scale=["#FEF3E2", "#FAB12F", "#FA812F", "#DD0303"],
    text="Sales Count"
)
fig_top_products.update_traces(textposition='outside')
fig_top_products.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig_top_products, use_container_width=True)

st.markdown("---")

# Discount Analysis
st.subheader("🎁 Discount & Promotion Analysis")

col1, col2 = st.columns(2)

with col1:
    # Discount usage
    discount_stats = df.groupby("Discount Applied").agg({
        "Customer ID": "count",
        "Purchase Amount (USD)": "mean",
        "Review Rating": "mean"
    }).round(2).reset_index()
    discount_stats.columns = ["Discount Applied", "Count", "Avg Spending ($)", "Avg Rating"]

    st.markdown("**Discount Impact**")
    st.dataframe(discount_stats, use_container_width=True, hide_index=True)

    # Visualization
    fig_discount = px.bar(
        discount_stats,
        x="Discount Applied",
        y="Avg Spending ($)",
        title="Average Spending: Discount vs No Discount",
        color="Discount Applied",
        color_discrete_map={"Yes": "#FA812F", "No": "#DD0303"},
        text="Avg Spending ($)"
    )
    fig_discount.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
    fig_discount.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_discount, use_container_width=True)

with col2:
    # Promo code usage
    promo_stats = df.groupby("Promo Code Used").agg({
        "Customer ID": "count",
        "Purchase Amount (USD)": "mean",
        "Review Rating": "mean"
    }).round(2).reset_index()
    promo_stats.columns = ["Promo Code Used", "Count", "Avg Spending ($)", "Avg Rating"]

    st.markdown("**Promo Code Impact**")
    st.dataframe(promo_stats, use_container_width=True, hide_index=True)

    # Visualization
    fig_promo = px.bar(
        promo_stats,
        x="Promo Code Used",
        y="Avg Spending ($)",
        title="Average Spending: Promo Code vs No Promo",
        color="Promo Code Used",
        color_discrete_map={"Yes": "#FAB12F", "No": "#FA812F"},
        text="Avg Spending ($)"
    )
    fig_promo.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
    fig_promo.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig_promo, use_container_width=True)

st.markdown("---")

# Payment Method Analysis
st.subheader("💳 Payment Method Preferences")

col1, col2 = st.columns(2)

with col1:
    # Payment method distribution
    payment_dist = df["Payment Method"].value_counts().reset_index()
    payment_dist.columns = ["Payment Method", "Count"]

    fig_payment = px.pie(
        payment_dist,
        values="Count",
        names="Payment Method",
        title="Payment Method Distribution",
        hole=0.4,
        color_discrete_sequence=["#FA812F", "#FAB12F", "#DD0303", "#FEF3E2"]
    )
    fig_payment.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_payment, use_container_width=True)

with col2:
    # Payment method by category
    payment_category = df.groupby(["Payment Method", "Category"]).size().reset_index(name="Count")
    payment_category = payment_category.sort_values("Count", ascending=False).head(15)

    fig_payment_cat = px.bar(
        payment_category,
        x="Payment Method",
        y="Count",
        color="Category",
        title="Payment Methods by Category",
        barmode="stack"
    )
    st.plotly_chart(fig_payment_cat, use_container_width=True)

st.markdown("---")

# Size and Color Analysis
st.subheader("🎨 Product Attributes Analysis")

col1, col2 = st.columns(2)

with col1:
    # Size distribution
    size_dist = df["Size"].value_counts().reset_index()
    size_dist.columns = ["Size", "Count"]

    fig_size = px.bar(
        size_dist,
        x="Size",
        y="Count",
        title="Product Size Distribution",
        color="Count",
        color_continuous_scale=["#FEF3E2", "#FAB12F", "#FA812F", "#DD0303"],
        text="Count"
    )
    fig_size.update_traces(textposition='outside')
    fig_size.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_size, use_container_width=True)

with col2:
    # Top 10 colors
    color_dist = df["Color"].value_counts().head(10).reset_index()
    color_dist.columns = ["Color", "Count"]

    fig_color = px.bar(
        color_dist,
        x="Count",
        y="Color",
        orientation='h',
        title="Top 10 Product Colors",
        color="Count",
        color_continuous_scale=["#FEF3E2", "#FAB12F", "#FA812F", "#DD0303"],
        text="Count"
    )
    fig_color.update_traces(textposition='outside')
    fig_color.update_layout(showlegend=False, height=400, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_color, use_container_width=True)

st.markdown("---")

# Shipping Analysis
st.subheader("🚚 Shipping Preferences")

shipping_dist = df.groupby("Shipping Type").agg({
    "Customer ID": "count",
    "Purchase Amount (USD)": "mean"
}).round(2).reset_index()
shipping_dist.columns = ["Shipping Type", "Count", "Avg Spending ($)"]
shipping_dist = shipping_dist.sort_values("Count", ascending=False)

col1, col2 = st.columns(2)

with col1:
    fig_ship_dist = px.pie(
        shipping_dist,
        values="Count",
        names="Shipping Type",
        title="Shipping Method Distribution",
        color_discrete_sequence=["#FA812F", "#FAB12F", "#DD0303", "#FEF3E2"]
    )
    st.plotly_chart(fig_ship_dist, use_container_width=True)

with col2:
    fig_ship_spend = px.bar(
        shipping_dist,
        x="Shipping Type",
        y="Avg Spending ($)",
        title="Average Spending by Shipping Method",
        color="Avg Spending ($)",
        color_continuous_scale=["#FEF3E2", "#FAB12F", "#FA812F", "#DD0303"],
        text="Avg Spending ($)"
    )
    fig_ship_spend.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
    fig_ship_spend.update_layout(showlegend=False)
    st.plotly_chart(fig_ship_spend, use_container_width=True)

st.markdown("---")

# Subscription Analysis
st.subheader("📬 Subscription Status Impact")

subscription_stats = df.groupby(["Subscription Status", "Category"]).agg({
    "Customer ID": "count",
    "Purchase Amount (USD)": "mean",
    "Review Rating": "mean"
}).round(2).reset_index()
subscription_stats.columns = ["Subscription", "Category", "Count", "Avg Spending ($)", "Avg Rating"]

col1, col2 = st.columns(2)

with col1:
    fig_sub = px.bar(
        subscription_stats,
        x="Category",
        y="Count",
        color="Subscription",
        barmode="group",
        title="Purchases by Subscription Status",
        color_discrete_map={"Yes": "#FA812F", "No": "#DD0303"}
    )
    st.plotly_chart(fig_sub, use_container_width=True)

with col2:
    fig_sub_spend = px.bar(
        subscription_stats,
        x="Category",
        y="Avg Spending ($)",
        color="Subscription",
        barmode="group",
        title="Avg Spending by Subscription Status",
        color_discrete_map={"Yes": "#FA812F", "No": "#DD0303"}
    )
    st.plotly_chart(fig_sub_spend, use_container_width=True)

st.markdown("---")

# Key Insights Summary
st.subheader("💡 Key Insights Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🏆 Top Category**")
    top_category = df["Category"].value_counts().index[0]
    top_category_count = df["Category"].value_counts().values[0]
    st.info(f"{top_category} with {top_category_count:,} sales")

with col2:
    st.markdown("**⭐ Highest Rated Category**")
    best_rated = df.groupby("Category")["Review Rating"].mean().idxmax()
    best_rating = df.groupby("Category")["Review Rating"].mean().max()
    st.success(f"{best_rated} ({best_rating:.2f} stars)")

with col3:
    st.markdown("**💰 Highest Revenue Category**")
    top_revenue_cat = df.groupby("Category")["Purchase Amount (USD)"].sum().idxmax()
    top_revenue = df.groupby("Category")["Purchase Amount (USD)"].sum().max()
    st.warning(f"{top_revenue_cat} (${top_revenue:,.0f})")

st.markdown("---")

# Export Options
st.subheader("📥 Export Product Data")

col1, col2, col3 = st.columns(3)

with col1:
    # Export category summary
    category_summary = df.groupby("Category").agg({
        "Customer ID": "count",
        "Purchase Amount (USD)": ["sum", "mean"],
        "Review Rating": "mean"
    }).round(2)
    category_summary.columns = ["Total Sales", "Total Revenue", "Avg Price", "Avg Rating"]

    st.download_button(
        label="Download Category Summary",
        data=category_summary.to_csv(),
        file_name="category_summary.csv",
        mime="text/csv"
    )

with col2:
    # Export product performance
    product_summary = df.groupby("Item Purchased").agg({
        "Customer ID": "count",
        "Purchase Amount (USD)": "mean",
        "Review Rating": "mean"
    }).round(2)
    product_summary.columns = ["Sales Count", "Avg Price", "Avg Rating"]

    st.download_button(
        label="Download Product Performance",
        data=product_summary.to_csv(),
        file_name="product_performance.csv",
        mime="text/csv"
    )

with col3:
    # Export full insights
    st.download_button(
        label="Download Full Dataset",
        data=df.to_csv(index=False),
        file_name="shopbis_full_data.csv",
        mime="text/csv"
    )
