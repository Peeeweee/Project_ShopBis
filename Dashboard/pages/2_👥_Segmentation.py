"""
Segmentation Page - ShopBis Dashboard
======================================
Customer clustering and segment analysis using K-Means
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Page config
st.set_page_config(page_title="Segmentation - ShopBis", page_icon="👥", layout="wide", initial_sidebar_state="collapsed")

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

    /* Sleek Card Design */
    .insight-card {
        background: white;
        border-radius: 1rem;
        padding: 2rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
        border: 1px solid #E9ECEF;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }

    .insight-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    }

    /* Cluster Badge */
    .cluster-badge {
        display: inline-block;
        padding: 0.5rem 1.25rem;
        border-radius: 2rem;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.25rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    /* Settings Panel */
    .settings-panel {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 1rem;
        padding: 2rem;
        border: 2px solid #E9ECEF;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
        margin-bottom: 2rem;
    }

    /* Section Header with Icon */
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
        font-weight: 700;
    }

    .section-header-custom .icon {
        font-size: 2rem;
        filter: drop-shadow(0 2px 4px rgba(250, 129, 47, 0.3));
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
        <a href="/" target="_self" class="nav-link">
            <span>🏠</span> Home & Overview
        </a>
        <a href="/Segmentation" target="_self" class="nav-link active">
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

# Hero Section
st.markdown("""
<div style="background: white; padding: 2.5rem 2rem; margin: 1rem -2rem 2rem -2rem; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);">
    <div style="text-align: center;">
        <h1 style="font-size: 3rem; font-weight: 800; color: #2C3E50; margin-bottom: 0.5rem; letter-spacing: -1px;">
            👥 Customer Segmentation
        </h1>
        <p style="font-size: 1.1rem; color: #6C757D; max-width: 700px; margin: 0 auto;">
            Discover natural customer groups using advanced machine learning clustering
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent.parent / "data" / "shopping_behavior_cleaned.csv"
    return pd.read_csv(data_path)

df = load_data()

# Interactive Configuration Panel
st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
st.markdown("### ⚙️ Clustering Configuration")
st.markdown("Customize your segmentation by selecting features and cluster count")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("**📊 Select Features for Analysis:**")
    feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)

    with feat_col1:
        use_age = st.checkbox("👤 Age", value=True)
    with feat_col2:
        use_purchase = st.checkbox("💰 Purchase Amount", value=True)
    with feat_col3:
        use_rating = st.checkbox("⭐ Review Rating", value=True)
    with feat_col4:
        use_previous = st.checkbox("📦 Previous Purchases", value=True)

with col2:
    st.markdown("**🎯 Number of Clusters:**")
    n_clusters = st.slider("", 2, 6, 3, help="Adjust the number of customer segments")
    st.metric("Segments", n_clusters, delta=None)

st.markdown('</div>', unsafe_allow_html=True)

# Prepare features for clustering
features = []
feature_names = []

if use_age:
    features.append(df["Age"].values)
    feature_names.append("Age")

if use_purchase:
    features.append(df["Purchase Amount (USD)"].values)
    feature_names.append("Purchase Amount")

if use_rating:
    features.append(df["Review Rating"].values)
    feature_names.append("Review Rating")

if use_previous:
    features.append(df["Previous Purchases"].values)
    feature_names.append("Previous Purchases")

# Check if at least 2 features are selected
if len(features) < 2:
    st.error("⚠️ Please select at least 2 features for clustering!")
    st.stop()

# Create feature matrix
X = np.column_stack(features)

# Perform clustering
@st.cache_data
def perform_clustering(X, n_clusters, random_state=42):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    return clusters, kmeans, scaler

clusters, kmeans, scaler = perform_clustering(X, n_clusters)

# Add clusters to dataframe
df_clustered = df.copy()
df_clustered["Cluster"] = clusters
df_clustered["Cluster_Label"] = df_clustered["Cluster"].apply(lambda x: f"Segment {x+1}")

st.markdown("---")

# Cluster Overview with Enhanced Metrics
st.markdown("""
<div class="section-header-custom">
    <span class="icon">📊</span>
    <h2>Segmentation Overview</h2>
</div>
""", unsafe_allow_html=True)

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric(
        label="👥 Total Customers",
        value=f"{len(df_clustered):,}",
        delta="Analyzed"
    )

with metric_col2:
    st.metric(
        label="🎯 Customer Segments",
        value=n_clusters,
        delta=f"{len(feature_names)} features"
    )

with metric_col3:
    largest_cluster = df_clustered["Cluster_Label"].value_counts().iloc[0]
    largest_pct = (largest_cluster / len(df_clustered)) * 100
    st.metric(
        label="📈 Largest Segment",
        value=f"{largest_pct:.1f}%",
        delta=f"{largest_cluster:,} customers"
    )

with metric_col4:
    inertia = kmeans.inertia_
    st.metric(
        label="🎲 Model Quality",
        value=f"{int(inertia):,}",
        delta="Inertia Score",
        help="Lower inertia indicates tighter clusters"
    )

st.markdown("---")

# Visual Distribution
st.markdown("""
<div class="section-header-custom">
    <span class="icon">📈</span>
    <h2>Segment Distribution</h2>
</div>
""", unsafe_allow_html=True)

dist_col1, dist_col2 = st.columns([1, 2])

with dist_col1:
    st.markdown('<div class="insight-card">', unsafe_allow_html=True)
    st.markdown("#### 📋 Segment Breakdown")
    cluster_counts = df_clustered["Cluster_Label"].value_counts().sort_index()

    for cluster, count in cluster_counts.items():
        percentage = (count / len(df_clustered)) * 100
        st.markdown(f"""
        <div style="padding: 0.75rem; margin: 0.5rem 0; background: #F8F9FA; border-radius: 0.5rem; border-left: 4px solid #FA812F;">
            <strong style="color: #2C3E50;">{cluster}</strong><br>
            <span style="color: #6C757D; font-size: 0.9rem;">{count:,} customers ({percentage:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with dist_col2:
    # Enhanced pie chart
    fig_dist = px.pie(
        values=cluster_counts.values,
        names=cluster_counts.index,
        title="Customer Segment Distribution",
        color_discrete_sequence=["#FA812F", "#FAB12F", "#DD0303", "#FEF3E2", "#FF9F5E", "#FFC166"],
        hole=0.4
    )
    fig_dist.update_traces(
        textposition='outside',
        textinfo='percent+label',
        marker=dict(line=dict(color='white', width=3))
    )
    fig_dist.update_layout(
        height=400,
        font=dict(family="Inter", color="#2C3E50"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")

# Interactive Scatter Plot
st.markdown("""
<div class="section-header-custom">
    <span class="icon">🎨</span>
    <h2>Visual Segment Analysis</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("**Customize your view by selecting different dimensions:**")

scatter_col1, scatter_col2 = st.columns(2)

with scatter_col1:
    x_axis = st.selectbox(
        "X-Axis Feature",
        ["Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases"],
        index=3,
        help="Select the horizontal axis"
    )

with scatter_col2:
    y_axis = st.selectbox(
        "Y-Axis Feature",
        ["Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases"],
        index=1,
        help="Select the vertical axis"
    )

# Enhanced scatter plot
fig_scatter = px.scatter(
    df_clustered,
    x=x_axis,
    y=y_axis,
    color="Cluster_Label",
    title=f"Customer Segments: {x_axis} vs {y_axis}",
    color_discrete_sequence=["#FA812F", "#FAB12F", "#DD0303", "#FEF3E2", "#FF9F5E", "#FFC166"],
    hover_data={
        "Age": True,
        "Gender": True,
        "Category": True,
        "Previous Purchases": True,
        "Purchase Amount (USD)": ':.2f',
        "Review Rating": ':.2f',
        "Cluster_Label": False
    },
    labels={"Cluster_Label": "Segment"}
)

fig_scatter.update_traces(
    marker=dict(
        size=10,
        opacity=0.7,
        line=dict(width=1, color='white')
    )
)
fig_scatter.update_layout(
    height=550,
    font=dict(family="Inter", color="#2C3E50"),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#F8F9FA',
    xaxis=dict(gridcolor='#E9ECEF'),
    yaxis=dict(gridcolor='#E9ECEF'),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#E9ECEF",
        borderwidth=1
    )
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# Segment Profiles
st.markdown("""
<div class="section-header-custom">
    <span class="icon">📋</span>
    <h2>Segment Profiles</h2>
</div>
""", unsafe_allow_html=True)

# Calculate cluster statistics
cluster_profiles = df_clustered.groupby("Cluster_Label").agg({
    "Age": "mean",
    "Purchase Amount (USD)": "mean",
    "Review Rating": "mean",
    "Previous Purchases": "mean",
    "Customer ID": "count"
}).round(2)

cluster_profiles.columns = ["Avg Age", "Avg Spending ($)", "Avg Rating", "Avg Previous Purchases", "Customer Count"]
cluster_profiles = cluster_profiles.reset_index()
cluster_profiles = cluster_profiles.rename(columns={"Cluster_Label": "Segment"})

st.dataframe(
    cluster_profiles,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Segment": st.column_config.TextColumn("Segment", width="medium"),
        "Avg Age": st.column_config.NumberColumn("Avg Age", format="%.1f years"),
        "Avg Spending ($)": st.column_config.NumberColumn("Avg Spending", format="$%.2f"),
        "Avg Rating": st.column_config.NumberColumn("Avg Rating", format="%.2f ⭐"),
        "Avg Previous Purchases": st.column_config.NumberColumn("Avg Previous", format="%.1f"),
        "Customer Count": st.column_config.NumberColumn("Customers", format="%d")
    }
)

st.markdown("---")

# Deep Dive Analysis
st.markdown("""
<div class="section-header-custom">
    <span class="icon">🔍</span>
    <h2>Deep Dive: Segment Analysis</h2>
</div>
""", unsafe_allow_html=True)

# Select cluster to analyze
selected_cluster = st.selectbox(
    "🎯 Select a segment for detailed analysis:",
    df_clustered["Cluster_Label"].unique(),
    help="Choose a segment to explore its characteristics in depth"
)

cluster_data = df_clustered[df_clustered["Cluster_Label"] == selected_cluster]

# Segment Header
st.markdown(f"""
<div style="background: linear-gradient(135deg, #FA812F 0%, #DD0303 100%); padding: 2rem; border-radius: 1rem; color: white; margin: 1rem 0; box-shadow: 0 4px 16px rgba(250, 129, 47, 0.3);">
    <h2 style="color: white; margin: 0;">{selected_cluster}</h2>
    <p style="font-size: 1.1rem; margin: 0.5rem 0 0 0; opacity: 0.95;">{len(cluster_data):,} customers ({len(cluster_data)/len(df_clustered)*100:.1f}% of total)</p>
</div>
""", unsafe_allow_html=True)

# Key Metrics for Selected Segment
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    avg_age = cluster_data["Age"].mean()
    st.metric(
        label="👤 Average Age",
        value=f"{avg_age:.1f}",
        delta="years"
    )

with metric_col2:
    avg_spending = cluster_data["Purchase Amount (USD)"].mean()
    global_avg = df["Purchase Amount (USD)"].mean()
    delta_spend = avg_spending - global_avg
    st.metric(
        label="💰 Avg Spending",
        value=f"${avg_spending:.2f}",
        delta=f"${delta_spend:+.2f} vs avg"
    )

with metric_col3:
    avg_rating = cluster_data["Review Rating"].mean()
    st.metric(
        label="⭐ Avg Rating",
        value=f"{avg_rating:.2f}",
        delta="satisfaction"
    )

with metric_col4:
    avg_previous = cluster_data["Previous Purchases"].mean()
    st.metric(
        label="📦 Avg Previous",
        value=f"{avg_previous:.1f}",
        delta="purchases"
    )

# Characteristics Visualization
char_col1, char_col2 = st.columns(2)

with char_col1:
    st.markdown(f"#### 👥 Gender Distribution")
    gender_dist = cluster_data["Gender"].value_counts()
    fig_gender = px.pie(
        values=gender_dist.values,
        names=gender_dist.index,
        color_discrete_sequence=["#FA812F", "#DD0303"],
        hole=0.4
    )
    fig_gender.update_traces(textposition='inside', textinfo='percent+label')
    fig_gender.update_layout(
        height=300,
        showlegend=False,
        font=dict(family="Inter", color="#2C3E50"),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_gender, use_container_width=True)

with char_col2:
    st.markdown(f"#### 📦 Top Product Categories")
    category_dist = cluster_data["Category"].value_counts().head(5)
    fig_category = px.bar(
        x=category_dist.values,
        y=category_dist.index,
        orientation='h',
        color=category_dist.values,
        color_continuous_scale=["#FEF3E2", "#FAB12F", "#FA812F", "#DD0303"],
        text=category_dist.values
    )
    fig_category.update_traces(textposition='outside')
    fig_category.update_layout(
        height=300,
        showlegend=False,
        xaxis_title="Count",
        yaxis_title="",
        font=dict(family="Inter", color="#2C3E50"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#F8F9FA'
    )
    st.plotly_chart(fig_category, use_container_width=True)

st.markdown("---")

# AI-Powered Insights
st.markdown("""
<div class="section-header-custom">
    <span class="icon">💡</span>
    <h2>AI-Generated Insights & Recommendations</h2>
</div>
""", unsafe_allow_html=True)

# Generate insights for each cluster
for cluster_num in range(n_clusters):
    cluster_label = f"Segment {cluster_num + 1}"
    cluster_subset = df_clustered[df_clustered["Cluster_Label"] == cluster_label]

    # Calculate characteristics
    avg_age = cluster_subset["Age"].mean()
    avg_spending = cluster_subset["Purchase Amount (USD)"].mean()
    avg_previous = cluster_subset["Previous Purchases"].mean()
    avg_rating = cluster_subset["Review Rating"].mean()

    # Calculate percentages
    pct_of_total = (len(cluster_subset) / len(df_clustered)) * 100

    # Determine characteristics
    age_level = "Young (18-34)" if avg_age < 35 else ("Middle-aged (35-54)" if avg_age < 55 else "Senior (55+)")
    spending_level = "Budget-conscious" if avg_spending < 50 else ("Moderate spenders" if avg_spending < 70 else "Premium buyers")
    loyalty_level = "New customers" if avg_previous < 15 else ("Regular shoppers" if avg_previous < 35 else "VIP loyalists")
    satisfaction_level = "Highly satisfied" if avg_rating >= 3.8 else "Moderately satisfied"

    # Color coding
    colors = ["#FA812F", "#FAB12F", "#DD0303", "#FEF3E2", "#FF9F5E", "#FFC166"]
    segment_color = colors[cluster_num % len(colors)]

    with st.expander(f"**{cluster_label}** - {len(cluster_subset):,} customers ({pct_of_total:.1f}%)", expanded=(cluster_num == 0)):
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {segment_color}15 0%, {segment_color}05 100%); padding: 1.5rem; border-radius: 0.75rem; border-left: 4px solid {segment_color}; margin: 1rem 0;">
            <h4 style="color: #2C3E50; margin-top: 0;">👤 Segment Profile</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0;">
                <div>
                    <strong style="color: {segment_color};">Age Group:</strong><br>
                    <span style="color: #495057;">{age_level}</span>
                </div>
                <div>
                    <strong style="color: {segment_color};">Spending Habit:</strong><br>
                    <span style="color: #495057;">{spending_level}</span>
                </div>
                <div>
                    <strong style="color: {segment_color};">Loyalty Level:</strong><br>
                    <span style="color: #495057;">{loyalty_level}</span>
                </div>
                <div>
                    <strong style="color: {segment_color};">Satisfaction:</strong><br>
                    <span style="color: #495057;">{satisfaction_level}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.metric("Avg Age", f"{avg_age:.1f} yrs")
        with metric_col2:
            st.metric("Avg Spending", f"${avg_spending:.2f}")
        with metric_col3:
            st.metric("Avg Previous", f"{avg_previous:.1f}")
        with metric_col4:
            st.metric("Avg Rating", f"{avg_rating:.2f} ⭐")

        # Top category and gender split
        top_category = cluster_subset['Category'].value_counts().index[0]
        gender_split = cluster_subset['Gender'].value_counts().to_dict()

        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border: 1px solid #E9ECEF;">
            <strong style="color: #2C3E50;">📦 Top Category:</strong> {top_category}<br>
            <strong style="color: #2C3E50;">👥 Gender Split:</strong> {' | '.join([f'{k}: {v}' for k, v in gender_split.items()])}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🎯 Actionable Recommendations")

        recommendations = []

        # Generate recommendations based on characteristics
        if avg_previous < 15:
            recommendations.append("🎁 **Welcome Campaign**: Create onboarding sequences with first-purchase incentives")
            recommendations.append("📧 **Email Nurturing**: Send educational content about product benefits")
        elif avg_previous >= 35:
            recommendations.append("💎 **VIP Program**: Offer exclusive early access to new collections")
            recommendations.append("🎊 **Loyalty Rewards**: Implement points-based reward system")

        if avg_spending > 70:
            recommendations.append("👑 **Premium Focus**: Showcase high-end products and luxury experiences")
            recommendations.append("🛍️ **Personalized Service**: Provide dedicated customer support")
        elif avg_spending < 50:
            recommendations.append("💰 **Value Messaging**: Highlight discounts, bundles, and value deals")
            recommendations.append("📊 **Bundle Offers**: Create attractive multi-item packages")

        if avg_rating < 3.5:
            recommendations.append("📞 **Feedback Loop**: Implement post-purchase satisfaction surveys")
            recommendations.append("🔧 **Service Improvement**: Focus on quality control and customer service")
        else:
            recommendations.append("⭐ **Review Incentives**: Encourage satisfied customers to leave reviews")
            recommendations.append("📣 **Brand Advocacy**: Convert satisfied customers into brand ambassadors")

        for rec in recommendations:
            st.markdown(f"- {rec}")

st.markdown("---")

# 3D Visualization
if len(features) >= 3:
    st.markdown("""
    <div class="section-header-custom">
        <span class="icon">🌐</span>
        <h2>3D Segment Visualization</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Explore segments in three dimensions:**")

    available_features = ["Age", "Purchase Amount (USD)", "Review Rating", "Previous Purchases"]

    col1, col2, col3 = st.columns(3)
    with col1:
        x_3d = st.selectbox("X-Axis (3D)", available_features, index=0, key="3d_x")
    with col2:
        y_3d = st.selectbox("Y-Axis (3D)", available_features, index=1, key="3d_y")
    with col3:
        z_3d = st.selectbox("Z-Axis (3D)", available_features, index=3, key="3d_z")

    fig_3d = px.scatter_3d(
        df_clustered,
        x=x_3d,
        y=y_3d,
        z=z_3d,
        color="Cluster_Label",
        title=f"3D Segment View: {x_3d} vs {y_3d} vs {z_3d}",
        color_discrete_sequence=["#FA812F", "#FAB12F", "#DD0303", "#FEF3E2", "#FF9F5E", "#FFC166"],
        opacity=0.7,
        labels={"Cluster_Label": "Segment"}
    )

    fig_3d.update_traces(marker=dict(size=5, line=dict(width=0.5, color='white')))
    fig_3d.update_layout(
        height=650,
        font=dict(family="Inter", color="#2C3E50"),
        scene=dict(
            xaxis=dict(backgroundcolor="#F8F9FA", gridcolor="#E9ECEF"),
            yaxis=dict(backgroundcolor="#F8F9FA", gridcolor="#E9ECEF"),
            zaxis=dict(backgroundcolor="#F8F9FA", gridcolor="#E9ECEF")
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E9ECEF",
            borderwidth=1
        )
    )

    st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")

# Export Section
st.markdown("""
<div class="section-header-custom">
    <span class="icon">📥</span>
    <h2>Export Segmentation Data</h2>
</div>
""", unsafe_allow_html=True)

export_col1, export_col2, export_col3 = st.columns(3)

with export_col1:
    st.download_button(
        label="📊 Download Full Dataset",
        data=df_clustered.to_csv(index=False),
        file_name="shopbis_segmented_customers.csv",
        mime="text/csv",
        use_container_width=True,
        help="Download complete customer data with segment assignments"
    )

with export_col2:
    st.download_button(
        label="📋 Download Segment Profiles",
        data=cluster_profiles.to_csv(index=False),
        file_name="shopbis_segment_profiles.csv",
        mime="text/csv",
        use_container_width=True,
        help="Download aggregated statistics for each segment"
    )

with export_col3:
    # Export only selected segment
    selected_data = cluster_data[["Customer ID", "Age", "Gender", "Purchase Amount (USD)", "Review Rating", "Previous Purchases", "Category", "Cluster_Label"]]
    st.download_button(
        label=f"🎯 Download {selected_cluster}",
        data=selected_data.to_csv(index=False),
        file_name=f"shopbis_{selected_cluster.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True,
        help="Download data for the currently selected segment"
    )

# Footer
st.markdown("""
<div style="margin-top: 3rem; padding: 2rem; background: linear-gradient(135deg, #2C3E50 0%, #1a252f 100%); border-radius: 1rem; text-align: center; color: white;">
    <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">
        Powered by K-Means Clustering Algorithm | Built with Scikit-learn
    </p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem; color: #FAB12F; font-weight: 600;">
        Developed by Kent Paulo Delgado
    </p>
</div>
""", unsafe_allow_html=True)
