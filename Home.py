"""
Retail Sales Forecasting - Streamlit Dashboard
Home Page
"""
import streamlit as st
import sys
from pathlib import Path

# Add src and parent to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from utils.styling import add_sidebar_branding, add_footer, apply_custom_css

# Page config
st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

# Additional page-specific CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3a5f 0%, #3d7ab5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.3rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1e3a5f;
        margin-bottom: 1rem;
    }
    .highlight-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #3d7ab5 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with branding
add_sidebar_branding()

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Navigation")
    st.markdown("""
    **Explore the Dashboard:**
    - 📊 Data Exploration
    - 🤖 Model Performance
    - 💼 Business Impact
    - 👨‍💻 About the Author
    """)

    st.markdown("---")
    st.markdown("""
    **Tech Stack**
    - LightGBM & XGBoost
    - 56 Engineered Features
    - Time Series Validation
    - Production-Ready Pipeline
    """)

# Main content
st.markdown('<h1 class="main-header">📈 Retail Sales Forecasting</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Production-Grade Gradient Boosting for Multi-Store Sales Prediction</p>', unsafe_allow_html=True)

# Hero section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ## 🎯 Project Overview
    
    This production-grade forecasting platform predicts sales for a major Ecuadorian grocery retailer 
    with **54 stores** and **33 product families**. Using advanced gradient boosting techniques, 
    we achieve **industry-leading 13.5% WAPE** for **16-day ahead forecasts**.
    
    ### Key Achievements
    """)
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #00CC96; margin: 0;">13.5%</h3>
            <p style="margin: 0.5rem 0 0 0;">WAPE Score</p>
            <small style="color: #888;">Production-grade accuracy</small>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #19D3F3; margin: 0;">2 Models</h3>
            <p style="margin: 0.5rem 0 0 0;">Gradient Boosting</p>
            <small style="color: #888;">LightGBM & XGBoost</small>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #AB63FA; margin: 0;">3M+</h3>
            <p style="margin: 0.5rem 0 0 0;">Training Records</p>
            <small style="color: #888;">2013-2017 data</small>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="highlight-box">
        <h3 style="margin-top: 0;">🚀 Unique Features</h3>
        <ul style="margin-bottom: 0;">
            <li><strong>Advanced Feature Engineering</strong><br/>
            <small>56 temporal, lag, and external features</small></li>
            <li><strong>Production-Ready Models</strong><br/>
            <small>LightGBM & XGBoost optimized</small></li>
            <li><strong>Multi-Horizon Forecasting</strong><br/>
            <small>16-day simultaneous predictions</small></li>
            <li><strong>Comprehensive Evaluation</strong><br/>
            <small>5 metrics + feature importance</small></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Dataset section
st.markdown("---")
st.markdown("## 📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Stores", "54", help="Across Ecuador in 22 cities")
with col2:
    st.metric("Product Families", "33", help="From Grocery to Electronics")
with col3:
    st.metric("Training Records", "3M+", help="2013-2017 daily sales")
with col4:
    st.metric("Features", "56", help="Engineered temporal & external features")

# Model comparison section
st.markdown("---")
st.markdown("## 🤖 Model Comparison")

st.markdown("""
We trained and compared **gradient boosting models** with comprehensive feature engineering 
to achieve production-grade forecasting accuracy:
""")

model_data = {
    "Model": ["LightGBM", "XGBoost"],
    "Type": ["Gradient Boosting", "Gradient Boosting"],
    "WAPE": ["13.53%", "13.91%"],
    "Key Feature": [
        "Fast training, categorical support",
        "Robust regularization, handles missing values"
    ],
    "Training Time": ["~15 min", "~20 min"],
    "Interpretability": ["⭐⭐⭐⭐", "⭐⭐⭐⭐"]
}

st.table(model_data)

# Business impact section
st.markdown("---")
st.markdown("## 💼 Business Impact")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Expected Outcomes
    
    - **15-25% reduction** in inventory waste
    - **30-40% fewer stockouts** through better planning
    - **10-20% improvement** in labor efficiency
    - **$500K+ annual savings** for a mid-size retailer
    
    ### Use Cases
    
    - 📦 **Inventory Optimization**: Right products, right quantities
    - 👥 **Staffing Planning**: Match workforce to demand
    - 🎯 **Promotional Planning**: Optimize discount timing
    - 🚚 **Supply Chain**: Reduce lead times and costs
    """)

with col2:
    st.markdown("""
    ### Industry Applications
    
    This forecasting approach extends beyond retail:
    
    - **E-commerce**: Demand forecasting for logistics
    - **Manufacturing**: Production planning & scheduling
    - **Energy**: Load forecasting for grid management
    - **Healthcare**: Patient volume prediction
    - **Finance**: Transaction volume forecasting
    
    ### Technical Highlights
    
    - ✅ Advanced gradient boosting (LightGBM & XGBoost)
    - ✅ 56 engineered features (lag, rolling, temporal, external)
    - ✅ Production-ready with model versioning
    - ✅ Comprehensive evaluation (5 metrics)
    """)

# Tech stack section
st.markdown("---")
st.markdown("## 🛠️ Tech Stack")

tech_col1, tech_col2, tech_col3 = st.columns(3)

with tech_col1:
    st.markdown("""
    **Machine Learning**
    - LightGBM & XGBoost
    - Scikit-learn
    - Time Series CV
    - Feature Engineering
    """)

with tech_col2:
    st.markdown("""
    **Data Processing**
    - Pandas & NumPy
    - Feature Engineering
    - Time Series CV
    - Data Validation
    """)

with tech_col3:
    st.markdown("""
    **Visualization**
    - Streamlit
    - Plotly
    - Matplotlib & Seaborn
    - Interactive Dashboards
    """)

# Footer
add_footer()
