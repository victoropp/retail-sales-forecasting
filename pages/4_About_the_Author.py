"""
About the Author page for Retail Sales Forecasting Dashboard.
Displays professional credentials, expertise, and contact information.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.styling import add_sidebar_branding, add_footer, apply_custom_css

# Page configuration
st.set_page_config(
    page_title="About the Author | Retail Sales Forecasting",
    page_icon="👨‍💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

# Sidebar branding
add_sidebar_branding()

# Main content
st.markdown(
    """
    <div style='background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #3d7ab5 100%);
                padding: 40px; border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='color: white; margin: 0; font-size: 2.5rem;'>👨‍💻 About the Author</h1>
        <p style='color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 1.1rem;'>
            Data Scientist | Machine Learning Engineer | Finance Professional
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Profile section
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #1e3a5f 0%, #3d7ab5 100%);
                    padding: 30px; border-radius: 15px; text-align: center;'>
            <div style='background: white; width: 120px; height: 120px;
                        border-radius: 50%; margin: 0 auto 15px;
                        display: flex; align-items: center; justify-content: center;
                        box-shadow: 0 4px 20px rgba(0,0,0,0.2);'>
                <span style='font-size: 48px; font-weight: bold; color: #1e3a5f;'>VC</span>
            </div>
            <h2 style='color: white; margin: 0 0 5px 0;'>Victor Collins Oppon</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0;'>FCCA | MBA | BSc</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown("### Professional Summary")
    st.markdown(
        """
        I am a **Data Scientist and Machine Learning Engineer** with a unique blend of technical expertise
        and financial acumen. With credentials including **FCCA (Fellow Chartered Certified Accountant)**,
        **MBA in Finance**, and a strong foundation in data science, I bring a distinctive perspective to
        solving complex business problems through data-driven solutions.

        My expertise spans **time series forecasting**, **predictive analytics**, and **business intelligence**,
        with a particular focus on applications in retail, finance, and healthcare sectors. I specialize in
        building production-ready machine learning systems that deliver measurable business value.

        This Retail Sales Forecasting project demonstrates my ability to combine advanced ML techniques
        (LightGBM, XGBoost) with comprehensive feature engineering (56 features) to achieve **industry-excellent
        performance** (13.5% WAPE) while maintaining interpretability and business relevance.
        """
    )

st.markdown("---")

# Technical Expertise
st.markdown("### 🛠️ Technical Expertise")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div style='background: #f8f9fa; padding: 20px; border-radius: 10px; height: 100%;
                    border-top: 4px solid #1e3a5f;'>
            <h4 style='color: #1e3a5f; margin-bottom: 15px;'>🤖 Machine Learning</h4>
            <ul style='color: #4a4a4a; padding-left: 20px; margin: 0;'>
                <li>LightGBM & XGBoost</li>
                <li>Scikit-learn</li>
                <li>TensorFlow & Keras</li>
                <li>Feature Engineering</li>
                <li>Model Optimization</li>
                <li>Ensemble Methods</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div style='background: #f8f9fa; padding: 20px; border-radius: 10px; height: 100%;
                    border-top: 4px solid #2d5a87;'>
            <h4 style='color: #2d5a87; margin-bottom: 15px;'>📈 Time Series & Analytics</h4>
            <ul style='color: #4a4a4a; padding-left: 20px; margin: 0;'>
                <li>Demand Forecasting</li>
                <li>Prophet & ARIMA</li>
                <li>Temporal Features</li>
                <li>Lag & Rolling Statistics</li>
                <li>Seasonal Decomposition</li>
                <li>Business Intelligence</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div style='background: #f8f9fa; padding: 20px; border-radius: 10px; height: 100%;
                    border-top: 4px solid #3d7ab5;'>
            <h4 style='color: #3d7ab5; margin-bottom: 15px;'>🚀 Deployment & Tools</h4>
            <ul style='color: #4a4a4a; padding-left: 20px; margin: 0;'>
                <li>Streamlit & Plotly</li>
                <li>FastAPI</li>
                <li>Docker & Kubernetes</li>
                <li>AWS / Azure / GCP</li>
                <li>Git & CI/CD</li>
                <li>SQL & NoSQL</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Career Highlights
st.markdown("### 🏆 Career Highlights")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #e8f4f8 0%, #d1e8f0 100%);
                    padding: 20px; border-radius: 10px; margin-bottom: 15px;'>
            <h4 style='color: #1e3a5f; margin: 0 0 10px 0;'>📊 Data Science Excellence</h4>
            <p style='color: #4a4a4a; margin: 0; font-size: 0.9rem;'>
                Built production ML systems achieving state-of-the-art performance across
                multiple domains including retail forecasting, fraud detection, and healthcare analytics.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #e8f4f8 0%, #d1e8f0 100%);
                    padding: 20px; border-radius: 10px;'>
            <h4 style='color: #1e3a5f; margin: 0 0 10px 0;'>💼 Finance & Accounting</h4>
            <p style='color: #4a4a4a; margin: 0; font-size: 0.9rem;'>
                FCCA certified with MBA in Finance, bringing rigorous analytical thinking
                and business acumen to data science projects.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #e8f4f8 0%, #d1e8f0 100%);
                    padding: 20px; border-radius: 10px; margin-bottom: 15px;'>
            <h4 style='color: #1e3a5f; margin: 0 0 10px 0;'>🎯 Business Impact Focus</h4>
            <p style='color: #4a4a4a; margin: 0; font-size: 0.9rem;'>
                Specialized in translating ML models into measurable business outcomes,
                including ROI analysis and cost optimization strategies.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #e8f4f8 0%, #d1e8f0 100%);
                    padding: 20px; border-radius: 10px;'>
            <h4 style='color: #1e3a5f; margin: 0 0 10px 0;'>🌐 End-to-End Solutions</h4>
            <p style='color: #4a4a4a; margin: 0; font-size: 0.9rem;'>
                Full-stack ML capability from data engineering and model development
                to deployment and monitoring in production environments.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Contact Section
st.markdown("### 📬 Get in Touch")

st.markdown(
    """
    <div style='background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
                padding: 30px; border-radius: 12px; text-align: center;'>
        <p style='color: white; font-size: 1.1rem; margin-bottom: 20px;'>
            Interested in collaboration or have questions about this project?
        </p>
        <div style='display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;'>
            <a href='mailto:victoropp@gmail.com' target='_blank'
               style='background: white; color: #1e3a5f; padding: 12px 25px; border-radius: 8px;
                      text-decoration: none; font-weight: bold; display: inline-block;'>
                📧 Email Me
            </a>
            <a href='https://github.com/victoropp' target='_blank'
               style='background: white; color: #1e3a5f; padding: 12px 25px; border-radius: 8px;
                      text-decoration: none; font-weight: bold; display: inline-block;'>
                💻 GitHub
            </a>
            <a href='https://www.linkedin.com/in/victor-collins-oppon-fcca-mba-bsc-01541019/' target='_blank'
               style='background: white; color: #1e3a5f; padding: 12px 25px; border-radius: 8px;
                      text-decoration: none; font-weight: bold; display: inline-block;'>
                🔗 LinkedIn
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Footer
add_footer()
