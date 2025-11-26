"""
Styling utilities for Retail Sales Forecasting Dashboard.
Provides consistent branding, sidebar elements, and footer across all pages.
"""

import streamlit as st


def add_sidebar_branding():
    """Add professional branding card to sidebar with author info."""
    st.sidebar.markdown(
        """
        <div style='background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #3d7ab5 100%);
                    padding: 20px; border-radius: 12px; text-align: center;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-bottom: 20px;'>
            <div style='background: white; width: 70px; height: 70px;
                        border-radius: 50%; margin: 0 auto 12px;
                        display: flex; align-items: center; justify-content: center;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.15);'>
                <span style='font-size: 28px; font-weight: bold; color: #1e3a5f;'>VC</span>
            </div>
            <h3 style='color: white; margin: 0 0 5px 0; font-size: 1.1rem;'>Victor Collins Oppon</h3>
            <p style='color: rgba(255,255,255,0.9); margin: 0 0 8px 0; font-size: 0.85rem;'>
                Data Scientist | ML Engineer
            </p>
            <p style='color: rgba(255,255,255,0.75); margin: 0; font-size: 0.75rem;'>
                FCCA | MBA | BSc
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Expertise section
    st.sidebar.markdown(
        """
        <div style='background: #f8f9fa; padding: 15px; border-radius: 8px;
                    border-left: 4px solid #1e3a5f; margin-bottom: 15px;'>
            <h4 style='color: #1e3a5f; margin: 0 0 10px 0; font-size: 0.9rem;'>Core Expertise</h4>
            <ul style='color: #4a4a4a; margin: 0; padding-left: 18px; font-size: 0.8rem; line-height: 1.6;'>
                <li>Time Series Forecasting</li>
                <li>Gradient Boosting (LightGBM, XGBoost)</li>
                <li>Feature Engineering</li>
                <li>Business Analytics & ROI</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Social links
    st.sidebar.markdown(
        """
        <div style='text-align: center; padding: 10px 0;'>
            <a href='https://github.com/victoropp' target='_blank'
               style='color: #1e3a5f; text-decoration: none; margin: 0 8px; font-size: 0.85rem;'>
                GitHub
            </a>
            <span style='color: #ccc;'>|</span>
            <a href='https://www.linkedin.com/in/victor-collins-oppon-fcca-mba-bsc-01541019/' target='_blank'
               style='color: #1e3a5f; text-decoration: none; margin: 0 8px; font-size: 0.85rem;'>
                LinkedIn
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )


def add_footer():
    """Add professional footer with attribution and tech stack."""
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 20px 0; color: #6c757d;">
            <p style="margin: 0 0 8px 0; font-size: 0.95rem;">
                <strong style="color: #1e3a5f;">Retail Sales Forecasting Dashboard</strong>
            </p>
            <p style="margin: 0 0 5px 0; font-size: 0.85rem;">
                Developed by <strong style="color: #2d5a87;">Victor Collins Oppon</strong> | Data Scientist
            </p>
            <p style="margin: 0; font-size: 0.8rem; color: #adb5bd;">
                Built with Python • LightGBM • XGBoost • Streamlit • Plotly • Pandas
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_page_config():
    """Get standard page configuration for consistency."""
    return {
        "page_title": "Retail Sales Forecasting",
        "page_icon": "📈",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }


def apply_custom_css():
    """Apply custom CSS styling for consistent look across pages."""
    st.markdown(
        """
        <style>
        /* Main container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Metric card styling */
        .metric-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #1e3a5f;
            margin-bottom: 15px;
        }

        /* Header gradient */
        .gradient-header {
            background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #3d7ab5 100%);
            padding: 30px;
            border-radius: 12px;
            color: white;
            margin-bottom: 25px;
        }

        /* Info box styling */
        .info-box {
            background-color: #e7f3ff;
            border-left: 4px solid #1e3a5f;
            padding: 15px;
            border-radius: 0 8px 8px 0;
            margin: 15px 0;
        }

        /* Success box styling */
        .success-box {
            background-color: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            border-radius: 0 8px 8px 0;
            margin: 15px 0;
        }

        /* Warning box styling */
        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            border-radius: 0 8px 8px 0;
            margin: 15px 0;
        }

        /* Link styling */
        a {
            color: #1e3a5f;
        }

        a:hover {
            color: #3d7ab5;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
