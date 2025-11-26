"""
Model Performance Page
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.styling import add_sidebar_branding, add_footer, apply_custom_css

st.set_page_config(page_title="Model Performance", page_icon="🤖", layout="wide")

# Apply styling
apply_custom_css()
add_sidebar_branding()

st.title("🤖 Model Performance")
st.markdown("Comprehensive comparison of all forecasting models")

# Load model comparison
try:
    comparison_file = config.MODELS_DIR / 'model_comparison.csv'
    if comparison_file.exists():
        df_comparison = pd.read_csv(comparison_file, index_col=0)
        
        st.markdown("## 📊 Model Comparison Table")
        
        # Style the dataframe
        styled_df = df_comparison.style.background_gradient(cmap='RdYlGn_r', subset=['rmse', 'mae', 'mape', 'wape', 'smape'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Highlight best model
        best_model = df_comparison['wape'].idxmin()
        best_wape = df_comparison.loc[best_model, 'wape']
        
        st.success(f"🏆 **Best Model: {best_model}** with WAPE of **{best_wape:.2f}%**")
        
        # Visualizations
        st.markdown("---")
        st.markdown("## 📈 Performance Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Error metrics comparison
            fig_error = go.Figure()
            fig_error.add_trace(go.Bar(
                x=df_comparison.index,
                y=df_comparison['rmse'],
                name='RMSE',
                marker_color='#00CC96'
            ))
            fig_error.add_trace(go.Bar(
                x=df_comparison.index,
                y=df_comparison['mae'],
                name='MAE',
                marker_color='#19D3F3'
            ))
            fig_error.update_layout(
                title='Error Metrics Comparison (Lower is Better)',
                xaxis_title='Model',
                yaxis_title='Error Value',
                barmode='group',
                template='plotly_dark'
            )
            st.plotly_chart(fig_error, use_container_width=True)
        
        with col2:
            # Percentage metrics comparison
            fig_pct = go.Figure()
            for metric in ['mape', 'wape', 'smape']:
                fig_pct.add_trace(go.Bar(
                    x=df_comparison.index,
                    y=df_comparison[metric],
                    name=metric.upper()
                ))
            fig_pct.update_layout(
                title='Percentage Metrics Comparison (Lower is Better)',
                xaxis_title='Model',
                yaxis_title='Percentage (%)',
                barmode='group',
                template='plotly_dark'
            )
            st.plotly_chart(fig_pct, use_container_width=True)
        
        # Radar chart
        st.markdown("### Model Performance Radar")
        
        # Normalize metrics for radar chart (inverse for error metrics)
        df_norm = df_comparison.copy()
        for col in df_norm.columns:
            max_val = df_norm[col].max()
            df_norm[col] = 100 * (1 - df_norm[col] / max_val)  # Inverse normalization
        
        fig_radar = go.Figure()
        
        for model in df_norm.index:
            fig_radar.add_trace(go.Scatterpolar(
                r=df_norm.loc[model].values,
                theta=df_norm.columns,
                fill='toself',
                name=model
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=True,
            title='Model Performance Radar (Higher is Better)',
            template='plotly_dark'
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
    else:
        st.warning("⚠️ Model comparison file not found. Please train the models first.")
        st.code("python -m src.train", language="bash")

except Exception as e:
    st.error(f"Error loading model comparison: {str(e)}")
    st.info("Please train the models first by running: `python -m src.train`")

# Feature importance section
st.markdown("---")
st.markdown("## 💡 Feature Importance")

try:
    # Try to load LightGBM feature importance
    feat_imp_file = config.MODELS_DIR / 'lightgbm_feature_importance.csv'
    if feat_imp_file.exists():
        df_feat_imp = pd.read_csv(feat_imp_file)
        
        # Show top 20 features
        df_top_features = df_feat_imp.head(20)
        
        fig_feat = px.bar(
            df_top_features,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 20 Most Important Features (LightGBM)',
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            template='plotly_dark',
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig_feat.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        st.plotly_chart(fig_feat, use_container_width=True)
        
        # Feature categories
        st.markdown("### Feature Categories")
        
        # Categorize features
        def categorize_feature(feat_name):
            if 'lag' in feat_name.lower():
                return 'Lag Features'
            elif 'rolling' in feat_name.lower():
                return 'Rolling Statistics'
            elif 'oil' in feat_name.lower():
                return 'Oil Prices'
            elif 'transaction' in feat_name.lower():
                return 'Transactions'
            elif any(x in feat_name.lower() for x in ['day', 'week', 'month', 'year', 'quarter']):
                return 'Temporal Features'
            elif 'promotion' in feat_name.lower() or 'onpromotion' in feat_name.lower():
                return 'Promotions'
            elif 'holiday' in feat_name.lower():
                return 'Holidays'
            else:
                return 'Other'
        
        df_feat_imp['category'] = df_feat_imp['feature'].apply(categorize_feature)
        
        category_importance = df_feat_imp.groupby('category')['importance'].sum().sort_values(ascending=False)
        
        fig_cat = px.pie(
            values=category_importance.values,
            names=category_importance.index,
            title='Feature Importance by Category',
            template='plotly_dark',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        st.plotly_chart(fig_cat, use_container_width=True)
        
    else:
        st.info("Feature importance will be available after training gradient boosting models")
        
except Exception as e:
    st.error(f"Error loading feature importance: {str(e)}")

# Model details
st.markdown("---")
st.markdown("## 📋 Model Details")

model_details = {
    "LightGBM": {
        "Description": "Gradient boosting with leaf-wise tree growth and native categorical support",
        "Strengths": "Fast training, handles categorical features natively, excellent accuracy, feature importance",
        "Weaknesses": "Requires feature engineering, can overfit with small datasets",
        "Use Case": "Production forecasting with many features and categorical variables",
        "Performance": "13.53% WAPE - Best model for this dataset"
    },
    "XGBoost": {
        "Description": "Regularized gradient boosting with level-wise tree growth",
        "Strengths": "Robust regularization, handles missing values well, stable performance",
        "Weaknesses": "Slower than LightGBM, higher memory usage",
        "Use Case": "When robustness and stability are critical",
        "Performance": "13.91% WAPE - Excellent production-grade performance"
    }
}

for model_name, details in model_details.items():
    with st.expander(f"**{model_name}**"):
        st.markdown(f"**Description:** {details['Description']}")
        st.markdown(f"**✅ Strengths:** {details['Strengths']}")
        st.markdown(f"**⚠️ Weaknesses:** {details['Weaknesses']}")
        st.markdown(f"**🎯 Use Case:** {details['Use Case']}")
        st.markdown(f"**📊 Performance:** {details['Performance']}")

# Footer
add_footer()

