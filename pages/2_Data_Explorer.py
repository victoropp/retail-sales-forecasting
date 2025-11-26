"""
Data Explorer Page
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils.styling import add_sidebar_branding, add_footer, apply_custom_css

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

# Apply styling
apply_custom_css()
add_sidebar_branding()

st.title("📊 Data Explorer")
st.markdown("Explore the retail sales dataset and discover patterns")

# Load data
@st.cache_data
def load_sample_data():
    """Load a sample of the training data"""
    try:
        import data_loader
        df = data_loader.load_train_data()
        # Sample for faster loading
        return df.sample(n=min(100000, len(df)), random_state=42)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

df = load_sample_data()

if df is not None:
    # Dataset overview
    st.markdown("## 📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Stores", df['store_nbr'].nunique())
    with col3:
        st.metric("Product Families", df['family'].nunique())
    with col4:
        st.metric("Date Range", f"{(df['date'].max() - df['date'].min()).days} days")
    
    # Sample data
    with st.expander("📄 View Sample Data"):
        st.dataframe(df.head(100), use_container_width=True)
    
    # Sales trends
    st.markdown("---")
    st.markdown("## 📈 Sales Trends")
    
    # Aggregate daily sales
    daily_sales = df.groupby('date')['sales'].sum().reset_index()
    
    fig_trend = px.line(
        daily_sales,
        x='date',
        y='sales',
        title='Daily Total Sales Over Time',
        labels={'sales': 'Total Sales', 'date': 'Date'},
        template='plotly_dark'
    )
    fig_trend.update_traces(line_color='#00CC96')
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Sales by product family
    st.markdown("---")
    st.markdown("## 🏷️ Sales by Product Family")
    
    family_sales = df.groupby('family')['sales'].sum().sort_values(ascending=False).reset_index()
    
    fig_family = px.bar(
        family_sales.head(15),
        x='sales',
        y='family',
        orientation='h',
        title='Top 15 Product Families by Total Sales',
        labels={'sales': 'Total Sales', 'family': 'Product Family'},
        template='plotly_dark',
        color='sales',
        color_continuous_scale='Viridis'
    )
    fig_family.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_family, use_container_width=True)
    
    # Day of week patterns
    st.markdown("---")
    st.markdown("## 📅 Weekly Patterns")
    
    df['day_of_week'] = pd.to_datetime(df['date']).dt.day_name()
    dow_sales = df.groupby('day_of_week')['sales'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    
    fig_dow = px.bar(
        x=dow_sales.index,
        y=dow_sales.values,
        title='Average Sales by Day of Week',
        labels={'x': 'Day of Week', 'y': 'Average Sales'},
        template='plotly_dark',
        color=dow_sales.values,
        color_continuous_scale='Teal'
    )
    st.plotly_chart(fig_dow, use_container_width=True)
    
    # Promotions impact
    st.markdown("---")
    st.markdown("## 🎯 Promotions Impact")
    
    promo_sales = df.groupby('onpromotion')['sales'].mean()
    
    fig_promo = go.Figure(data=[
        go.Bar(
            x=['No Promotion', 'On Promotion'],
            y=promo_sales.values,
            marker_color=['#FF6692', '#00CC96']
        )
    ])
    fig_promo.update_layout(
        title='Average Sales: Promotion vs No Promotion',
        xaxis_title='Promotion Status',
        yaxis_title='Average Sales',
        template='plotly_dark'
    )
    st.plotly_chart(fig_promo, use_container_width=True)
    
    # Interactive filters
    st.markdown("---")
    st.markdown("## 🔍 Interactive Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_family = st.selectbox(
            "Select Product Family",
            options=sorted(df['family'].unique())
        )
    
    with col2:
        selected_store = st.selectbox(
            "Select Store",
            options=sorted(df['store_nbr'].unique())
        )
    
    # Filter data
    filtered_df = df[
        (df['family'] == selected_family) &
        (df['store_nbr'] == selected_store)
    ]
    
    if len(filtered_df) > 0:
        # Time series for selected combination
        fig_filtered = px.line(
            filtered_df.sort_values('date'),
            x='date',
            y='sales',
            title=f'Sales Trend: {selected_family} at Store {selected_store}',
            labels={'sales': 'Sales', 'date': 'Date'},
            template='plotly_dark'
        )
        fig_filtered.update_traces(line_color='#19D3F3')
        st.plotly_chart(fig_filtered, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Sales", f"{filtered_df['sales'].mean():.2f}")
        with col2:
            st.metric("Max Sales", f"{filtered_df['sales'].max():.2f}")
        with col3:
            st.metric("Min Sales", f"{filtered_df['sales'].min():.2f}")
        with col4:
            st.metric("Std Dev", f"{filtered_df['sales'].std():.2f}")
    else:
        st.warning("No data available for this combination")

else:
    st.error("Unable to load data. Please ensure the dataset is in the correct location.")
    st.info(f"Expected data location: {config.TRAIN_FILE}")

# Footer
add_footer()
