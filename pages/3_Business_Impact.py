"""
Business Impact Page
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.styling import add_sidebar_branding, add_footer, apply_custom_css

st.set_page_config(page_title="Business Impact", page_icon="💼", layout="wide")

# Apply styling
apply_custom_css()
add_sidebar_branding()

st.title("💼 Business Impact Analysis")
st.markdown("Quantify the value of accurate sales forecasting")

# ROI Calculator
st.markdown("## 🧮 ROI Calculator")
st.markdown("Customize the assumptions to calculate potential business impact")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Business Assumptions")
    
    avg_product_value = st.number_input(
        "Average Product Value ($)",
        min_value=1.0,
        max_value=1000.0,
        value=15.0,
        step=1.0,
        help="Average value per product unit"
    )
    
    daily_sales_volume = st.number_input(
        "Daily Sales Volume (units)",
        min_value=100,
        max_value=100000,
        value=10000,
        step=100,
        help="Average daily sales across all stores"
    )
    
    inventory_holding_cost_pct = st.slider(
        "Inventory Holding Cost (%/year)",
        min_value=5.0,
        max_value=30.0,
        value=15.0,
        step=1.0,
        help="Annual cost of holding inventory as % of product value"
    )
    
    stockout_cost_multiplier = st.slider(
        "Stockout Cost Multiplier",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="Lost revenue multiplier for stockouts (includes lost sales + customer dissatisfaction)"
    )

with col2:
    st.markdown("### Forecasting Improvements")
    
    current_forecast_error = st.slider(
        "Current Forecast Error (MAPE %)",
        min_value=10.0,
        max_value=50.0,
        value=25.0,
        step=1.0,
        help="Your current forecasting accuracy"
    )
    
    new_forecast_error = st.slider(
        "New Forecast Error (MAPE %)",
        min_value=5.0,
        max_value=30.0,
        value=12.0,
        step=1.0,
        help="Expected accuracy with our models"
    )
    
    waste_reduction_pct = st.slider(
        "Waste Reduction (%)",
        min_value=0.0,
        max_value=50.0,
        value=20.0,
        step=1.0,
        help="Reduction in inventory waste/spoilage"
    )
    
    stockout_reduction_pct = st.slider(
        "Stockout Reduction (%)",
        min_value=0.0,
        max_value=50.0,
        value=35.0,
        step=1.0,
        help="Reduction in stockout incidents"
    )

# Calculations
st.markdown("---")
st.markdown("## 📊 Impact Analysis")

# Annual metrics
annual_sales_volume = daily_sales_volume * 365
annual_sales_value = annual_sales_volume * avg_product_value

# Current state
current_forecast_error_units = annual_sales_volume * (current_forecast_error / 100)
current_overstock_cost = (current_forecast_error_units / 2) * avg_product_value * (inventory_holding_cost_pct / 100)
current_stockout_cost = (current_forecast_error_units / 2) * avg_product_value * stockout_cost_multiplier
current_total_cost = current_overstock_cost + current_stockout_cost

# New state
new_forecast_error_units = annual_sales_volume * (new_forecast_error / 100)
new_overstock_cost = (new_forecast_error_units / 2) * avg_product_value * (inventory_holding_cost_pct / 100) * (1 - waste_reduction_pct / 100)
new_stockout_cost = (new_forecast_error_units / 2) * avg_product_value * stockout_cost_multiplier * (1 - stockout_reduction_pct / 100)
new_total_cost = new_overstock_cost + new_stockout_cost

# Savings
total_savings = current_total_cost - new_total_cost
roi_pct = (total_savings / current_total_cost) * 100 if current_total_cost > 0 else 0

# Display results
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Annual Sales",
        f"${annual_sales_value:,.0f}",
        help="Total annual sales value"
    )

with col2:
    st.metric(
        "Current Cost",
        f"${current_total_cost:,.0f}",
        help="Current forecasting error costs"
    )

with col3:
    st.metric(
        "New Cost",
        f"${new_total_cost:,.0f}",
        delta=f"-${total_savings:,.0f}",
        delta_color="inverse",
        help="Forecasting costs with new models"
    )

with col4:
    st.metric(
        "Annual Savings",
        f"${total_savings:,.0f}",
        delta=f"{roi_pct:.1f}% ROI",
        help="Total annual cost savings"
    )

# Breakdown visualization
st.markdown("### Cost Breakdown")

fig = go.Figure()

fig.add_trace(go.Bar(
    name='Current State',
    x=['Overstock Cost', 'Stockout Cost'],
    y=[current_overstock_cost, current_stockout_cost],
    marker_color='#FF6692'
))

fig.add_trace(go.Bar(
    name='With New Models',
    x=['Overstock Cost', 'Stockout Cost'],
    y=[new_overstock_cost, new_stockout_cost],
    marker_color='#00CC96'
))

fig.update_layout(
    title='Cost Comparison: Current vs New Forecasting',
    xaxis_title='Cost Type',
    yaxis_title='Annual Cost ($)',
    barmode='group',
    template='plotly_dark'
)

st.plotly_chart(fig, use_container_width=True)

# Key benefits
st.markdown("---")
st.markdown("## 🎯 Key Benefits")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Inventory Optimization
    
    - **Reduced Waste**: {:.1f}% reduction in spoilage and obsolescence
    - **Lower Holding Costs**: Optimize stock levels across all stores
    - **Better Cash Flow**: Free up capital tied in excess inventory
    - **Improved Turnover**: Faster inventory rotation
    
    ### Operational Efficiency
    
    - **Staffing Optimization**: Match workforce to predicted demand
    - **Reduced Overtime**: Better planning reduces emergency staffing
    - **Warehouse Efficiency**: Optimize storage and picking operations
    - **Transport Optimization**: Consolidate deliveries based on forecasts
    """.format(waste_reduction_pct))

with col2:
    st.markdown("""
    ### Customer Satisfaction
    
    - **Fewer Stockouts**: {:.1f}% reduction in out-of-stock incidents
    - **Better Product Availability**: Right products at right time
    - **Improved Experience**: Customers find what they need
    - **Increased Loyalty**: Consistent availability builds trust
    
    ### Strategic Planning
    
    - **Promotional Planning**: Optimize discount timing and depth
    - **New Product Launch**: Better demand estimation
    - **Seasonal Planning**: Prepare for demand fluctuations
    - **Supplier Negotiations**: Accurate volume forecasts improve terms
    """.format(stockout_reduction_pct))

# Industry benchmarks
st.markdown("---")
st.markdown("## 📈 Industry Benchmarks")

st.info("""
**Retail Industry Averages:**
- Forecast Accuracy (MAPE): 20-30%
- Inventory Holding Cost: 15-25% per year
- Stockout Rate: 8-10% of SKUs
- Waste/Shrinkage: 2-5% of inventory value

**Best-in-Class Performance:**
- Forecast Accuracy (MAPE): < 15%
- Stockout Rate: < 3%
- Inventory Turnover: 8-12x per year
- Waste Rate: < 1%
""")

# Implementation timeline
st.markdown("---")
st.markdown("## 🗓️ Implementation Timeline")

timeline_data = {
    "Phase": ["Phase 1", "Phase 2", "Phase 3", "Phase 4"],
    "Activity": [
        "Data Integration & Setup",
        "Model Training & Validation",
        "Pilot Testing (2-3 stores)",
        "Full Rollout"
    ],
    "Duration": ["2-3 weeks", "3-4 weeks", "4-6 weeks", "2-3 weeks"],
    "Expected Impact": ["0%", "Testing", "20-30% of savings", "100% of savings"]
}

st.table(pd.DataFrame(timeline_data))

st.success(f"""
**Projected Payback Period**: 2-4 months

Based on annual savings of **${total_savings:,.0f}**, the investment in advanced forecasting
typically pays for itself within the first quarter of implementation.
""")

# Footer
add_footer()
