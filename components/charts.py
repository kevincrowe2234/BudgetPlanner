import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def render_balance_chart(monthly_df, title="Monthly Budget Projection"):
    """Render a monthly balance chart"""
    fig = go.Figure()
    
    # Convert index to proper format if using date strings
    x_values = monthly_df.index
    
    # Add monthly balance line
    fig.add_trace(go.Scatter(
        x=x_values, 
        y=monthly_df['Balance'],
        mode='lines+markers',
        name='Balance',
        line=dict(color='#1E88E5', width=3)
    ))
    
    # Add monthly income and expenses as bar charts
    fig.add_trace(go.Bar(
        x=x_values,
        y=monthly_df['Income'],
        name='Income',
        marker_color='rgba(0, 128, 0, 0.7)'
    ))
    
    fig.add_trace(go.Bar(
        x=x_values,
        y=monthly_df['Expense'] * -1,  # Negative for visualization
        name='Expenses',
        marker_color='rgba(255, 0, 0, 0.7)'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Month',
        yaxis_title='Amount ($)',
        barmode='relative',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_summary_metrics(income_total, expense_total, net_cashflow):
    """Render summary metrics in columns"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Income", f"${income_total:.2f}")
    with col2:
        st.metric("Total Expenses", f"${expense_total:.2f}")
    with col3:
        st.metric("Net Cashflow", f"${net_cashflow:.2f}")

def render_category_chart(df, category_column, value_column, title):
    """Render a pie chart for income/expense categories"""
    if df.empty:
        st.info(f"No data available for {title}")
        return
        
    # Aggregate data by category
    category_data = df.groupby(category_column)[value_column].sum().reset_index()
    
    # Create pie chart
    fig = px.pie(
        category_data, 
        values=value_column, 
        names=category_column,
        title=title,
        hole=0.4
    )
    
    st.plotly_chart(fig, use_container_width=True)