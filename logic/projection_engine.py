import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

def plot_balance_over_time(timeline_df, monthly_df):
    """Create plots for balance projections (daily and monthly)"""
    # Daily balance chart
    daily_chart = go.Figure()
    daily_chart.add_trace(go.Scatter(
        x=timeline_df.index, 
        y=timeline_df['Balance'],
        mode='lines',
        name='Balance',
        line=dict(color='#1E88E5', width=2)
    ))
    
    # Add income and expenses as bar charts
    daily_chart.add_trace(go.Bar(
        x=timeline_df.index,
        y=timeline_df['Income'],
        name='Income',
        marker_color='rgba(0, 128, 0, 0.5)',
        opacity=0.7
    ))
    
    daily_chart.add_trace(go.Bar(
        x=timeline_df.index,
        y=timeline_df['Expense'] * -1,  # Negative for visualization
        name='Expenses',
        marker_color='rgba(255, 0, 0, 0.5)',
        opacity=0.7
    ))
    
    # Update layout
    daily_chart.update_layout(
        title='Daily Balance Projection',
        xaxis_title='Date',
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
    
    # Monthly balance chart
    monthly_chart = go.Figure()
    
    # Convert index to proper format if using date strings
    x_values = monthly_df.index
    
    # Add monthly balance line
    monthly_chart.add_trace(go.Scatter(
        x=x_values, 
        y=monthly_df['Balance'],
        mode='lines+markers',
        name='Balance',
        line=dict(color='#1E88E5', width=3)
    ))
    
    # Add monthly income and expenses as bar charts
    monthly_chart.add_trace(go.Bar(
        x=x_values,
        y=monthly_df['Income'],
        name='Income',
        marker_color='rgba(0, 128, 0, 0.7)'
    ))
    
    monthly_chart.add_trace(go.Bar(
        x=x_values,
        y=monthly_df['Expense'] * -1,  # Negative for visualization
        name='Expenses',
        marker_color='rgba(255, 0, 0, 0.7)'
    ))
    
    # Update layout
    monthly_chart.update_layout(
        title='Monthly Budget Projection',
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
    
    return daily_chart, monthly_chart

def calculate_financial_metrics(monthly_df):
    """Calculate key financial metrics from projection data"""
    metrics = {
        'average_monthly_income': monthly_df['Income'].mean(),
        'average_monthly_expenses': monthly_df['Expense'].mean(),
        'average_monthly_savings': monthly_df['Net'].mean(),
        'savings_rate': (monthly_df['Net'].sum() / monthly_df['Income'].sum()) * 100 if monthly_df['Income'].sum() > 0 else 0,
        'months_positive': (monthly_df['Net'] > 0).sum(),
        'months_negative': (monthly_df['Net'] < 0).sum()
    }
    
    return metrics