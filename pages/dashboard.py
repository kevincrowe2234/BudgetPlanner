import streamlit as st
import pandas as pd
from logic.budget_processor import prepare_dataframes_for_projection, generate_timeline
from logic.projection_engine import plot_balance_over_time, calculate_financial_metrics
from components.transaction_table import render_transaction_table

def render_dashboard():
    """Render the dashboard page with summary stats and projections"""
    st.header("Budget Dashboard")
    
    # Basic check for required session state
    if not hasattr(st.session_state, 'income_data') or not hasattr(st.session_state, 'expense_data'):
        st.warning("Budget data not initialized. Please configure your budget in the sidebar.")
        return
    
    # Display imported transactions if available
    if 'transactions' in st.session_state and st.session_state['transactions']:
        with st.expander("View Imported Transactions"):
            st.write("Processed Transactions:")
            render_transaction_table(st.session_state['transactions'])
    
    # Summary statistics
    st.subheader("Summary Statistics")
    total_income = st.session_state.income_data['Amount'].sum() if not st.session_state.income_data.empty else 0
    total_expense = st.session_state.expense_data['Amount'].sum() if not st.session_state.expense_data.empty else 0
    net_cashflow = total_income - total_expense
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Income", f"${total_income:.2f}")
    with col2:
        st.metric("Total Expenses", f"${total_expense:.2f}")
    with col3:
        st.metric("Net Cashflow", f"${net_cashflow:.2f}")
    
    # Generate and display projections
    st.subheader("Budget Projections")
    
    if st.session_state.income_data.empty and st.session_state.expense_data.empty:
        st.info("No income or expenses added yet. Add some data to see projections.")
        return
        
    try:
        # Prepare data for projection
        income_clean, expense_clean = prepare_dataframes_for_projection(
            st.session_state.income_data, 
            st.session_state.expense_data
        )
        
        # Generate timeline data
        timeline_df, monthly_df = generate_timeline(
            income_clean, 
            expense_clean, 
            st.session_state.get('bank_balance', 0.0),
            months=st.session_state.get('projection_months', 12)
        )
        
        # Create and display charts
        daily_chart, monthly_chart = plot_balance_over_time(timeline_df, monthly_df)
        st.plotly_chart(monthly_chart, use_container_width=True)
        
        with st.expander("Show Detailed Daily Projection"):
            st.plotly_chart(daily_chart, use_container_width=True)
        
        with st.expander("Show Detailed Monthly Projections"):
            st.dataframe(monthly_df)
            
    except Exception as e:
        st.error(f"Error generating projections: {str(e)}")
        st.info("There may be an issue with your data. Try adding some sample income and expenses.")