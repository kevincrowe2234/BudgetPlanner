import streamlit as st
import pandas as pd
from datetime import datetime
from utils.file_operations import save_session_data

def render_income_form():
    """Render form for adding new income entry"""
    with st.form("income_form"):
        st.subheader("Add New Income")
        income_desc = st.text_input("Description", key="income_desc")
        income_amount = st.number_input("Amount ($)", min_value=0.01, key="income_amount")
        income_date = st.date_input("Date", key="income_date")
        income_recurring = st.checkbox("Recurring?", key="income_recurring")
        income_freq = st.selectbox("Frequency", 
                                 ["None", "Daily", "Weekly", "Biweekly", "Monthly", "Quarterly", "Annually"], 
                                 key="income_freq")
        
        submit_income = st.form_submit_button("Add Income")
    
    if submit_income:
        if income_desc and income_amount > 0:
            # Add to income dataframe
            new_income = pd.DataFrame({
                'Description': [income_desc],
                'Amount': [income_amount],
                'Date': [income_date],
                'Recurring': [income_recurring],
                'Frequency': [income_freq],
                'Type': ['Income']
            })
            
            if st.session_state.income_data.empty:
                st.session_state.income_data = new_income
            else:
                st.session_state.income_data = pd.concat([st.session_state.income_data, new_income], ignore_index=True)
            
            st.session_state.last_update = datetime.now()
            save_session_data()
            st.success("Income added successfully!")
            st.rerun()
        else:
            st.error("Please enter a description and amount greater than 0.")

def render_expense_form():
    """Render form for adding new expense entry"""
    with st.form("expense_form"):
        st.subheader("Add New Expense")
        expense_desc = st.text_input("Description", key="expense_desc")
        expense_amount = st.number_input("Amount ($)", min_value=0.01, key="expense_amount")
        expense_date = st.date_input("Date", key="expense_date")
        expense_recurring = st.checkbox("Recurring?", key="expense_recurring")
        expense_freq = st.selectbox("Frequency", 
                                  ["None", "Daily", "Weekly", "Biweekly", "Monthly", "Quarterly", "Annually"], 
                                  key="expense_freq")
        
        submit_expense = st.form_submit_button("Add Expense")
    
    if submit_expense:
        if expense_desc and expense_amount > 0:
            # Add to expense dataframe
            new_expense = pd.DataFrame({
                'Description': [expense_desc],
                'Amount': [expense_amount],
                'Date': [expense_date],
                'Recurring': [expense_recurring],
                'Frequency': [expense_freq],
                'Type': ['Expense']
            })
            
            if st.session_state.expense_data.empty:
                st.session_state.expense_data = new_expense
            else:
                st.session_state.expense_data = pd.concat([st.session_state.expense_data, new_expense], ignore_index=True)
            
            st.session_state.last_update = datetime.now()
            save_session_data()
            st.success("Expense added successfully!")
            st.rerun()
        else:
            st.error("Please enter a description and amount greater than 0.")