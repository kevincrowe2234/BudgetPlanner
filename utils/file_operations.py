import json
import os
import pandas as pd
import streamlit as st
from datetime import datetime
from utils.config import BUDGET_DATA_FILE, TRANSACTIONS_FILE, EMPTY_BUDGET_DF_COLS, DEFAULT_PROJECTION_MONTHS, DEFAULT_BANK_BALANCE

def save_transactions(transactions):
    """Save transactions to a JSON file"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(TRANSACTIONS_FILE), exist_ok=True)
        
        with open(TRANSACTIONS_FILE, "w") as f:
            # Convert datetime objects to strings before saving
            serializable_transactions = []
            for t in transactions:
                t_copy = t.copy()
                if isinstance(t_copy["Date"], datetime):
                    t_copy["Date"] = t_copy["Date"].strftime("%Y-%m-%d")
                serializable_transactions.append(t_copy)
            
            json.dump(serializable_transactions, f, indent=2)
        return True
    except Exception as e:
        st.sidebar.error(f"Failed to save transactions: {e}")
        return False

def load_transactions():
    """Load transactions from a JSON file"""
    if not os.path.exists(TRANSACTIONS_FILE):
        return []
        
    try:
        # Check if file exists and is not empty
        if os.path.getsize(TRANSACTIONS_FILE) > 0:
            with open(TRANSACTIONS_FILE, "r") as f:
                transactions = json.load(f)
                # Convert date strings back to datetime objects
                for t in transactions:
                    if isinstance(t["Date"], str):
                        try:
                            t["Date"] = datetime.strptime(t["Date"], "%Y-%m-%d")
                        except ValueError:
                            from dateutil.parser import parse
                            t["Date"] = parse(t["Date"])
                
                return transactions
        else:
            return []  # Return empty list for empty file
    except Exception as e:
        print(f"Failed to load transactions: {e}")  # Log error but don't show in UI
        return []  # Return empty list on error

def save_session_data():
    """Save budget data to a JSON file"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(BUDGET_DATA_FILE), exist_ok=True)
        
        data = {
            'income_data': st.session_state.income_data.to_dict('records') if not st.session_state.income_data.empty else [],
            'expense_data': st.session_state.expense_data.to_dict('records') if not st.session_state.expense_data.empty else [],
            'bank_balance': st.session_state.bank_balance,
            'projection_months': st.session_state.projection_months,
            'last_update': datetime.now().isoformat()
        }
        
        with open(BUDGET_DATA_FILE, 'w') as f:
            json.dump(data, f, default=str)
        
        return True
    except Exception as e:
        print(f"Auto-save failed: {e}")
        st.error(f"Failed to save data: {e}")
        return False

def load_session_data():
    """Load budget data from a JSON file"""
    try:
        if os.path.exists(BUDGET_DATA_FILE) and os.path.getsize(BUDGET_DATA_FILE) > 0:
            with open(BUDGET_DATA_FILE, 'r') as f:
                data = json.load(f)
            
            # Initialize session state variables
            initialize_session_state()
            
            # Restore income data
            if data.get('income_data'):
                income_df = pd.DataFrame(data['income_data'])
                if not income_df.empty:
                    income_df['Date'] = pd.to_datetime(income_df['Date'])
                    income_df['Recurring'] = income_df['Recurring'].astype(bool)
                    income_df['Amount'] = income_df['Amount'].astype(float)
                    st.session_state.income_data = income_df
            
            # Restore expense data
            if data.get('expense_data'):
                expense_df = pd.DataFrame(data['expense_data'])
                if not expense_df.empty:
                    expense_df['Date'] = pd.to_datetime(expense_df['Date'])
                    expense_df['Recurring'] = expense_df['Recurring'].astype(bool)
                    expense_df['Amount'] = expense_df['Amount'].astype(float)
                    st.session_state.expense_data = expense_df
            
            # Restore other settings
            if 'bank_balance' in data:
                st.session_state.bank_balance = float(data['bank_balance'])
            if 'projection_months' in data:
                st.session_state.projection_months = int(data['projection_months'])
            if 'last_update' in data:
                st.session_state.last_update = datetime.fromisoformat(data['last_update'])
            
            # Load transactions if available
            st.session_state.transactions = load_transactions()
            
            return True
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(BUDGET_DATA_FILE), exist_ok=True)
            # Initialize empty state
            initialize_session_state()
            return False
    except Exception as e:
        print(f"Auto-load failed: {e}")
        initialize_session_state()
        return False

def initialize_session_state():
    """Initialize session state with default values"""
    if 'income_data' not in st.session_state:
        st.session_state.income_data = pd.DataFrame(columns=EMPTY_BUDGET_DF_COLS)

    if 'expense_data' not in st.session_state:
        st.session_state.expense_data = pd.DataFrame(columns=EMPTY_BUDGET_DF_COLS)

    if 'bank_balance' not in st.session_state:
        st.session_state.bank_balance = DEFAULT_BANK_BALANCE

    if 'projection_months' not in st.session_state:
        st.session_state.projection_months = DEFAULT_PROJECTION_MONTHS

    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()

    if 'transactions' not in st.session_state:
        st.session_state.transactions = []