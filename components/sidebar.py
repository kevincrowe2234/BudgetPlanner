import streamlit as st
import pandas as pd
import json
import io
import csv
from datetime import datetime
from utils.file_operations import save_session_data, save_transactions
from logic.transaction_processor import parse_transaction_csv, transactions_to_budget_data
from utils.config import TRANSACTIONS_FILE

def render_configuration():
    """Render configuration settings in the sidebar"""
    st.subheader("Budget Settings")
    
    # Bank balance configuration
    current_balance = st.number_input(
        "Current Bank Balance ($)",
        value=st.session_state.get('bank_balance', 0.0),  # Use get method with default
        step=100.0
    )
    
    if 'bank_balance' not in st.session_state or current_balance != st.session_state.bank_balance:
        st.session_state.bank_balance = current_balance
        st.session_state.last_update = datetime.now()
        save_session_data()
    
    # Projection months configuration
    projection_months = st.slider(
        "Projection Months",
        min_value=1,
        max_value=60,
        value=st.session_state.get('projection_months', 12)  # Use get method with default
    )
    
    if 'projection_months' not in st.session_state or projection_months != st.session_state.projection_months:
        st.session_state.projection_months = projection_months
        st.session_state.last_update = datetime.now()
        save_session_data()

def render_data_management():
    """Render data management options in the sidebar"""
    st.subheader("Data Management")
    
    if st.button("Download Budget Data"):
        if hasattr(st.session_state, 'income_data') and hasattr(st.session_state, 'expense_data'):
            data = {
                'income_data': st.session_state.income_data.to_dict('records') if not st.session_state.income_data.empty else [],
                'expense_data': st.session_state.expense_data.to_dict('records') if not st.session_state.expense_data.empty else [],
                'bank_balance': st.session_state.get('bank_balance', 0.0),
                'projection_months': st.session_state.get('projection_months', 12),
            }
            
            # Convert to JSON string
            json_str = json.dumps(data, default=str)
            
            # Create download button
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="budget_data.json",
                mime="application/json"
            )
        else:
            st.warning("No budget data to download.")
    
    if st.button("Reset All Data"):
        # Initialize empty dataframes if not already present
        if not hasattr(st.session_state, 'income_data') or not hasattr(st.session_state, 'expense_data'):
            from utils.file_operations import initialize_session_state
            initialize_session_state()
            
        if st.session_state.income_data.empty and st.session_state.expense_data.empty:
            st.info("No data to reset.")
        else:
            # Reset to empty dataframes
            st.session_state.income_data = pd.DataFrame(columns=['Description', 'Amount', 'Date', 'Recurring', 'Frequency', 'Type'])
            st.session_state.expense_data = pd.DataFrame(columns=['Description', 'Amount', 'Date', 'Recurring', 'Frequency', 'Type'])
            st.session_state.last_update = datetime.now()
            save_session_data()
            st.success("All budget data has been reset.")
            st.rerun()

def render_transaction_import():
    """Render transaction import options in the sidebar"""
    with st.expander("Import Past Transactions"):
        st.write("Upload a transaction history")
        
        # CSV file uploader
        uploaded_file = st.file_uploader("Drag and drop file here or", type="csv", key="FileUpload", 
                                         help="Limit 200MB per file - CSV")
        
        # Add Clear Previous Data button
        if st.button("Clear Previous Data"):
            # Clear session state
            if 'transactions' in st.session_state:
                st.session_state.transactions = []
            
            # Create empty transactions file
            try:
                with open(TRANSACTIONS_FILE, "w") as f:
                    json.dump([], f)
                st.success("All transaction data cleared successfully.")
            except Exception as e:
                st.error(f"Failed to clear transaction data: {e}")
            
            # Force a rerun to update the UI
            st.rerun()

        # Process uploaded file
        if uploaded_file is not None:
            try:
                # Read the file
                uploaded_file.seek(0)  # Reset file pointer
                file_content = uploaded_file.read().decode('utf-8')
                
                # Parse CSV into transactions
                new_transactions = parse_transaction_csv(file_content)
                
                if new_transactions:
                    st.session_state.transactions = new_transactions
                    save_transactions(new_transactions)
                    st.success(f"Successfully imported {len(new_transactions)} transactions.")
                else:
                    st.warning("No valid transactions found in the CSV file.")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.write("Please ensure the CSV contains the required columns and is properly formatted.")

        # Convert transactions to budget data
        if 'transactions' in st.session_state and st.session_state['transactions']:
            st.write(f"**{len(st.session_state['transactions'])} transactions available for import to budget**")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Add to Budget Data"):
                    # Convert transactions to budget format
                    income_df, expense_df = transactions_to_budget_data(st.session_state['transactions'])
                    
                    # Add to existing data
                    if not income_df.empty:
                        if not hasattr(st.session_state, 'income_data') or st.session_state.income_data.empty:
                            st.session_state.income_data = income_df
                        else:
                            st.session_state.income_data = pd.concat(
                                [st.session_state.income_data, income_df], 
                                ignore_index=True
                            )
                    
                    if not expense_df.empty:
                        if not hasattr(st.session_state, 'expense_data') or st.session_state.expense_data.empty:
                            st.session_state.expense_data = expense_df
                        else:
                            st.session_state.expense_data = pd.concat(
                                [st.session_state.expense_data, expense_df], 
                                ignore_index=True
                            )
                    
                    # Save the updated budget data
                    st.session_state.last_update = datetime.now()
                    save_session_data()
                    
                    st.success(f"Added {len(income_df) if not income_df.empty else 0} income entries and {len(expense_df) if not expense_df.empty else 0} expense entries to your budget!")
                    st.rerun()
            
            with col2:
                if st.button("Replace Budget Data"):
                    # Convert transactions to budget format
                    income_df, expense_df = transactions_to_budget_data(st.session_state['transactions'])
                    
                    # Replace existing data
                    st.session_state.income_data = income_df if not income_df.empty else pd.DataFrame({
                        'Description': [],
                        'Amount': [],
                        'Date': [],
                        'Recurring': [],
                        'Frequency': [],
                        'Type': []
                    })
                    
                    st.session_state.expense_data = expense_df if not expense_df.empty else pd.DataFrame({
                        'Description': [],
                        'Amount': [],
                        'Date': [],
                        'Recurring': [],
                        'Frequency': [],
                        'Type': []
                    })
                    
                    # Save the updated budget data
                    st.session_state.last_update = datetime.now()
                    save_session_data()
                    
                    st.success(f"Replaced budget data with {len(income_df) if not income_df.empty else 0} income entries and {len(expense_df) if not expense_df.empty else 0} expense entries!")
                    st.rerun()
        else:
            st.info("Import transactions first before adding them to your budget.")