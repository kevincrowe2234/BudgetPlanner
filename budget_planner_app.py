import streamlit as st
import csv
from datetime import datetime
from dateutil.parser import parse
import io
import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
import base64
from io import StringIO
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
from dateutil.relativedelta import relativedelta
import html

st.set_page_config(page_title="Budget Planner", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 1rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1E88E5;
    }
    h2 {
        color: #0D47A1;
    }
    h3 {
        color: #283593;
    }

    .fixed-table {
        width: 100%;
        table-layout: fixed;
        border-collapse: collapse;
        margin-bottom: 10px;
    }
    .fixed-table th, .fixed-table td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
        word-wrap: break-word;
    }
    .fixed-table th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    .fixed-table .col-date { width: 20%; }
    .fixed-table .col-desc { width: 50%; }
    .fixed-table .col-amount { width: 30%; }
    .table-container {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #ccc;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# File paths
DATA_FILE = "budget_data_autosave.json"
TRANSACTIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transactions.json")

# Functions for saving and loading transactions
def save_transactions(transactions):
    """Save transactions to a file"""
    try:
        with open(TRANSACTIONS_FILE, "w") as f:
            # Convert datetime objects to strings before saving
            serializable_transactions = []
            for t in transactions:
                t_copy = t.copy()
                if isinstance(t_copy["Date"], datetime):
                    t_copy["Date"] = t_copy["Date"].strftime("%Y-%m-%d")
                serializable_transactions.append(t_copy)
            
            json.dump(serializable_transactions, f, indent=2)
        st.sidebar.success(f"Saved {len(transactions)} transactions")
    except Exception as e:
        st.sidebar.error(f"Failed to save transactions: {e}")

def load_transactions():
    """Load transactions from a file"""
    if not os.path.exists(TRANSACTIONS_FILE):
        return []
        
    try:
        with open(TRANSACTIONS_FILE, "r") as f:
            transactions = json.load(f)
            # Convert date strings back to datetime objects
            for t in transactions:
                if isinstance(t["Date"], str):
                    try:
                        t["Date"] = datetime.strptime(t["Date"], "%Y-%m-%d")
                    except ValueError:
                        # If standard format fails, try parsing with dateutil
                        t["Date"] = parse(t["Date"])
            
            return transactions
    except Exception as e:
        st.sidebar.error(f"Failed to load transactions: {e}")
        return []

# ADD THIS SECTION HERE (after CSS, before session state)
def save_session_data():
    """Save session data to local file"""
    try:
        data = {
            'income_data': st.session_state.income_data.to_dict('records') if not st.session_state.income_data.empty else [],
            'expense_data': st.session_state.expense_data.to_dict('records') if not st.session_state.expense_data.empty else [],
            'bank_balance': st.session_state.bank_balance,
            'projection_months': st.session_state.projection_months,
            'last_update': st.session_state.last_update.isoformat() if hasattr(st.session_state, 'last_update') else datetime.now().isoformat()
        }
        
        with open(DATA_FILE, 'w') as f:
            json.dump(data, f, default=str)
        
        print(f"DEBUG: Auto-saved session data to {DATA_FILE}")
        return True
    except Exception as e:
        print(f"DEBUG: Auto-save failed: {e}")
        return False

def load_session_data():
    """Load session data from local file"""
    try:
        if Path(DATA_FILE).exists():
            with open(DATA_FILE, 'r') as f:
                data = json.load(f)
            
            # Restore income data
            if data.get('income_data'):
                income_df = pd.DataFrame(data['income_data'])
                income_df['Date'] = pd.to_datetime(income_df['Date'])
                income_df['Recurring'] = income_df['Recurring'].astype(bool)
                income_df['Amount'] = income_df['Amount'].astype(float)
                st.session_state.income_data = income_df
            
            # Restore expense data
            if data.get('expense_data'):
                expense_df = pd.DataFrame(data['expense_data'])
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
            
            print(f"DEBUG: Auto-loaded session data from {DATA_FILE}")
            print(f"DEBUG: Loaded {len(st.session_state.income_data)} income entries")
            print(f"DEBUG: Loaded {len(st.session_state.expense_data)} expense entries")
            return True
    except Exception as e:
        print(f"DEBUG: Auto-load failed: {e}")
        return False
    return False

# Initialize session state variables if they don't exist
if 'income_data' not in st.session_state:
    st.session_state.income_data = pd.DataFrame({
        'Description': [],
        'Amount': [],
        'Date': [],
        'Recurring': [],
        'Frequency': [],
        'Type': []
    })

if 'expense_data' not in st.session_state:
    st.session_state.expense_data = pd.DataFrame({
        'Description': [],
        'Amount': [],
        'Date': [],
        'Recurring': [],
        'Frequency': [],
        'Type': []
    })

if 'bank_balance' not in st.session_state:
    st.session_state.bank_balance = 0.0

if 'projection_months' not in st.session_state:
    st.session_state.projection_months = 18

# Correct initialization of session state variables
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# Auto-load data on startup (only once per session)
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = True
    load_session_data()

# Functions for data manipulation
def get_dataframe_download_link(df, filename, text):
    """Generate a link to download the dataframe as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    return href


def save_data(income_df, expense_df, balance, filename):
    """Prepare all budget data for saving to a CSV file"""
    # Create copies to avoid modifying originals
    income_copy = income_df.copy() if not income_df.empty else pd.DataFrame({
        'Description': [], 'Amount': [], 'Date': [], 'Recurring': [], 'Frequency': [], 'Type': []
    })
    expense_copy = expense_df.copy() if not expense_df.empty else pd.DataFrame({
        'Description': [], 'Amount': [], 'Date': [], 'Recurring': [], 'Frequency': [], 'Type': []
    })
    
    # Add Type column safely
    if not income_copy.empty:
        income_copy['Type'] = 'Income'
    if not expense_copy.empty:
        expense_copy['Type'] = 'Expense'

    # Combine dataframes
    combined_df = pd.concat([income_copy, expense_copy], ignore_index=True)

    # Add balance as metadata
    metadata_df = pd.DataFrame({
        'Description': ['INITIAL_BALANCE'],
        'Amount': [balance],
        'Date': [datetime.now().strftime('%Y-%m-%d')],
        'Recurring': [False],
        'Frequency': ['None'],
        'Type': ['Metadata']
    })

    final_df = pd.concat([metadata_df, combined_df], ignore_index=True)
    return final_df


def load_data(uploaded_file):
    """Load budget data from a CSV file using robust parsing"""
    if uploaded_file is not None:
        try:
            # Use robust CSV parsing for consistency
            df, header_row, skipped_lines = read_transaction_csv_robust(
                uploaded_file, 
                expected_columns=['Description', 'Amount', 'Date', 'Recurring', 'Frequency', 'Type']
            )
            
            if df.empty:
                st.error("No data found in the uploaded file")
                return pd.DataFrame(), pd.DataFrame(), 0.0
            
            # Show parsing info
            if skipped_lines > 0:
                st.warning(f"⚠️ Skipped {skipped_lines} malformed lines while loading")

            # Extract metadata
            metadata = df[df['Type'] == 'Metadata']
            balance_row = metadata[metadata['Description'] == 'INITIAL_BALANCE']
            if not balance_row.empty:
                balance = float(balance_row.iloc[0]['Amount'])
            else:
                balance = 0.0

            # Extract income and expense data
            income_df = df[df['Type'] == 'Income'].copy()
            expense_df = df[df['Type'] == 'Expense'].copy()

            # Convert date strings to datetime objects
            if not income_df.empty:
                income_df['Date'] = pd.to_datetime(income_df['Date'])
                income_df['Recurring'] = income_df['Recurring'].astype(bool)
                income_df['Amount'] = pd.to_numeric(income_df['Amount'], errors='coerce')

            if not expense_df.empty:
                expense_df['Date'] = pd.to_datetime(expense_df['Date'])
                expense_df['Recurring'] = expense_df['Recurring'].astype(bool)
                expense_df['Amount'] = pd.to_numeric(expense_df['Amount'], errors='coerce')

            st.success("✅ Budget data loaded successfully!")
            return income_df, expense_df, balance
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            
            # Fallback to original pandas method for files saved by older versions
            try:
                st.warning("Trying fallback method for older file format...")
                df = pd.read_csv(uploaded_file)

                # Extract metadata
                metadata = df[df['Type'] == 'Metadata']
                balance_row = metadata[metadata['Description'] == 'INITIAL_BALANCE']
                if not balance_row.empty:
                    balance = float(balance_row.iloc[0]['Amount'])
                else:
                    balance = 0.0

                # Extract income and expense data
                income_df = df[df['Type'] == 'Income'].copy()
                expense_df = df[df['Type'] == 'Expense'].copy()

                # Convert date strings to datetime objects
                if not income_df.empty:
                    income_df['Date'] = pd.to_datetime(income_df['Date'])
                    income_df['Recurring'] = income_df['Recurring'].astype(bool)

                if not expense_df.empty:
                    expense_df['Date'] = pd.to_datetime(expense_df['Date'])
                    expense_df['Recurring'] = expense_df['Recurring'].astype(bool)

                st.success("✅ Budget data loaded using fallback method!")
                return income_df, expense_df, balance
                
            except Exception as e2:
                st.error(f"Both loading methods failed: {e2}")
                return pd.DataFrame(), pd.DataFrame(), 0.0
    
    return pd.DataFrame(), pd.DataFrame(), 0.0


def read_transaction_csv_robust(uploaded_file, expected_columns=None):
    """
    Robust CSV reader using Python's csv module instead of pandas.
    Specifically designed for transaction CSV files with inconsistent column counts.
    """
    if expected_columns is None:
        expected_columns = ['Date', 'Description', 'Debit', 'Credit', 'Balance', 'Category', 'SubCategory']
    
    # Add extra columns to handle overflow
    all_columns = expected_columns + [f'Extra{i}' for i in range(1, 4)]  # Extra1-Extra3
    
    try:
        # Read the uploaded file content
        if hasattr(uploaded_file, 'read'):
            content = uploaded_file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            uploaded_file.seek(0)  # Reset file pointer
        else:
            content = str(uploaded_file)
        
        # Parse CSV using Python's csv module
        csv_reader = csv.reader(StringIO(content), quoting=csv.QUOTE_MINIMAL)
        rows = []
        header_row = None
        skipped_lines = 0
        line_num = 0
        
        for row in csv_reader:
            try:
                line_num += 1
                
                if line_num == 1:
                    # Process header row
                    header_row = row
                    st.info(f"📋 CSV Header: {header_row}")
                    continue
                
                # Skip empty rows
                if not row or all(cell.strip() == '' for cell in row if cell):
                    continue
                
                # Handle rows with different column counts
                processed_row = []
                
                # If row has fewer columns than expected, pad with empty strings
                while len(row) < len(all_columns):
                    row.append('')
                
                # If row has more columns than expected, combine overflow into SubCategory
                if len(row) > len(expected_columns):
                    # Take the first expected columns as-is
                    processed_row = row[:len(expected_columns)]
                    
                    # Combine remaining columns into SubCategory if there's overflow
                    if len(row) > len(expected_columns):
                        overflow_data = ', '.join(row[len(expected_columns):])
                        if len(processed_row) > 6:  # SubCategory is index 6
                            processed_row[6] = str(processed_row[6]) + ', ' + overflow_data
                        else:
                            processed_row.append(overflow_data)
                else:
                    processed_row = row[:len(expected_columns)]
                
                # Ensure we have exactly the right number of columns
                while len(processed_row) < len(expected_columns):
                    processed_row.append('')
                
                processed_row = processed_row[:len(expected_columns)]
                rows.append(processed_row)
                
            except Exception as e:
                print(f"DEBUG: Skipping malformed line {line_num}: {e}")
                skipped_lines += 1
                continue
        
        # Create DataFrame from processed rows
        if rows:
            df = pd.DataFrame(rows, columns=expected_columns)
            # Ensure all expected columns exist
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = None  # Add missing columns with default values
            
            
            
            # Clean up empty columns
            for col in expected_columns:
                if col in df.columns:
                    if df[col].fillna('').astype(str).str.strip().eq('').all():
                        df = df.drop(columns=[col])
            
            print(f"DEBUG: Successfully parsed CSV with {len(df)} rows")
            if skipped_lines > 0:
                print(f"DEBUG: Skipped {skipped_lines} malformed lines")
                
            return df, header_row, skipped_lines
        else:
            return pd.DataFrame(), [], 0
            
    except Exception as e:
        print(f"DEBUG: CSV parsing error: {e}")
        return pd.DataFrame(), [], 0

def clean_transaction_data(df):
    """
    Clean the transaction DataFrame after reading.
    """
    if df.empty:
        return df
    
    try:
        # Convert Date column to datetime (try multiple formats)
        if 'Date' in df.columns:
            # Try different date formats
            date_formats = ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']
            for fmt in date_formats:
                try:
                    df['Date'] = pd.to_datetime(df['Date'], format=fmt, errors='coerce')
                    valid_dates = df['Date'].notna().sum()
                    if valid_dates > 0:
                        st.info(f"✅ Parsed {valid_dates} dates using format {fmt}")
                        break
                except:
                    continue
            
            # If none of the formats worked, try pandas' flexible parser
            if df['Date'].isna().all():
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Convert numeric columns, handling various formats
        numeric_columns = ['Debit', 'Credit', 'Balance', 'Amount']
        for col in numeric_columns:
            if col in df.columns:
                # Clean currency symbols and commas
                df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '').str.replace('(', '').str.replace(')', '').str.replace('£', '').str.replace('€', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any completely empty rows
        df = df.dropna(how='all')
        
        # Strip whitespace from string columns
        string_columns = ['Description', 'Category', 'SubCategory']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                # Replace 'nan' strings with empty strings
                df[col] = df[col].replace('nan', '')
        
        return df
        
    except Exception as e:
        print(f"DEBUG: Error cleaning data: {e}")
        return df

def import_transactions_robust(uploaded_file, date_col="Date", amount_col="Amount", desc_col="Description", 
                              debit_col="Debit", credit_col="Credit", balance_col="Balance"):
    """
    Import transactions using robust CSV parsing - replaces the pandas-based version
    """
    if uploaded_file is not None:
        try:
            st.info("🔄 Reading CSV file with robust parser...")
            
            # Use robust CSV reader
            df, header_row, skipped_lines = read_transaction_csv_robust(uploaded_file)
            
            if df.empty:
                st.error("No valid data found in CSV file")
                return pd.DataFrame(), pd.DataFrame()
            
            # Show parsing results
            st.success(f"✅ Successfully parsed CSV with {len(df)} rows")
            if skipped_lines > 0:
                st.warning(f"⚠️ Skipped {skipped_lines} malformed lines")
            
            # Clean the data
            df = clean_transaction_data(df)
            
            st.info(f"📊 After cleaning: {len(df)} rows with {len(df.columns)} columns")
            st.write("**Available columns:**", list(df.columns))
            
            # Show preview
            st.write("**Data Preview:**")
            st.dataframe(df.head())
            
            # Check format based on available columns
            has_debit_credit = debit_col in df.columns and credit_col in df.columns
            has_amount = amount_col in df.columns
            
            if has_debit_credit:
                required_cols = [date_col, desc_col, debit_col, credit_col]
                format_type = "debit_credit"
                st.info("✅ Detected Debit/Credit format")
            elif has_amount:
                required_cols = [date_col, amount_col, desc_col]
                format_type = "amount"
                st.info("✅ Detected Amount format")
            else:
                st.error("CSV must contain either 'Debit' and 'Credit' columns OR an 'Amount' column")
                return pd.DataFrame(), pd.DataFrame()

            # Check if required columns exist
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                return pd.DataFrame(), pd.DataFrame()

            # Work with only the required columns
            df_filtered = df[required_cols].copy()
            
            # Remove rows where required columns are all empty
            df_filtered = df_filtered.dropna(subset=required_cols, how='all')
            
            if df_filtered.empty:
                st.error("No valid data found after filtering")
                return pd.DataFrame(), pd.DataFrame()

            # Validate dates
            date_issues = df_filtered[date_col].isna().sum()
            if date_issues > 0:
                st.warning(f"⚠️ Found {date_issues} rows with invalid dates - these will be removed")
                df_filtered = df_filtered.dropna(subset=[date_col])
            
            if df_filtered.empty:
                st.error("No rows with valid dates found")
                return pd.DataFrame(), pd.DataFrame()

            # Process based on format type
            if format_type == "debit_credit":
                st.info("🔄 Processing Debit/Credit format...")
                
                # Remove rows where both debit and credit are 0 or empty
                df_filtered = df_filtered[
                    (df_filtered[debit_col].fillna(0) > 0) | (df_filtered[credit_col].fillna(0) > 0)
                ]
                
                if df_filtered.empty:
                    st.error("No rows with valid amounts found")
                    return pd.DataFrame(), pd.DataFrame()
                
                # Separate income (credits) and expenses (debits)
                income_mask = df_filtered[credit_col].fillna(0) > 0
                income_df = df_filtered[income_mask].copy()
                
                expense_mask = df_filtered[debit_col].fillna(0) > 0
                expense_df = df_filtered[expense_mask].copy()
                
                # Format income data
                if not income_df.empty:
                    format_income = pd.DataFrame({
                        'Description': income_df[desc_col].astype(str),
                        'Amount': income_df[credit_col].fillna(0),
                        'Date': income_df[date_col],
                        'Recurring': False,
                        'Frequency': 'None',
                        'Type': 'Income'
                    })
                else:
                    format_income = pd.DataFrame({
                        'Description': [], 'Amount': [], 'Date': [], 
                        'Recurring': [], 'Frequency': [], 'Type': []
                    })
                
                # Format expense data
                if not expense_df.empty:
                    format_expense = pd.DataFrame({
                        'Description': expense_df[desc_col].astype(str),
                        'Amount': expense_df[debit_col].fillna(0),
                        'Date': expense_df[date_col],
                        'Recurring': False,
                        'Frequency': 'None',
                        'Type': 'Expense'
                    })
                else:
                    format_expense = pd.DataFrame({
                        'Description': [], 'Amount': [], 'Date': [], 
                        'Recurring': [], 'Frequency': [], 'Type': []
                    })
                
            else:  # Amount format
                st.info("🔄 Processing Amount format...")
                
                # Remove rows with zero amounts
                df_filtered = df_filtered[df_filtered[amount_col].fillna(0) != 0]
                
                if df_filtered.empty:
                    st.error("No rows with valid amounts found")
                    return pd.DataFrame(), pd.DataFrame()
                
                # Separate income and expenses based on amount sign
                income_mask = df_filtered[amount_col] > 0
                income_df = df_filtered[income_mask].copy()
                expense_df = df_filtered[~income_mask].copy()
                
                # Make expense amounts positive
                if not expense_df.empty:
                    expense_df[amount_col] = expense_df[amount_col].abs()

                # Format data
                if not income_df.empty:
                    format_income = pd.DataFrame({
                        'Description': income_df[desc_col].astype(str),
                        'Amount': income_df[amount_col],
                        'Date': income_df[date_col],
                        'Recurring': False,
                        'Frequency': 'None',
                        'Type': 'Income'
                    })
                else:
                    format_income = pd.DataFrame({
                        'Description': [], 'Amount': [], 'Date': [], 
                        'Recurring': [], 'Frequency': [], 'Type': []
                    })

                if not expense_df.empty:
                    format_expense = pd.DataFrame({
                        'Description': expense_df[desc_col].astype(str),
                        'Amount': expense_df[amount_col],
                        'Date': expense_df[date_col],
                        'Recurring': False,
                        'Frequency': 'None',
                        'Type': 'Expense'
                    })
                else:
                    format_expense = pd.DataFrame({
                        'Description': [], 'Amount': [], 'Date': [], 
                        'Recurring': [], 'Frequency': [], 'Type': []
                    })

            # Final cleanup
            for df_result in [format_income, format_expense]:
                if not df_result.empty:
                    # Remove rows with zero or negative amounts
                    df_result = df_result[df_result['Amount'] > 0]
                    # Remove rows with empty descriptions
                    df_result = df_result[df_result['Description'].str.strip() != '']
                    df_result = df_result.reset_index(drop=True)

            st.success(f"✅ Processing complete!")
            st.write(f"- Income entries: {len(format_income)}")
            st.write(f"- Expense entries: {len(format_expense)}")
            
            return format_income, format_expense
            
        except Exception as e:
            st.error(f"Error importing transactions: {e}")
            st.write("**Troubleshooting tips:**")
            st.write("1. Ensure your CSV file is properly formatted")
            st.write("2. Check for special characters or unusual formatting")
            st.write("3. Try saving your file as a standard CSV format")
            
            import traceback
            st.write("**Error Details:**")
            st.code(traceback.format_exc())
    
    return pd.DataFrame(), pd.DataFrame()


def prepare_dataframes_for_projection(income_df, expense_df):
    """Prepare dataframes for projection by ensuring correct data types"""
    print(f"DEBUG prepare_dataframes: Input income shape: {income_df.shape}")
    print(f"DEBUG prepare_dataframes: Input expense shape: {expense_df.shape}")

    # Check for dropped rows
    before_income = len(income_df)
    income_df = income_df.dropna(subset=['Amount', 'Date'])
    after_income = len(income_df)
    print(f"DEBUG prepare_dataframes: Dropped {before_income - after_income} income rows")

    before_expense = len(expense_df)
    expense_df = expense_df.dropna(subset=['Amount', 'Date'])
    after_expense = len(expense_df)
    print(f"DEBUG prepare_dataframes: Dropped {before_expense - after_expense} expense rows")
    
    # Create copies to avoid modifying the originals
    income_copy = income_df.copy() if not income_df.empty else pd.DataFrame({
        'Description': [],
        'Amount': [],
        'Date': [],
        'Recurring': [],
        'Frequency': [],
        'Type': []
    })

    expense_copy = expense_df.copy() if not expense_df.empty else pd.DataFrame({
        'Description': [],
        'Amount': [],
        'Date': [],
        'Recurring': [],
        'Frequency': [],
        'Type': []
    })

    # Process income data
    if not income_copy.empty:
        print(f"DEBUG prepare_dataframes: Processing {len(income_copy)} income entries")
        
        # FIXED: Only drop rows with NaN in essential columns for projection
        essential_cols = ['Amount', 'Date']
        
        # Check which essential columns actually exist
        existing_essential = [col for col in essential_cols if col in income_copy.columns]
        if existing_essential:
            before_count = len(income_copy)
            income_copy = income_copy.dropna(subset=existing_essential)
            after_count = len(income_copy)
            print(f"DEBUG prepare_dataframes: Income rows after NaN removal: {before_count} -> {after_count}")
        
        # Ensure Date is datetime
        try:
            if 'Date' in income_copy.columns:
                income_copy['Date'] = pd.to_datetime(income_copy['Date'])
        except Exception as e:
            print(f"DEBUG prepare_dataframes: Income date conversion error: {e}")

        # Ensure Amount is float and positive
        try:
            if 'Amount' in income_copy.columns:
                income_copy['Amount'] = pd.to_numeric(income_copy['Amount'], errors='coerce')
                income_copy = income_copy[income_copy['Amount'] > 0]  # Only keep positive amounts
        except Exception as e:
            print(f"DEBUG prepare_dataframes: Income amount conversion error: {e}")

        # Ensure Recurring exists and is boolean
        try:
            if 'Recurring' not in income_copy.columns:
                income_copy['Recurring'] = False  # Default to False if missing
            income_copy['Recurring'] = income_copy['Recurring'].fillna(False).astype(bool)
        except Exception as e:
            print(f"DEBUG prepare_dataframes: Income recurring conversion error: {e}")
        
        # Ensure Frequency exists and is string
        try:
            if 'Frequency' not in income_copy.columns:
                income_copy['Frequency'] = 'None'  # Default to 'None' if missing
            income_copy['Frequency'] = income_copy['Frequency'].fillna('None').astype(str)
        except Exception as e:
            print(f"DEBUG prepare_dataframes: Income frequency conversion error: {e}")

        # Ensure Description exists
        try:
            if 'Description' not in income_copy.columns:
                income_copy['Description'] = 'Imported Transaction'
            income_copy['Description'] = income_copy['Description'].fillna('Imported Transaction').astype(str)
        except Exception as e:
            print(f"DEBUG prepare_dataframes: Income description conversion error: {e}")

    # Process expense data
    if not expense_copy.empty:
        print(f"DEBUG prepare_dataframes: Processing {len(expense_copy)} expense entries")
        
        # FIXED: Only drop rows with NaN in essential columns for projection
        essential_cols = ['Amount', 'Date']
        
        # Check which essential columns actually exist
        existing_essential = [col for col in essential_cols if col in expense_copy.columns]
        if existing_essential:
            before_count = len(expense_copy)
            expense_copy = expense_copy.dropna(subset=existing_essential)
            after_count = len(expense_copy)
            print(f"DEBUG prepare_dataframes: Expense rows after NaN removal: {before_count} -> {after_count}")
        
        # Ensure Date is datetime
        try:
            if 'Date' in expense_copy.columns:
                expense_copy['Date'] = pd.to_datetime(expense_copy['Date'])
        except Exception as e:
            print(f"DEBUG prepare_dataframes: Expense date conversion error: {e}")

        # Ensure Amount is float and positive
        try:
            if 'Amount' in expense_copy.columns:
                expense_copy['Amount'] = pd.to_numeric(expense_copy['Amount'], errors='coerce')
                expense_copy = expense_copy[expense_copy['Amount'] > 0]  # Only keep positive amounts
        except Exception as e:
            print(f"DEBUG prepare_dataframes: Expense amount conversion error: {e}")

        # Ensure Recurring exists and is boolean
        try:
            if 'Recurring' not in expense_copy.columns:
                expense_copy['Recurring'] = False  # Default to False if missing
            expense_copy['Recurring'] = expense_copy['Recurring'].fillna(False).astype(bool)
        except Exception as e:
            print(f"DEBUG prepare_dataframes: Expense recurring conversion error: {e}")
        
        # Ensure Frequency exists and is string
        try:
            if 'Frequency' not in expense_copy.columns:
                expense_copy['Frequency'] = 'None'  # Default to 'None' if missing
            expense_copy['Frequency'] = expense_copy['Frequency'].fillna('None').astype(str)
        except Exception as e:
            print(f"DEBUG prepare_dataframes: Expense frequency conversion error: {e}")

        # Ensure Description exists
        try:
            if 'Description' not in expense_copy.columns:
                expense_copy['Description'] = 'Imported Transaction'
            expense_copy['Description'] = expense_copy['Description'].fillna('Imported Transaction').astype(str)
        except Exception as e:
            print(f"DEBUG prepare_dataframes: Expense description conversion error: {e}")

    print(f"DEBUG prepare_dataframes: Output income shape: {income_copy.shape}")
    print(f"DEBUG prepare_dataframes: Output expense shape: {expense_copy.shape}")
    
    return income_copy, expense_copy


def generate_timeline(income_df, expense_df, initial_balance, months=18):
    """Generate a timeline of balance changes over the specified period"""
    today = datetime.now().date()
    end_date = today + relativedelta(months=months)

    # Create a date range for each day in the projection period
    date_range = pd.date_range(start=today, end=end_date, freq='D')
    timeline_df = pd.DataFrame(index=date_range)
    timeline_df['Income'] = 0.0
    timeline_df['Expense'] = 0.0
    timeline_df['Net'] = 0.0
    timeline_df['Balance'] = initial_balance

    # Debug: Print what we're processing
    print(f"DEBUG: Processing {len(income_df)} income entries and {len(expense_df)} expense entries")
    print(f"DEBUG: Date range from {today} to {end_date}")
    
    # Debug: Print all expense entries
    if not expense_df.empty:
        print("DEBUG: All expense entries:")
        for idx, row in expense_df.iterrows():
            print(f"  - {row['Description']}: ${row['Amount']} on {row['Date']}, Recurring: {row['Recurring']}")

    # Helper function to convert dates safely
    def safe_date_conversion(date_value):
        if pd.isna(date_value):
            return None
        try:
            if isinstance(date_value, str):
                return pd.to_datetime(date_value).date()
            elif isinstance(date_value, pd.Timestamp):
                return date_value.date()
            elif hasattr(date_value, 'date'):
                return date_value.date()
            else:
                return date_value
        except:
            return None

    # Process income transactions
    if not income_df.empty:
        for idx, row in income_df.iterrows():
            try:
                amount = float(row['Amount'])
                date = safe_date_conversion(row['Date'])
                
                print(f"DEBUG Income: {row['Description']} - ${amount} on {date}, Recurring: {row['Recurring']}, Freq: {row['Frequency']}")
                
                if date is None:
                    print(f"DEBUG: Skipping income entry due to invalid date")
                    continue

                # Handle recurring income
                if row['Recurring'] and row['Frequency'] != 'None':
                    freq = row['Frequency']
                    current_date = max(date, today)
                    count = 0

                    while current_date <= end_date and count < 1000:  # Safety limit
                        index_date = pd.Timestamp(current_date)
                        if index_date in timeline_df.index:
                            timeline_df.loc[index_date, 'Income'] += amount
                            count += 1

                        # Move to the next occurrence
                        if freq == 'Daily':
                            current_date += relativedelta(days=1)
                        elif freq == 'Weekly':
                            current_date += relativedelta(weeks=1)
                        elif freq == 'Biweekly':
                            current_date += relativedelta(weeks=2)
                        elif freq == 'Monthly':
                            current_date += relativedelta(months=1)
                        elif freq == 'Quarterly':
                            current_date += relativedelta(months=3)
                        elif freq == 'Annually':
                            current_date += relativedelta(years=1)
                        else:
                            break
                    print(f"DEBUG: Added {count} recurring income entries")
                else:
                    # One-time income
                    if date >= today:
                        index_date = pd.Timestamp(date)
                        if index_date in timeline_df.index:
                            timeline_df.loc[index_date, 'Income'] += amount
                            print(f"DEBUG: Added one-time income on {date}")
                        else:
                            print(f"DEBUG: Date {date} not in timeline range")
                    else:
                        print(f"DEBUG: Income date {date} is in the past, skipping")
            except Exception as e:
                print(f"Error processing income row {idx}: {e}")
                continue

    # Process expense transactions
    if not expense_df.empty:
        for idx, row in expense_df.iterrows():
            try:
                amount = float(row['Amount'])
                date = safe_date_conversion(row['Date'])
                
                print(f"DEBUG Expense: {row['Description']} - ${amount} on {date}, Recurring: {row['Recurring']}, Freq: {row['Frequency']}")
                
                if date is None:
                    print(f"DEBUG: Skipping expense entry due to invalid date")
                    continue

                # Handle recurring expenses
                if row['Recurring'] and row['Frequency'] != 'None':
                    freq = row['Frequency']
                    current_date = max(date, today)
                    count = 0

                    while current_date <= end_date and count < 1000:  # Safety limit
                        index_date = pd.Timestamp(current_date)
                        if index_date in timeline_df.index:
                            timeline_df.loc[index_date, 'Expense'] += amount
                            count += 1

                        # Move to the next occurrence
                        if freq == 'Daily':
                            current_date += relativedelta(days=1)
                        elif freq == 'Weekly':
                            current_date += relativedelta(weeks=1)
                        elif freq == 'Biweekly':
                            current_date += relativedelta(weeks=2)
                        elif freq == 'Monthly':
                            current_date += relativedelta(months=1)
                        elif freq == 'Quarterly':
                            current_date += relativedelta(months=3)
                        elif freq == 'Annually':
                            current_date += relativedelta(years=1)
                        else:
                            break
                    print(f"DEBUG: Added {count} recurring expense entries")
                else:
                    # One-time expense
                    if date >= today:
                        index_date = pd.Timestamp(date)
                        if index_date in timeline_df.index:
                            timeline_df.loc[index_date, 'Expense'] += amount
                            print(f"DEBUG: Added one-time expense on {date}")
                        else:
                            print(f"DEBUG: Date {date} not in timeline range")
                    else:
                        print(f"DEBUG: Expense date {date} is in the past, skipping")
            except Exception as e:
                print(f"Error processing expense row {idx}: {e}")
                continue

    # Calculate net and running balance
    timeline_df['Net'] = timeline_df['Income'] - timeline_df['Expense']

    # Debug: Print totals before balance calculation
    total_income = timeline_df['Income'].sum()
    total_expenses = timeline_df['Expense'].sum()
    print(f"DEBUG: Total timeline income: ${total_income:.2f}")
    print(f"DEBUG: Total timeline expenses: ${total_expenses:.2f}")

    # Calculate cumulative balance
    running_balance = initial_balance
    for i in range(len(timeline_df)):
        current_date = timeline_df.index[i]
        daily_net = timeline_df.loc[current_date, 'Net']
        running_balance += daily_net
        timeline_df.loc[current_date, 'Balance'] = running_balance

    # Create monthly summary
    monthly_summary = timeline_df.resample('ME').agg({
        'Income': 'sum',
        'Expense': 'sum',
        'Net': 'sum',
        'Balance': 'last'
    })

    return timeline_df, monthly_summary


def plot_balance_over_time(timeline_df, monthly_df):
    """Create a plot of the bank balance over time"""
    # Daily chart with plotly
    fig1 = go.Figure()

    # Enhanced approach: Handle zero crossings properly
    balance_values = timeline_df['Balance'].values
    dates = timeline_df.index
    
    # Create line segments with appropriate colors, handling zero crossings
    for i in range(len(dates) - 1):
        current_balance = balance_values[i]
        next_balance = balance_values[i + 1]
        
        # Check if this segment crosses zero
        if (current_balance >= 0 and next_balance < 0) or (current_balance < 0 and next_balance >= 0):
            # This segment crosses zero - split it at the zero crossing point
            
            # Calculate the date where balance crosses zero
            # Linear interpolation to find zero crossing point
            if current_balance != next_balance:  # Avoid division by zero
                ratio = abs(current_balance) / abs(current_balance - next_balance)
                time_diff = dates[i + 1] - dates[i]
                zero_crossing_date = dates[i] + ratio * time_diff
                
                # First segment (from current to zero crossing)
                first_color = '#2196F3' if current_balance >= 0 else '#F44336'
                fig1.add_trace(
                    go.Scatter(
                        x=[dates[i], zero_crossing_date],
                        y=[current_balance, 0],
                        mode='lines',
                        line=dict(color=first_color, width=2),
                        showlegend=False,
                        hovertemplate='%{x}<br>Balance: $%{y:.2f}<extra></extra>',
                    )
                )
                
                # Second segment (from zero crossing to next point)
                second_color = '#2196F3' if next_balance >= 0 else '#F44336'
                fig1.add_trace(
                    go.Scatter(
                        x=[zero_crossing_date, dates[i + 1]],
                        y=[0, next_balance],
                        mode='lines',
                        line=dict(color=second_color, width=2),
                        showlegend=False,
                        hovertemplate='%{x}<br>Balance: $%{y:.2f}<extra></extra>',
                    )
                )
            else:
                # Fallback for edge case
                color = '#2196F3' if current_balance >= 0 else '#F44336'
                fig1.add_trace(
                    go.Scatter(
                        x=[dates[i], dates[i + 1]],
                        y=[current_balance, next_balance],
                        mode='lines',
                        line=dict(color=color, width=2),
                        showlegend=False,
                        hovertemplate='%{x}<br>Balance: $%{y:.2f}<extra></extra>',
                    )
                )
        else:
            # Normal segment - doesn't cross zero
            color = '#2196F3' if current_balance >= 0 else '#F44336'
            fig1.add_trace(
                go.Scatter(
                    x=[dates[i], dates[i + 1]],
                    y=[current_balance, next_balance],
                    mode='lines',
                    line=dict(color=color, width=2),
                    showlegend=False,
                    hovertemplate='%{x}<br>Balance: $%{y:.2f}<extra></extra>',
                )
            )
    
    # Add legend entries (dummy traces for legend only)
    has_positive = (timeline_df['Balance'] >= 0).any()
    has_negative = (timeline_df['Balance'] < 0).any()
    
    if has_positive:
        fig1.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='#2196F3', width=2),
                name='Positive Balance',
                showlegend=True
            )
        )
    
    if has_negative:
        fig1.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='#F44336', width=2),
                name='Negative Balance',
                showlegend=True
            )
        )

    # Add zero line for reference
    fig1.add_shape(
        type="line",
        x0=timeline_df.index.min(),
        y0=0,
        x1=timeline_df.index.max(),
        y1=0,
        line=dict(color="gray", width=1.5, dash="dash"),
    )

    # Update layout
    fig1.update_layout(
        title='Daily Bank Balance Projection',
        xaxis_title='Date',
        yaxis_title='Balance ($)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=500,
    )

    # Monthly chart with same zero-crossing logic
    fig2 = go.Figure()

    # Enhanced approach for monthly chart too
    monthly_balance_values = monthly_df['Balance'].values
    monthly_dates = monthly_df.index
    
    # Create monthly line segments with appropriate colors, handling zero crossings
    for i in range(len(monthly_dates) - 1):
        current_balance = monthly_balance_values[i]
        next_balance = monthly_balance_values[i + 1]
        
        # Check if this segment crosses zero
        if (current_balance >= 0 and next_balance < 0) or (current_balance < 0 and next_balance >= 0):
            # This segment crosses zero - split it at the zero crossing point
            
            # Calculate the date where balance crosses zero
            if current_balance != next_balance:  # Avoid division by zero
                ratio = abs(current_balance) / abs(current_balance - next_balance)
                time_diff = monthly_dates[i + 1] - monthly_dates[i]
                zero_crossing_date = monthly_dates[i] + ratio * time_diff
                
                # First segment (from current to zero crossing)
                first_color = '#2196F3' if current_balance >= 0 else '#F44336'
                fig2.add_trace(
                    go.Scatter(
                        x=[monthly_dates[i], zero_crossing_date],
                        y=[current_balance, 0],
                        mode='lines',
                        line=dict(color=first_color, width=3),
                        showlegend=False,
                        hovertemplate='%{x}<br>Balance: $%{y:.2f}<extra></extra>',
                    )
                )
                
                # Second segment (from zero crossing to next point)
                second_color = '#2196F3' if next_balance >= 0 else '#F44336'
                fig2.add_trace(
                    go.Scatter(
                        x=[zero_crossing_date, monthly_dates[i + 1]],
                        y=[0, next_balance],
                        mode='lines',
                        line=dict(color=second_color, width=3),
                        showlegend=False,
                        hovertemplate='%{x}<br>Balance: $%{y:.2f}<extra></extra>',
                    )
                )
            else:
                # Fallback for edge case
                color = '#2196F3' if current_balance >= 0 else '#F44336'
                fig2.add_trace(
                    go.Scatter(
                        x=[monthly_dates[i], monthly_dates[i + 1]],
                        y=[current_balance, next_balance],
                        mode='lines',
                        line=dict(color=color, width=3),
                        showlegend=False,
                        hovertemplate='%{x}<br>Balance: $%{y:.2f}<extra></extra>',
                    )
                )
        else:
            # Normal segment - doesn't cross zero
            color = '#2196F3' if current_balance >= 0 else '#F44336'
            fig2.add_trace(
                go.Scatter(
                    x=[monthly_dates[i], monthly_dates[i + 1]],
                    y=[current_balance, next_balance],
                    mode='lines',
                    line=dict(color=color, width=3),
                    showlegend=False,
                    hovertemplate='%{x}<br>Balance: $%{y:.2f}<extra></extra>',
                )
            )

    # Add monthly income bars
    fig2.add_trace(
        go.Bar(
            x=monthly_df.index,
            y=monthly_df['Income'],
            name='Monthly Income',
            marker_color='#4CAF50',
            hovertemplate='%{x}<br>Income: $%{y:.2f}<extra></extra>',
            opacity=0.7,
        )
    )

    # Add monthly expense bars
    fig2.add_trace(
        go.Bar(
            x=monthly_df.index,
            y=monthly_df['Expense'],
            name='Monthly Expenses',
            marker_color='#F44336',
            hovertemplate='%{x}<br>Expenses: $%{y:.2f}<extra></extra>',
            opacity=0.7,
        )
    )

    # Add legend entries for monthly balance line
    monthly_has_positive = (monthly_df['Balance'] >= 0).any()
    monthly_has_negative = (monthly_df['Balance'] < 0).any()
    
    if monthly_has_positive:
        fig2.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='#2196F3', width=3),
                name='Positive Balance',
                showlegend=True
            )
        )
    
    if monthly_has_negative:
        fig2.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='#F44336', width=3),
                name='Negative Balance',
                showlegend=True
            )
        )

    # Add zero line for reference
    fig2.add_shape(
        type="line",
        x0=monthly_df.index.min(),
        y0=0,
        x1=monthly_df.index.max(),
        y1=0,
        line=dict(color="gray", width=1.5, dash="dash"),
    )

    # Update layout
    fig2.update_layout(
        title='Monthly Income, Expenses, and Balance',
        xaxis_title='Month',
        yaxis_title='Amount ($)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        barmode='group',
        height=500,
    )

    return fig1, fig2


# Move this function definition to BEFORE the import section (around line 1100)
def process_imported_data(income_from_import, expenses_from_import):
    print("DEBUG: process_imported_data function called")
    print(f"DEBUG: Income from Import - {len(income_from_import)} rows")
    print(f"DEBUG: Expenses from Import - {len(expenses_from_import)} rows")
    print(f"DEBUG: Income from Import - Empty: {income_from_import.empty}")
    print(f"DEBUG: Expenses from Import - Empty: {expenses_from_import.empty}")

    # First, ensure proper data types for the imported data
    if not income_from_import.empty:
        income_from_import = income_from_import.copy()
        income_from_import['Recurring'] = income_from_import['Recurring'].astype(bool)
        income_from_import['Amount'] = income_from_import['Amount'].astype(float)
        income_from_import['Date'] = pd.to_datetime(income_from_import['Date'])
        income_from_import['Description'] = income_from_import['Description'].astype(str)
        income_from_import['Frequency'] = income_from_import['Frequency'].astype(str)
        income_from_import['Type'] = income_from_import['Type'].astype(str)

    if not expenses_from_import.empty:
        expenses_from_import = expenses_from_import.copy()
        expenses_from_import['Recurring'] = expenses_from_import['Recurring'].astype(bool)
        expenses_from_import['Amount'] = expenses_from_import['Amount'].astype(float)
        expenses_from_import['Date'] = pd.to_datetime(expenses_from_import['Date'])
        expenses_from_import['Description'] = expenses_from_import['Description'].astype(str)
        expenses_from_import['Frequency'] = expenses_from_import['Frequency'].astype(str)
        expenses_from_import['Type'] = expenses_from_import['Type'].astype(str)

    # Debug: Display imported data preview
    st.write("**Debug: Imported Data Preview**")
    if not income_from_import.empty:
        st.write("Income from Import:")
        st.dataframe(income_from_import.head())
    else:
        st.write("No income data imported.")

    if not expenses_from_import.empty:
        st.write("Expenses from Import:")
        st.dataframe(expenses_from_import.head())
    else:
        st.write("No expense data imported.")

    # Let user choose how to handle the imported data
    selected_option = st.radio(
        "How would you like to handle the imported data?",
        ("Add to existing data", "Replace existing data"),
        key="import_handling_option_temp"
    )

    # Always display the Confirm Import button
    if st.button("Confirm Import"):
        try:
            print("DEBUG: Confirm Import button clicked")
            print(f"DEBUG: Selected handling option: {selected_option}")
            print(f"DEBUG: Income from Import - {len(income_from_import)} rows")
            print(f"DEBUG: Expenses from Import - {len(expenses_from_import)} rows")

            if selected_option == "Add to existing data":
                if not income_from_import.empty:
                    if st.session_state.income_data.empty:
                        st.session_state.income_data = income_from_import.copy()
                    else:
                        st.session_state.income_data = pd.concat([st.session_state.income_data, income_from_import], ignore_index=True)

                if not expenses_from_import.empty:
                    if st.session_state.expense_data.empty:
                        st.session_state.expense_data = expenses_from_import.copy()
                    else:
                        st.session_state.expense_data = pd.concat([st.session_state.expense_data, expenses_from_import], ignore_index=True)

            elif selected_option == "Replace existing data":
                if not income_from_import.empty:
                    st.session_state.income_data = income_from_import.copy()
                else:
                    st.session_state.income_data = pd.DataFrame()

                if not expenses_from_import.empty:
                    st.session_state.expense_data = expenses_from_import.copy()
                else:
                    st.session_state.expense_data = pd.DataFrame()

            st.session_state.last_update = datetime.now()
            save_session_data()

            # Debug: Display session state after adding imported data
            st.write("**Debug: Current Session State After Import**")
            st.write("Income Data:")
            st.dataframe(st.session_state.income_data.head())
            st.write("Expense Data:")
            st.dataframe(st.session_state.expense_data.head())

            print("✅ Import confirmed. Data saved. Triggering rerun.")
            print("Income entries:", len(st.session_state.income_data))
            print("Expense entries:", len(st.session_state.expense_data))

        except Exception as e:
            st.error(f"Error during import: {e}")
            import traceback
            st.code(traceback.format_exc())

        st.rerun()

# Main application interface
st.title("💰 Personal Budget Planner 0.028")

# Sidebar for configuration
st.sidebar.header("Configuration")
st.session_state.bank_balance = st.sidebar.number_input("Initial Bank Balance ($)", value=st.session_state.bank_balance,
                                                        step=100.0)
st.session_state.projection_months = st.sidebar.slider("Projection Period (Months)", min_value=1, max_value=60,
                                                       value=st.session_state.projection_months)

# Data Import/Export Section in sidebar
st.sidebar.header("Data Management")

# Save Data
save_option = st.sidebar.expander("Save Budget Data")
with save_option:
    save_filename = st.text_input("Filename", "my_budget_plan")
    
    # Choose save location
    save_location = st.selectbox(
        "Save Location",
        ["Current Directory", "Downloads Folder", "Custom Path"]
    )
    
    if save_location == "Custom Path":
        custom_path = st.text_input("Enter full path", "C:\\")
    
    # Generate the CSV data
    if st.session_state.income_data.empty and st.session_state.expense_data.empty:
        st.info("No data to save yet.")
    else:
        if st.button("💾 Save Budget Data to File"):
            try:
                # Debug: Check what data we're trying to save
                st.write("DEBUG - Income data shape:", st.session_state.income_data.shape)
                st.write("DEBUG - Expense data shape:", st.session_state.expense_data.shape)
                st.write("DEBUG - Bank balance:", st.session_state.bank_balance)
                
                saved_data = save_data(
                    st.session_state.income_data,
                    st.session_state.expense_data,
                    st.session_state.bank_balance,
                    save_filename
                )
                
                st.write("DEBUG - Final data shape:", saved_data.shape)
                st.write("DEBUG - Final data preview:")
                st.dataframe(saved_data.head())
                
                # Determine save path
                if save_location == "Current Directory":
                    save_path = f"{save_filename}.csv"
                elif save_location == "Downloads Folder":
                    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
                    save_path = os.path.join(downloads_path, f"{save_filename}.csv")
                else:  # Custom Path
                    save_path = os.path.join(custom_path, f"{save_filename}.csv")
                
                st.write("DEBUG - Save path:", save_path)
                
                # Save to file
                saved_data.to_csv(save_path, index=False)
                
                # Verify file was created
                if os.path.exists(save_path):
                    file_size = os.path.getsize(save_path)
                    st.success(f"✅ Budget data saved successfully!")
                    st.info(f"📁 File: {save_path}")
                    st.info(f"📊 Size: {file_size} bytes")
                else:
                    st.error("❌ File was not created!")
                
            except Exception as e:
                st.error(f"❌ Error saving file: {str(e)}")
                st.write("Full error details:", e)
                import traceback
                st.code(traceback.format_exc())

    # Load Data
load_option = st.sidebar.expander("Load Budget Data")
with load_option:
    uploaded_file = st.file_uploader("Upload a saved budget file", type="csv")

    if uploaded_file is not None and st.button("Load Budget Data"):
        income_df, expense_df, balance = load_data(uploaded_file)
        if not income_df.empty or not expense_df.empty:
            st.session_state.income_data = income_df
            st.session_state.expense_data = expense_df
            st.session_state.bank_balance = balance
            
            # Auto-save after loading data
            save_session_data()
            
            st.success("Budget data loaded successfully!")
            st.rerun()  # Force refresh after loading
        else:
            st.error("Failed to load budget data. Please check the file format.")

# Import Transactions - updated to use robust CSV parser
import_option = st.sidebar.expander("Import Past Transactions")
with import_option:
    st.write("Upload a transaction history")
    
    # CSV file uploader
    uploaded_file = st.file_uploader("Drag and drop file here or", type="csv", key="TransactionFileUpload", 
                                     help="Limit 200MB per file - CSV")
    
    # Add Clear Previous Data button
    if st.button("Clear Previous Data"):
        # Clear session state
        if 'transactions' in st.session_state:
            st.session_state['transactions'] = []
        
        # Create empty transactions file
        try:
            with open(TRANSACTIONS_FILE, "w") as f:
                json.dump([], f)
            st.success("All transaction data cleared successfully.")
        except Exception as e:
            st.error(f"Failed to clear transaction data: {e}")
        
        # Force a rerun to update the UI
        st.rerun()
    
    # Initialize variables
    new_transactions = []
    error_message = None

    # Only process if a file is uploaded
    if uploaded_file is not None:
        try:
            # Read the file
            uploaded_file.seek(0)  # Reset file pointer
            file_content = uploaded_file.read().decode('utf-8')
            csv_reader = csv.reader(io.StringIO(file_content))

            # Get header row
            headers = next(csv_reader)  # Assume header exists
            headers = [h.lower().strip() for h in headers]  # Normalize headers

            # Identify relevant column indices
            date_col = None
            desc_col = None
            amount_col = None
            credit_col = None
            debit_col = None

            # Map headers to indices
            for idx, header in enumerate(headers):
                if "date" in header.lower():
                    date_col = idx
                elif "description" in header.lower() or "memo" in header.lower() or "payee" in header.lower():
                    desc_col = idx
                elif "amount" in header.lower() or "amt" in header.lower():
                    amount_col = idx
                elif "credit" in header.lower():
                    credit_col = idx
                elif "debit" in header.lower():
                    debit_col = idx

            # Validate required columns
            if date_col is None or desc_col is None:
                raise ValueError("CSV must contain 'Date' and 'Description' columns.")
            if amount_col is None and (credit_col is None or debit_col is None):
                raise ValueError("CSV must contain either an 'Amount' column or both 'Credit' and 'Debit' columns.")

            # Windows-standard date formats to try
            date_formats = [
                "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y",
                "%b %d %Y", "%d %b %Y", "%Y %b %d",
                "%m/%d/%y", "%d/%m/%y", "%y-%m-%d"
            ]

            # Process each row
            uploaded_file.seek(0)  # Reset to start
            csv_reader = csv.reader(io.StringIO(uploaded_file.read().decode('utf-8')))
            next(csv_reader)  # Skip header
            for row in csv_reader:
                try:
                    # Parse date
                    date_str = row[date_col].strip()
                    date = None
                    try:
                        date = parse(date_str, fuzzy=True)
                    except ValueError:
                        for fmt in date_formats:
                            try:
                                date = datetime.strptime(date_str, fmt)
                                break
                            except ValueError:
                                continue
                    if date is None:
                        raise ValueError(f"Could not parse date: {date_str}")

                    # Get description
                    description = row[desc_col].strip()

                    # Get amount
                    amount = None
                    if amount_col is not None:
                        amount_str = row[amount_col].strip().replace('$', '').replace(',', '')
                        amount = float(amount_str)
                    else:
                        credit_str = row[credit_col].strip().replace('$', '').replace(',', '') if row[credit_col].strip() else '0'
                        debit_str = row[debit_col].strip().replace('$', '').replace(',', '') if row[debit_col].strip() else '0'
                        credit = float(credit_str) if credit_str else 0.0
                        debit = float(debit_str) if debit_str else 0.0
                        amount = credit - debit

                    # Store transaction
                    new_transactions.append({
                        "Date": date,
                        "Description": description,
                        "Amount": amount
                    })

                except Exception as e:
                    st.warning(f"Skipped row due to error: {str(e)}")
                    continue

            # Store transactions in session state and save to file
            if new_transactions:
                st.session_state['transactions'] = new_transactions
                save_transactions(new_transactions)
                st.success(f"Successfully imported {len(new_transactions)} transactions.")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please ensure the CSV contains the required columns and is properly formatted.")

    # Convert transactions to budget format and add to budget data
    if 'transactions' in st.session_state and st.session_state['transactions']:
        st.write(f"**{len(st.session_state['transactions'])} transactions available for import to budget**")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add to Budget Data"):
                # Convert transactions to budget format
                income_entries = []
                expense_entries = []
                
                for t in st.session_state['transactions']:
                    # Positive amounts are income, negative are expenses
                    if t["Amount"] > 0:
                        income_entries.append({
                            'Description': t["Description"],
                            'Amount': abs(t["Amount"]),  # Ensure positive
                            'Date': t["Date"],
                            'Recurring': False,
                            'Frequency': "None",
                            'Type': 'Income'
                        })
                    else:
                        expense_entries.append({
                            'Description': t["Description"],
                            'Amount': abs(t["Amount"]),  # Make positive for expenses
                            'Date': t["Date"],
                            'Recurring': False,
                            'Frequency': "None",
                            'Type': 'Expense'
                        })
                
                # Create DataFrames from the lists
                if income_entries:
                    new_income_df = pd.DataFrame(income_entries)
                    # Add to existing income data
                    st.session_state.income_data = pd.concat(
                        [st.session_state.income_data, new_income_df], 
                        ignore_index=True
                    )
                    
                if expense_entries:
                    new_expense_df = pd.DataFrame(expense_entries)
                    # Add to existing expense data
                    st.session_state.expense_data = pd.concat(
                        [st.session_state.expense_data, new_expense_df], 
                        ignore_index=True
                    )
                
                # Save the updated budget data
                st.session_state.last_update = datetime.now()
                save_session_data()
                
                st.success(f"Added {len(income_entries)} income entries and {len(expense_entries)} expense entries to your budget!")
                st.rerun()
        
        with col2:
            if st.button("Replace Budget Data"):
                # Convert transactions to budget format
                income_entries = []
                expense_entries = []
                
                for t in st.session_state['transactions']:
                    # Positive amounts are income, negative are expenses
                    if t["Amount"] > 0:
                        income_entries.append({
                            'Description': t["Description"],
                            'Amount': abs(t["Amount"]),  # Ensure positive
                            'Date': t["Date"],
                            'Recurring': False,
                            'Frequency': "None",
                            'Type': 'Income'
                        })
                    else:
                        expense_entries.append({
                            'Description': t["Description"],
                            'Amount': abs(t["Amount"]),  # Make positive for expenses
                            'Date': t["Date"],
                            'Recurring': False,
                            'Frequency': "None",
                            'Type': 'Expense'
                        })
                
                # Replace existing data with new data
                st.session_state.income_data = pd.DataFrame(income_entries) if income_entries else pd.DataFrame({
                    'Description': [],
                    'Amount': [],
                    'Date': [],
                    'Recurring': [],
                    'Frequency': [],
                    'Type': []
                })
                
                st.session_state.expense_data = pd.DataFrame(expense_entries) if expense_entries else pd.DataFrame({
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
                
                st.success(f"Replaced budget data with {len(income_entries)} income entries and {len(expense_entries)} expense entries!")
                st.rerun()
    else:
        st.info("Import transactions first before adding them to your budget.")

# Add this code at the end of your file, after all the sidebar components

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Dashboard", "Income", "Expenses"])

# Dashboard Tab (tab1)
with tab1:
    st.header("Budget Dashboard")
    
    # Display imported transactions if available
    if 'transactions' in st.session_state and st.session_state['transactions']:
        with st.expander("View Imported Transactions"):
            st.write("Processed Transactions:")
            
            # Build HTML table row by row
            header_row = "<tr><th class='col-date'>Date</th><th class='col-desc'>Description</th><th class='col-amount'>Amount</th></tr>"
            rows = []
            for t in st.session_state['transactions']:
                # Format the data and escape HTML special characters
                date = html.escape(t["Date"].strftime("%Y-%m-%d"))
                desc = html.escape(t["Description"])
                amount = html.escape(f"${t['Amount']:.2f}")
                row = f"<tr><td class='col-date'>{date}</td><td class='col-desc'>{desc}</td><td class='col-amount'>{amount}</td></tr>"
                rows.append(row)

            # Combine into a single HTML string
            table_html = f"""
            <div class="table-container">
              <table class="fixed-table">
                <thead>{header_row}</thead>
                <tbody>{"".join(rows)}</tbody>
              </table>
            </div>
            """

            # Display the table
            st.markdown(table_html, unsafe_allow_html=True)
            st.write(f"Total transactions loaded: {len(st.session_state['transactions'])}")

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
    
    # Prepare data for projection
    income_clean, expense_clean = prepare_dataframes_for_projection(
        st.session_state.income_data, 
        st.session_state.expense_data
    )
    
    # Generate timeline data
    timeline_df, monthly_df = generate_timeline(
        income_clean, 
        expense_clean, 
        st.session_state.bank_balance,
        months=st.session_state.projection_months
    )
    
    # Create and display charts
    daily_chart, monthly_chart = plot_balance_over_time(timeline_df, monthly_df)
    st.plotly_chart(monthly_chart, use_container_width=True)
    
    with st.expander("Show Detailed Daily Projection"):
        st.plotly_chart(daily_chart, use_container_width=True)
    
    with st.expander("Show Detailed Monthly Projections"):
        st.dataframe(monthly_df)

# Income Tab (tab2)
with tab2:
    st.header("Income Management")
    
    # Add income entry form
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
    
    # Display existing income entries
    st.subheader("Existing Income")
    if not st.session_state.income_data.empty:
        income_view = st.session_state.income_data.copy()
        income_view['Date'] = income_view['Date'].dt.strftime('%Y-%m-%d')
        st.dataframe(income_view, use_container_width=True)
        
        # Delete income entries
        if st.button("Delete Selected Income"):
            st.session_state.income_data = pd.DataFrame({
                'Description': [], 'Amount': [], 'Date': [], 
                'Recurring': [], 'Frequency': [], 'Type': []
            })
            st.session_state.last_update = datetime.now()
            save_session_data()
            st.success("All income entries deleted!")
            st.rerun()
    else:
        st.info("No income entries yet. Add some using the form above.")

# Expenses Tab (tab3)
with tab3:
    st.header("Expense Management")
    
    # Add expense entry form
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
    
    # Display existing expense entries
    st.subheader("Existing Expenses")
    if not st.session_state.expense_data.empty:
        expense_view = st.session_state.expense_data.copy()
        expense_view['Date'] = expense_view['Date'].dt.strftime('%Y-%m-%d')
        st.dataframe(expense_view, use_container_width=True)
        
        # Delete expense entries
        if st.button("Delete Selected Expenses"):
            st.session_state.expense_data = pd.DataFrame({
                'Description': [], 'Amount': [], 'Date': [], 
                'Recurring': [], 'Frequency': [], 'Type': []
            })
            st.session_state.last_update = datetime.now()
            save_session_data()
            st.success("All expense entries deleted!")
            st.rerun()
    else:
        st.info("No expense entries yet. Add some using the form above.")
