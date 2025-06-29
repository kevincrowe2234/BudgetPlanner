import pandas as pd
import csv
from io import StringIO
from datetime import datetime
from dateutil.parser import parse
import streamlit as st
from utils.config import DATE_FORMATS

@st.cache_data
def parse_transaction_csv(file_content):
    """Parse CSV content into a dataframe with proper column detection"""
    try:
        csv_reader = csv.reader(StringIO(file_content))
        headers = next(csv_reader)  # Assume header exists
        headers = [h.lower().strip() for h in headers]  # Normalize headers
        
        # Map headers to indices
        column_mapping = {}
        for idx, header in enumerate(headers):
            if "date" in header:
                column_mapping["Date"] = idx
            elif any(term in header for term in ["description", "memo", "payee", "narration"]):
                column_mapping["Description"] = idx
            elif "amount" in header or "amt" in header:
                column_mapping["Amount"] = idx
            elif "credit" in header:
                column_mapping["Credit"] = idx
            elif "debit" in header:
                column_mapping["Debit"] = idx
        
        # Validate required columns
        if "Date" not in column_mapping or "Description" not in column_mapping:
            raise ValueError("CSV must contain 'Date' and 'Description' columns")
        
        if "Amount" not in column_mapping and ("Credit" not in column_mapping or "Debit" not in column_mapping):
            raise ValueError("CSV must contain either 'Amount' or both 'Credit' and 'Debit' columns")
        
        # Process rows into transactions
        transactions = []
        for row in csv_reader:
            if not row or all(cell.strip() == '' for cell in row):
                continue  # Skip empty rows
                
            try:
                # Parse date with multiple formats
                date_str = row[column_mapping["Date"]].strip()
                date = parse_date(date_str)
                if date is None:
                    continue
                    
                # Get description
                description = row[column_mapping["Description"]].strip()
                
                # Get amount
                if "Amount" in column_mapping:
                    amount = parse_amount(row[column_mapping["Amount"]])
                else:
                    credit = parse_amount(row[column_mapping["Credit"]]) if row[column_mapping["Credit"]].strip() else 0
                    debit = parse_amount(row[column_mapping["Debit"]]) if row[column_mapping["Debit"]].strip() else 0
                    amount = credit - debit
                
                # Add transaction
                transactions.append({
                    "Date": date,
                    "Description": description,
                    "Amount": amount
                })
                
            except Exception as e:
                # Skip problematic rows and continue
                print(f"Error processing row: {e}")
                continue
                
        return transactions
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {str(e)}")

def parse_date(date_str):
    """Parse date string using multiple formats"""
    try:
        return parse(date_str, fuzzy=True)
    except ValueError:
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
    return None

def parse_amount(amount_str):
    """Parse amount string to float, handling currency symbols and commas"""
    clean_amount = amount_str.strip().replace('$', '').replace(',', '')
    clean_amount = clean_amount.replace('(', '-').replace(')', '')
    return float(clean_amount) if clean_amount else 0.0

def transactions_to_budget_data(transactions, replace=False):
    """Convert transactions to budget data format"""
    income_entries = []
    expense_entries = []
    
    for t in transactions:
        if t["Amount"] > 0:
            income_entries.append({
                'Description': t["Description"],
                'Amount': abs(t["Amount"]),
                'Date': t["Date"],
                'Recurring': False,
                'Frequency': "None",
                'Type': 'Income'
            })
        else:
            expense_entries.append({
                'Description': t["Description"],
                'Amount': abs(t["Amount"]),
                'Date': t["Date"],
                'Recurring': False,
                'Frequency': "None",
                'Type': 'Expense'
            })
    
    return pd.DataFrame(income_entries) if income_entries else pd.DataFrame(), \
           pd.DataFrame(expense_entries) if expense_entries else pd.DataFrame()