import streamlit as st
import csv
from datetime import datetime
from dateutil.parser import parse
import io
import json
from pathlib import Path
import html
import os

# Use an absolute path for the transactions file
TRANSACTIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transactions.json")

# Function to save transactions to a file
def save_transactions(transactions):
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
        st.sidebar.success(f"Saved {len(transactions)} transactions to {TRANSACTIONS_FILE}")
    except Exception as e:
        st.sidebar.error(f"Failed to save transactions: {e}")

# Function to load transactions from a file
def load_transactions():
    if not os.path.exists(TRANSACTIONS_FILE):
        st.sidebar.warning(f"No transaction file found at {TRANSACTIONS_FILE}")
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
            
            # Success message removed
            return transactions
    except Exception as e:
        st.sidebar.error(f"Failed to load transactions: {e}")
        return []

# Initialize session state if needed
if "transactions" not in st.session_state:
    loaded_transactions = load_transactions()
    st.session_state["transactions"] = loaded_transactions

# Add CSS for the table
st.markdown("""
<style>
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

# Sidebar configuration with collapsible section
with st.sidebar:
    with st.expander("Import Past Transactions"):
        st.write("Upload a transaction history")
        
        # CSV file uploader
        uploaded_file = st.file_uploader("Drag and drop file here or", type="csv", key="FileUpload", 
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

# Always display transactions in the main window (regardless of sidebar state)
# This is now completely independent of the expander state
if 'transactions' in st.session_state and st.session_state['transactions']:
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
else:
    st.write("No transactions to display.")