import streamlit as st
import html

# Sample transactions for testing
transactions = [
    {"Date": "2026-06-20", "Description": "Internet Withdrawal 20Jun19:06 Fiona For Sylvia Birthda", "Amount": "-100.00"},
    {"Date": "2026-06-20", "Description": "Eftpos Debit 20Jun07:52 Flickinv-000784931", "Amount": "-28.99"},
    {"Date": "2026-06-19", "Description": "Visa Purchase 16Jun Amazon Au Sydney So", "Amount": "-21.60"},
]

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

# Build HTML table row by row
header_row = "<tr><th class='col-date'>Date</th><th class='col-desc'>Description</th><th class='col-amount'>Amount</th></tr>"
rows = []
for t in transactions:
    # Use html.escape to prevent any HTML injection issues
    date = html.escape(t["Date"])
    desc = html.escape(t["Description"])
    amount = html.escape(t["Amount"])
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