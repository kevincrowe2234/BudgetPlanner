import streamlit as st
import html

def render_transaction_table(transactions):
    """Render transactions in a HTML table with fixed column widths"""
    if not transactions:
        st.info("No transactions to display.")
        return

    # Build HTML table row by row
    header_row = "<tr><th class='col-date'>Date</th><th class='col-desc'>Description</th><th class='col-amount'>Amount</th></tr>"
    rows = []
    
    for t in transactions:
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
    st.write(f"Total transactions: {len(transactions)}")