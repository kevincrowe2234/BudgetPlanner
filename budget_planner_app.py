import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import plotly.graph_objects as go
import plotly.express as px
from dateutil.relativedelta import relativedelta
import os
import base64
from io import StringIO

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
</style>
""", unsafe_allow_html=True)

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

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.datetime.now()

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


# Functions for data manipulation
def get_dataframe_download_link(df, filename, text):
    """Generate a link to download the dataframe as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{text}</a>'
    return href


def save_data(income_df, expense_df, balance, filename):
    """Save all budget data to a CSV file"""
    income_df['Type'] = 'Income'
    expense_df['Type'] = 'Expense'

    # Combine dataframes
    combined_df = pd.concat([income_df, expense_df], ignore_index=True)

    # Add balance as a separate row or metadata
    metadata_df = pd.DataFrame({
        'Description': ['INITIAL_BALANCE'],
        'Amount': [balance],
        'Date': [datetime.datetime.now().strftime('%Y-%m-%d')],
        'Recurring': [False],
        'Frequency': ['None'],
        'Type': ['Metadata']
    })

    final_df = pd.concat([metadata_df, combined_df], ignore_index=True)
    return final_df


def load_data(uploaded_file):
    """Load budget data from a CSV file"""
    if uploaded_file is not None:
        try:
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
            income_df['Date'] = pd.to_datetime(income_df['Date'])
            expense_df['Date'] = pd.to_datetime(expense_df['Date'])

            # Convert boolean strings to actual booleans
            income_df['Recurring'] = income_df['Recurring'].astype(bool)
            expense_df['Recurring'] = expense_df['Recurring'].astype(bool)

            return income_df, expense_df, balance
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame(), pd.DataFrame(), 0.0
    return pd.DataFrame(), pd.DataFrame(), 0.0


def import_transactions(uploaded_file, date_col="Date", amount_col="Amount", desc_col="Description"):
    """Import past transactions from a CSV file"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Ensure required columns exist
            required_cols = [date_col, amount_col, desc_col]
            if not all(col in df.columns for col in required_cols):
                st.error(f"CSV must contain the following columns: {', '.join(required_cols)}")
                return pd.DataFrame(), pd.DataFrame()

            # Convert date strings to datetime objects
            df[date_col] = pd.to_datetime(df[date_col])

            # Separate income and expenses based on amount sign
            income_mask = df[amount_col] > 0

            income_df = df[income_mask].copy()
            expense_df = df[~income_mask].copy()
            expense_df[amount_col] = expense_df[amount_col].abs()  # Make expense amounts positive

            # Format for our application
            format_income = pd.DataFrame({
                'Description': income_df[desc_col],
                'Amount': income_df[amount_col],
                'Date': income_df[date_col],
                'Recurring': False,
                'Frequency': 'Monthly',  # Default assumption
                'Type': 'Income'
            })

            format_expense = pd.DataFrame({
                'Description': expense_df[desc_col],
                'Amount': expense_df[amount_col],
                'Date': expense_df[date_col],
                'Recurring': False,
                'Frequency': 'Monthly',  # Default assumption
                'Type': 'Expense'
            })

            return format_income, format_expense
        except Exception as e:
            st.error(f"Error importing transactions: {e}")
    return pd.DataFrame(), pd.DataFrame()


def prepare_dataframes_for_projection(income_df, expense_df):
    """Prepare dataframes for projection by ensuring correct data types"""
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

    if not income_copy.empty:
        # Ensure Date is datetime
        income_copy['Date'] = pd.to_datetime(income_copy['Date'])

        # Ensure Amount is float
        income_copy['Amount'] = income_copy['Amount'].astype(float)

        # Ensure Recurring is boolean
        income_copy['Recurring'] = income_copy['Recurring'].astype(bool)

    if not expense_copy.empty:
        # Ensure Date is datetime
        expense_copy['Date'] = pd.to_datetime(expense_copy['Date'])

        # Ensure Amount is float
        expense_copy['Amount'] = expense_copy['Amount'].astype(float)

        # Ensure Recurring is boolean
        expense_copy['Recurring'] = expense_copy['Recurring'].astype(bool)

    return income_copy, expense_copy


def generate_timeline(income_df, expense_df, initial_balance, months=18):
    """Generate a timeline of balance changes over the specified period"""
    today = datetime.datetime.now().date()
    end_date = today + relativedelta(months=months)

    # Create a date range for each day in the projection period
    date_range = pd.date_range(start=today, end=end_date, freq='D')
    timeline_df = pd.DataFrame(index=date_range)
    timeline_df['Income'] = 0.0
    timeline_df['Expense'] = 0.0
    timeline_df['Net'] = 0.0
    timeline_df['Balance'] = initial_balance

    # Process income transactions
    if not income_df.empty:
        for _, row in income_df.iterrows():
            amount = float(row['Amount'])

            # Convert date to datetime.date
            if isinstance(row['Date'], str):
                date = pd.to_datetime(row['Date']).date()
            elif isinstance(row['Date'], pd.Timestamp):
                date = row['Date'].date()
            else:
                date = row['Date']

            # Skip if the date is before our timeline
            # Handle NaT values gracefully
            if pd.isna(date) or (date is not None and date < today):
                continue

            # Handle recurring income
            if row['Recurring']:
                freq = row['Frequency']
                current_date = date

                while current_date <= end_date:
                    # Find the nearest date in the index (sometimes exact matches fail)
                    index_date = pd.Timestamp(current_date)
                    if index_date in timeline_df.index:
                        timeline_df.loc[index_date, 'Income'] += amount

                    # Move to the next occurrence based on frequency
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
                        break  # Unknown frequency
            else:
                # One-time income
                index_date = pd.Timestamp(date)
                if index_date in timeline_df.index:
                    timeline_df.loc[index_date, 'Income'] += amount

    # Process expense transactions
    if not expense_df.empty:
        for _, row in expense_df.iterrows():
            amount = float(row['Amount'])

            # Convert date to datetime.date
            if isinstance(row['Date'], str):
                date = pd.to_datetime(row['Date']).date()
            elif isinstance(row['Date'], pd.Timestamp):
                date = row['Date'].date()
            else:
                date = row['Date']

            # Skip if the date is before our timeline
            # Handle NaT values gracefully
            if pd.isna(date) or (date is not None and date < today):
                continue

            # Handle recurring expenses
            if row['Recurring']:
                freq = row['Frequency']
                current_date = date

                while current_date <= end_date:
                    # Find the nearest date in the index
                    index_date = pd.Timestamp(current_date)
                    if index_date in timeline_df.index:
                        timeline_df.loc[index_date, 'Expense'] += amount

                    # Move to the next occurrence based on frequency
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
                        break  # Unknown frequency
            else:
                # One-time expense
                index_date = pd.Timestamp(date)
                if index_date in timeline_df.index:
                    timeline_df.loc[index_date, 'Expense'] += amount

    # Calculate net and running balance
    timeline_df['Net'] = timeline_df['Income'] - timeline_df['Expense']

    # Calculate cumulative balance
    for i in range(1, len(timeline_df)):
        prev_date = timeline_df.index[i - 1]
        curr_date = timeline_df.index[i]
        timeline_df.loc[curr_date, 'Balance'] = timeline_df.loc[prev_date, 'Balance'] + timeline_df.loc[
            curr_date, 'Net']

    # Create monthly summary for easier visualization
    monthly_summary = timeline_df.resample('M').agg({
        'Income': 'sum',
        'Expense': 'sum',
        'Net': 'sum',
        'Balance': 'last'  # Take the last balance of the month
    })

    return timeline_df, monthly_summary


def plot_balance_over_time(timeline_df, monthly_df):
    """Create a plot of the bank balance over time"""
    # Daily chart with plotly
    fig1 = go.Figure()

    # Add daily balance line
    fig1.add_trace(
        go.Scatter(
            x=timeline_df.index,
            y=timeline_df['Balance'],
            name='Daily Balance',
            line=dict(color='#2196F3', width=2),
            hovertemplate='%{x}<br>Balance: $%{y:.2f}<extra></extra>',
        )
    )

    # Add zero line for reference
    fig1.add_shape(
        type="line",
        x0=timeline_df.index.min(),
        y0=0,
        x1=timeline_df.index.max(),
        y1=0,
        line=dict(color="red", width=1.5, dash="dash"),
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

    # Monthly chart with income, expenses and balance
    fig2 = go.Figure()

    # Add monthly balance line
    fig2.add_trace(
        go.Scatter(
            x=monthly_df.index,
            y=monthly_df['Balance'],
            name='End of Month Balance',
            line=dict(color='#2196F3', width=3),
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

    # Add zero line for reference
    fig2.add_shape(
        type="line",
        x0=monthly_df.index.min(),
        y0=0,
        x1=monthly_df.index.max(),
        y1=0,
        line=dict(color="red", width=1.5, dash="dash"),
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


# Main application interface
st.title("💰 Personal Budget Planner")

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

    if st.button("Save Budget Data"):
        saved_data = save_data(
            st.session_state.income_data,
            st.session_state.expense_data,
            st.session_state.bank_balance,
            save_filename
        )

        # Generate download link and automatically click it with JavaScript
        csv = saved_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()

        # Create a JavaScript to automatically trigger download
        download_js = f"""
        <script>
            function download(filename, text) {{
                var element = document.createElement('a');
                element.setAttribute('href', 'data:file/csv;base64,{b64}');
                element.setAttribute('download', "{save_filename}.csv");
                element.style.display = 'none';
                document.body.appendChild(element);
                element.click();
                document.body.removeChild(element);
            }}

            download("{save_filename}.csv", "data");
        </script>
        """

        # Display success message
        st.success("Budget data downloaded!")

        # Inject JavaScript to trigger the download
        st.components.v1.html(download_js, height=0)

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
            st.success("Budget data loaded successfully!")
        else:
            st.error("Failed to load budget data. Please check the file format.")

        # Import Transactions
import_option = st.sidebar.expander("Import Past Transactions")
with import_option:
    import_file = st.file_uploader("Upload a transaction history", type="csv")

    if import_file is not None:
        # Column mapping
        st.write("Map your CSV columns:")
        col1, col2, col3 = st.columns(3)
        with col1:
            date_col = st.text_input("Date Column", "Date")
        with col2:
            amount_col = st.text_input("Amount Column", "Amount")
        with col3:
            desc_col = st.text_input("Description Column", "Description")

        if st.button("Import Transactions"):
            income_from_import, expenses_from_import = import_transactions(
                import_file,
                date_col=date_col,
                amount_col=amount_col,
                desc_col=desc_col
            )

            # Ensure proper data types
            if not income_from_import.empty:
                income_from_import['Recurring'] = income_from_import['Recurring'].astype(bool)

            if not expenses_from_import.empty:
                expenses_from_import['Recurring'] = expenses_from_import['Recurring'].astype(bool)

            if not income_from_import.empty or not expenses_from_import.empty:
                # Let user decide how to handle imported data
                handling_option = st.radio(
                    "How would you like to handle the imported data?",
                    ("Add to existing data", "Replace existing data")
                )

                if handling_option == "Add to existing data":
                    st.session_state.income_data = pd.concat([st.session_state.income_data, income_from_import],
                                                             ignore_index=True)
                    st.session_state.expense_data = pd.concat([st.session_state.expense_data, expenses_from_import],
                                                              ignore_index=True)
                else:
                    st.session_state.income_data = income_from_import
                    st.session_state.expense_data = expenses_from_import

                st.success("Transactions imported successfully!")
                st.rerun()  # Force refresh after importing
            else:
                st.error("No valid transactions found in the file.")

            # Main content area with tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💸 Income", "💳 Expenses"])

with tab1:
    st.header("Budget Dashboard")

    # Debug information
    with st.expander("Debug Information"):
        st.subheader("Current Data State")

        # Show income data information
        st.write("Income Data:")
        if not st.session_state.income_data.empty:
            st.write(f"- Number of entries: {len(st.session_state.income_data)}")
            st.write(f"- Total amount: ${st.session_state.income_data['Amount'].sum():.2f}")
            st.write(f"- Data types: {st.session_state.income_data.dtypes}")
            st.dataframe(st.session_state.income_data)
        else:
            st.write("No income data available")

        # Show expense data information
        st.write("Expense Data:")
        if not st.session_state.expense_data.empty:
            st.write(f"- Number of entries: {len(st.session_state.expense_data)}")
            st.write(f"- Total amount: ${st.session_state.expense_data['Amount'].sum():.2f}")
            st.write(f"- Data types: {st.session_state.expense_data.dtypes}")
            st.dataframe(st.session_state.expense_data)
        else:
            st.write("No expense data available")

        # Show bank balance
        st.write(f"Current bank balance: ${st.session_state.bank_balance:.2f}")
        st.write(f"Projection months: {st.session_state.projection_months}")

        # Force refresh button
        if st.button("Force Refresh"):
            st.rerun()

    # Add a refresh button
    if st.button("Refresh Projections"):
        st.rerun()

    # Prepare data with consistent types
    income_prepared, expense_prepared = prepare_dataframes_for_projection(
        st.session_state.income_data,
        st.session_state.expense_data
    )

    # Generate and display the budget timeline with prepared data
    timeline_df, monthly_df = generate_timeline(
        income_prepared,
        expense_prepared,
        st.session_state.bank_balance,
        months=st.session_state.projection_months
    )

    # Check if we have data to display
    has_data = not income_prepared.empty or not expense_prepared.empty

    if has_data:
        # Check if balance goes negative at any point
        min_balance = timeline_df['Balance'].min()
        if min_balance < 0:
            st.error(f"⚠️ Warning: Your balance will go negative! Minimum balance: ${min_balance:.2f}")

            # Find when balance first goes negative
            negative_dates = timeline_df[timeline_df['Balance'] < 0]
            if not negative_dates.empty:
                first_negative = negative_dates.index[0]
                st.error(f"Balance first goes negative on {first_negative.strftime('%Y-%m-%d')}")
        else:
            st.success("✅ Your balance stays positive throughout the projection period!")

            # Create and display the visualizations
        daily_fig, monthly_fig = plot_balance_over_time(timeline_df, monthly_df)
        st.plotly_chart(daily_fig, use_container_width=True)
        st.plotly_chart(monthly_fig, use_container_width=True)

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Starting Balance", f"${st.session_state.bank_balance:.2f}")
        with col2:
            st.metric("Ending Balance", f"${timeline_df['Balance'].iloc[-1]:.2f}")
        with col3:
            st.metric("Total Income", f"${timeline_df['Income'].sum():.2f}")
        with col4:
            st.metric("Total Expenses", f"${timeline_df['Expense'].sum():.2f}")

        # Display the data tables
        with st.expander("Show Detailed Monthly Projections"):
            st.dataframe(monthly_df.reset_index().rename(columns={'index': 'Date'}), use_container_width=True)
    else:
        st.info("Add some income and expenses to see your budget projection!")

with tab2:
    st.header("Income")

    # Form for adding new income
    with st.form("add_income_form"):
        st.subheader("Add New Income")

        col1, col2 = st.columns(2)
        with col1:
            income_desc = st.text_input("Description", key="income_desc")
            income_amount = st.number_input("Amount ($)", min_value=0.0, step=10.0, key="income_amount")
        with col2:
            income_date = st.date_input("Date", value=datetime.datetime.now(), key="income_date")
            income_recurring = st.checkbox("Recurring Income", key="income_recurring")

        if income_recurring:
            income_frequency = st.selectbox(
                "Frequency",
                ["Daily", "Weekly", "Biweekly", "Monthly", "Quarterly", "Annually"],
                index=3,  # Monthly as default
                key="income_frequency"
            )
        else:
            income_frequency = "None"

        submit_income = st.form_submit_button("Add Income")
        if submit_income:
            if income_desc and income_amount > 0:
                # Create a DataFrame with the new income
                new_income = pd.DataFrame({
                    'Description': [income_desc],
                    'Amount': [float(income_amount)],  # Ensure this is a float
                    'Date': [income_date],  # Use the date object directly
                    'Recurring': [bool(income_recurring)],  # Ensure this is a boolean
                    'Frequency': [income_frequency],
                    'Type': ['Income']
                })

                # Update timestamp for dashboard refresh
                st.session_state.last_update = datetime.datetime.now()

                # Append to existing income data
                st.session_state.income_data = pd.concat([st.session_state.income_data, new_income], ignore_index=True)

                # Ensure consistent data types
                if not st.session_state.income_data.empty:
                    # Convert dates to datetime
                    st.session_state.income_data['Date'] = pd.to_datetime(st.session_state.income_data['Date'])
                    # Convert recurring to boolean
                    st.session_state.income_data['Recurring'] = st.session_state.income_data['Recurring'].astype(bool)
                    # Ensure amounts are float
                    st.session_state.income_data['Amount'] = st.session_state.income_data['Amount'].astype(float)

                st.success(f"Added {income_desc} to income!")

                # Force a rerun of the app to update all components
                st.rerun()
            else:
                st.error("Please provide a description and amount greater than zero.")

                # Display current income data
    if not st.session_state.income_data.empty:
        st.subheader("Current Income")

        # Convert data types before displaying editor
        # First convert dates to datetime
        st.session_state.income_data['Date'] = pd.to_datetime(st.session_state.income_data['Date'])
        # Then convert recurring to boolean
        st.session_state.income_data['Recurring'] = st.session_state.income_data['Recurring'].astype(bool)
        # Ensure amounts are float
        st.session_state.income_data['Amount'] = st.session_state.income_data['Amount'].astype(float)

        # Display as a table with option to delete rows
        edited_income = st.data_editor(
            st.session_state.income_data,
            column_config={
                "Description": st.column_config.TextColumn("Description"),
                "Amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
                "Date": st.column_config.DateColumn("Date"),
                "Recurring": st.column_config.CheckboxColumn("Recurring"),
                "Frequency": st.column_config.SelectboxColumn(
                    "Frequency",
                    options=["None", "Daily", "Weekly", "Biweekly", "Monthly", "Quarterly", "Annually"]
                ),
                "Type": st.column_config.TextColumn("Type", disabled=True)
            },
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic"
        )

        # Update the session state with edited data
        if not edited_income.equals(st.session_state.income_data):
            st.session_state.income_data = edited_income
            st.rerun()  # Force refresh when income data is edited
    else:
        st.info("No income entries yet. Add some income to get started!")

with tab3:
    st.header("Expenses")

    # Form for adding new expense
    with st.form("add_expense_form"):
        st.subheader("Add New Expense")

        col1, col2 = st.columns(2)
        with col1:
            expense_desc = st.text_input("Description", key="expense_desc")
            expense_amount = st.number_input("Amount ($)", min_value=0.0, step=10.0, key="expense_amount")
        with col2:
            expense_date = st.date_input("Date", value=datetime.datetime.now(), key="expense_date")
            expense_recurring = st.checkbox("Recurring Expense", key="expense_recurring")

        if expense_recurring:
            expense_frequency = st.selectbox(
                "Frequency",
                ["Daily", "Weekly", "Biweekly", "Monthly", "Quarterly", "Annually"],
                index=3,  # Monthly as default
                key="expense_frequency"
            )
        else:
            expense_frequency = "None"

        submit_expense = st.form_submit_button("Add Expense")
        if submit_expense:
            if expense_desc and expense_amount > 0:
                # Create a DataFrame with the new expense
                new_expense = pd.DataFrame({
                    'Description': [expense_desc],
                    'Amount': [float(expense_amount)],  # Ensure this is a float
                    'Date': [expense_date],  # Use the date object directly
                    'Recurring': [bool(expense_recurring)],  # Ensure this is a boolean
                    'Frequency': [expense_frequency],
                    'Type': ['Expense']
                })

                # Update timestamp for dashboard refresh
                st.session_state.last_update = datetime.datetime.now()

                # Append to existing expense data
                st.session_state.expense_data = pd.concat([st.session_state.expense_data, new_expense],
                                                          ignore_index=True)

                # Ensure consistent data types
                if not st.session_state.expense_data.empty:
                    # Convert dates to datetime
                    st.session_state.expense_data['Date'] = pd.to_datetime(st.session_state.expense_data['Date'])
                    # Convert recurring to boolean
                    st.session_state.expense_data['Recurring'] = st.session_state.expense_data['Recurring'].astype(bool)
                    # Ensure amounts are float
                    st.session_state.expense_data['Amount'] = st.session_state.expense_data['Amount'].astype(float)

                st.success(f"Added {expense_desc} to expenses!")

                # Force a rerun of the app to update all components
                st.rerun()
            else:
                st.error("Please provide a description and amount greater than zero.")

                # Display current expense data
    if not st.session_state.expense_data.empty:
        st.subheader("Current Expenses")

        # Convert data types before displaying editor
        # First convert dates to datetime
        st.session_state.expense_data['Date'] = pd.to_datetime(st.session_state.expense_data['Date'])
        # Then convert recurring to boolean
        st.session_state.expense_data['Recurring'] = st.session_state.expense_data['Recurring'].astype(bool)
        # Ensure amounts are float
        st.session_state.expense_data['Amount'] = st.session_state.expense_data['Amount'].astype(float)

        # Display as a table with option to delete rows
        edited_expense = st.data_editor(
            st.session_state.expense_data,
            column_config={
                "Description": st.column_config.TextColumn("Description"),
                "Amount": st.column_config.NumberColumn("Amount ($)", format="$%.2f"),
                "Date": st.column_config.DateColumn("Date"),
                "Recurring": st.column_config.CheckboxColumn("Recurring"),
                "Frequency": st.column_config.SelectboxColumn(
                    "Frequency",
                    options=["None", "Daily", "Weekly", "Biweekly", "Monthly", "Quarterly", "Annually"]
                ),
                "Type": st.column_config.TextColumn("Type", disabled=True)
            },
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic"
        )

        # Update the session state with edited data
        if not edited_expense.equals(st.session_state.expense_data):
            st.session_state.expense_data = edited_expense
            st.rerun()  # Force refresh when expense data is edited
    else:
        st.info("No expense entries yet. Add some expenses to get started!")

# Footer
st.markdown("""
---
### How to Use This Budget Planner

1. **Initial Setup**: Set your initial bank balance and projection period in the sidebar.
2. **Add Income & Expenses**: Use the Income and Expenses tabs to add your financial data.
3. **Recurring Transactions**: For regular income or expenses, check the "Recurring" option and select the frequency.
4. **View Projections**: The Dashboard tab shows your projected balance over time.
5. **Save & Load**: Use the sidebar options to save your budget or load a previously saved one.
6. **Import Transactions**: Import past transactions from a CSV file to create a future budget based on your spending history.
7. **Debug Information**: If you encounter issues, check the Debug Information in the Dashboard tab.
8. **Refresh Projections**: Use the Refresh Projections button if the charts don't update automatically.

### CSV File Formats

**Saving/Loading Budget Data:**
- The app saves/loads data as a CSV file with headers.
- Headers include: Description, Amount, Date, Recurring, Frequency, Type
- Date format: YYYY-MM-DD
- Recurring: True/False
- Frequency: None, Daily, Weekly, Biweekly, Monthly, Quarterly, Annually
- Type: Income, Expense, Metadata

**Importing Transaction History:**
- Your CSV should have headers.
- Required columns: Date, Amount, Description (customize names in the import section)
- Date column should contain dates in a standard format (YYYY-MM-DD is preferred)
- Amount column: positive numbers for income, negative numbers for expenses
- Additional columns are ignored

This planner helps you ensure your bank balance remains positive throughout your projection period!
""")