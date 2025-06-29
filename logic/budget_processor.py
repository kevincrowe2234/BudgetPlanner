import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def prepare_dataframes_for_projection(income_df, expense_df):
    """Prepare income and expense dataframes for projection calculations"""
    # Make copies to avoid modifying the original dataframes
    income_clean = income_df.copy() if not income_df.empty else pd.DataFrame({
        'Description': [], 'Amount': [], 'Date': [], 'Recurring': [], 'Frequency': []
    })
    
    expense_clean = expense_df.copy() if not expense_df.empty else pd.DataFrame({
        'Description': [], 'Amount': [], 'Date': [], 'Recurring': [], 'Frequency': []
    })
    
    # Ensure dates are datetime objects
    if not income_clean.empty:
        income_clean['Date'] = pd.to_datetime(income_clean['Date'])
    
    if not expense_clean.empty:
        expense_clean['Date'] = pd.to_datetime(expense_clean['Date'])
        
    return income_clean, expense_clean

def generate_timeline(income_df, expense_df, starting_balance, months=18):
    """Generate timeline with daily and monthly balance projections"""
    # Generate date range
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + relativedelta(months=months)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create timeline dataframe
    timeline_df = pd.DataFrame(index=date_range)
    timeline_df.index.name = 'Date'
    timeline_df['Income'] = 0.0
    timeline_df['Expense'] = 0.0
    timeline_df['Net'] = 0.0
    timeline_df['Balance'] = starting_balance
    
    # Process income entries
    if not income_df.empty:
        for _, income in income_df.iterrows():
            process_transaction(timeline_df, income, is_income=True)
    
    # Process expense entries
    if not expense_df.empty:
        for _, expense in expense_df.iterrows():
            process_transaction(timeline_df, expense, is_income=False)
    
    # Calculate cumulative balance
    timeline_df['Net'] = timeline_df['Income'] - timeline_df['Expense']
    timeline_df['Balance'] = starting_balance + timeline_df['Net'].cumsum()
    
    # Create monthly summary
    monthly_df = create_monthly_summary(timeline_df)
    
    return timeline_df, monthly_df

def process_transaction(timeline_df, transaction, is_income=True):
    """Process a single transaction (income or expense) into the timeline"""
    col = 'Income' if is_income else 'Expense'
    date = pd.to_datetime(transaction['Date'])
    amount = float(transaction['Amount'])
    
    # Skip if date is outside of our timeline
    if date < timeline_df.index.min() or date > timeline_df.index.max():
        return
    
    # Add the initial transaction
    if date in timeline_df.index:
        timeline_df.at[date, col] += amount
    
    # If recurring, add future occurrences
    if transaction['Recurring']:
        next_date = get_next_occurrence(date, transaction['Frequency'])
        while next_date is not None and next_date <= timeline_df.index.max():
            if next_date in timeline_df.index:
                timeline_df.at[next_date, col] += amount
            next_date = get_next_occurrence(next_date, transaction['Frequency'])

def get_next_occurrence(date, frequency):
    """Calculate the next occurrence of a recurring transaction"""
    if frequency == 'Daily':
        return date + timedelta(days=1)
    elif frequency == 'Weekly':
        return date + timedelta(days=7)
    elif frequency == 'Biweekly':
        return date + timedelta(days=14)
    elif frequency == 'Monthly':
        return date + relativedelta(months=1)
    elif frequency == 'Quarterly':
        return date + relativedelta(months=3)
    elif frequency == 'Annually':
        return date + relativedelta(years=1)
    else:
        return None

def create_monthly_summary(timeline_df):
    """Create monthly summary of financial data"""
    # Resample to monthly frequency
    monthly_df = timeline_df.resample('MS').agg({
        'Income': 'sum',
        'Expense': 'sum',
        'Net': 'sum'
    })
    
    # Calculate month-end balances - fix the deprecation warning here
    monthly_df['Balance'] = timeline_df.resample('ME')['Balance'].last().values
    
    # Format the date index for better readability
    monthly_df.index = monthly_df.index.strftime('%b %Y')
    
    return monthly_df