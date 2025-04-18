# 💰 Personal Budget Planner

A Streamlit application for planning and visualizing your personal finances.

## Features

- **Track Income & Expenses**: Add one-time or recurring financial transactions
- **Visual Projections**: See your future balance with interactive charts
- **Savings Tracking**: Monitor your financial health with detailed projections
- **Import & Export**: Save, load, and import financial data
- **Responsive Design**: Works across different devices

## Screenshots

![Dashboard Screenshot](https://via.placeholder.com/800x450.png?text=Budget+Planner+Dashboard)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/budget-planner.git
   cd budget-planner
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run budget_planner_app.py
   ```

## Usage Guide

### Initial Setup

1. Set your initial bank balance in the sidebar
2. Set your desired projection period (1-60 months)

### Adding Income

1. Navigate to the Income tab
2. Fill in the description, amount, and date
3. Check "Recurring Income" if this is a repeating transaction
4. Select the frequency for recurring income
5. Click "Add Income"

### Adding Expenses

1. Navigate to the Expenses tab
2. Fill in the description, amount, and date
3. Check "Recurring Expense" if this is a repeating transaction
4. Select the frequency for recurring expenses
5. Click "Add Expense"

### Viewing Projections

1. Navigate to the Dashboard tab
2. View your daily balance projection chart
3. View monthly income, expenses, and balance chart
4. Check summary statistics
5. Expand "Show Detailed Monthly Projections" for a detailed breakdown

### Managing Data

- **Save Data**: Save your current budget to a CSV file
- **Load Data**: Upload a previously saved budget file
- **Import Transactions**: Import historical transactions from a CSV

## CSV File Formats

### Budget Data (Save/Load)

- Headers: Description, Amount, Date, Recurring, Frequency, Type
- Date format: YYYY-MM-DD
- Recurring: True/False
- Frequency: None, Daily, Weekly, Biweekly, Monthly, Quarterly, Annually
- Type: Income, Expense, Metadata

### Transaction Import

- Required headers: Date, Amount, Description
- Date column: Standard date format (YYYY-MM-DD preferred)
- Amount column: Positive numbers for income, negative for expenses

## Troubleshooting

If you encounter issues:

1. Check the Debug Information in the Dashboard tab
2. Use the "Force Refresh" button to refresh the application state
3. Ensure all dates are in the correct format
4. Verify that your CSV files match the expected format

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
