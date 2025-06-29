import os

# App configuration
APP_TITLE = "Personal Budget Planner"
APP_VERSION = "0.028"
APP_LAYOUT = "wide"

# File paths
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGET_DATA_FILE = os.path.join(DATA_DIR, "data", "budget_data.json")
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "data", "transactions.json")

# Default values for new users
DEFAULT_PROJECTION_MONTHS = 18
DEFAULT_BANK_BALANCE = 0.0

# Create empty dataframe structure for budget data
EMPTY_BUDGET_DF_COLS = [
    'Description', 'Amount', 'Date', 'Recurring', 'Frequency', 'Type'
]

# Date formats to try when parsing dates
DATE_FORMATS = [
    "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y",
    "%b %d %Y", "%d %b %Y", "%Y %b %d",
    "%m/%d/%y", "%d/%m/%y", "%y-%m-%d"
]

# Custom CSS for styling
CUSTOM_CSS = """
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
"""