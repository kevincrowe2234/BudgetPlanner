import streamlit as st
from utils.config import APP_TITLE, APP_LAYOUT, CUSTOM_CSS
import os
from utils.config import DATA_DIR

# Debug mode
DEBUG = True

# Page configuration
st.set_page_config(page_title=APP_TITLE, layout=APP_LAYOUT)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Ensure data directory exists
os.makedirs(os.path.join(DATA_DIR, "data"), exist_ok=True)

if DEBUG:
    st.write("Debug mode enabled")

# Wrap imports in try/except to catch errors
try:
    from utils.file_operations import load_session_data
    if DEBUG:
        st.write("✅ Successfully imported modules")
except Exception as e:
    st.error(f"❌ Error importing modules: {str(e)}")
    st.stop()

# Initialize session state if needed
try:
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = True
        load_session_data()
        if DEBUG:
            st.write("✅ Session data loaded")
            st.write(f"Keys in session_state: {list(st.session_state.keys())}")
except Exception as e:
    st.error(f"❌ Error loading session data: {str(e)}")

# App header
st.title(f"💰 {APP_TITLE}")

# Navigation using radio buttons instead of tabs
page = st.sidebar.radio("Navigation", ["Dashboard", "Income", "Expenses"], index=0)

# Clear main area before rendering the selected page
st.empty()

if page == "Dashboard":
    try:
        print("Displaying Dashboard")
        import importlib
        import pages.dashboard
        importlib.reload(pages.dashboard)
        pages.dashboard.render_dashboard()
        print("Dashboard rendered successfully")
    except Exception as e:
        import traceback
        print(f"ERROR IN DASHBOARD: {str(e)}")
        print(traceback.format_exc())
        st.error(f"Error in dashboard: {str(e)}")
        st.code(traceback.format_exc())
        
elif page == "Income":
    try:
        print("Displaying Income page")
        import importlib
        import pages.income
        importlib.reload(pages.income)
        pages.income.render_income_page()
        print("Income page rendered successfully")
    except Exception as e:
        import traceback
        print(f"ERROR IN INCOME TAB: {str(e)}")
        print(traceback.format_exc())
        st.error(f"Error in income tab: {str(e)}")
        st.code(traceback.format_exc())
        
elif page == "Expenses":
    try:
        print("Displaying Expenses page")
        import importlib
        import pages.expenses
        importlib.reload(pages.expenses)
        pages.expenses.render_expenses_page()
        print("Expenses page rendered successfully")
    except Exception as e:
        import traceback
        print(f"ERROR IN EXPENSES TAB: {str(e)}")
        print(traceback.format_exc())
        st.error(f"Error in expenses tab: {str(e)}")
        st.code(traceback.format_exc())

# Sidebar with configuration and data management
st.sidebar.header("Configuration")

# Add debug info
if DEBUG:
    st.sidebar.write("Debug Info:")
    st.sidebar.write(f"Working directory: {os.getcwd()}")
    st.sidebar.write(f"DATA_DIR: {DATA_DIR}")
    st.sidebar.write(f"DATA_DIR exists: {os.path.exists(DATA_DIR)}")
    data_dir = os.path.join(DATA_DIR, "data")
    st.sidebar.write(f"data/ exists: {os.path.exists(data_dir)}")

with st.sidebar:
    try:
        from components.sidebar import render_configuration, render_data_management, render_transaction_import
        if DEBUG:
            st.write("✅ Successfully imported sidebar components")
        render_configuration()
        if DEBUG:
            st.write("✅ Configuration rendered")
        render_data_management()
        if DEBUG:
            st.write("✅ Data management rendered")
        render_transaction_import()
        if DEBUG:
            st.write("✅ Transaction import rendered")
    except Exception as e:
        st.error(f"❌ Error rendering sidebar: {str(e)}")