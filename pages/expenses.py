# Ultra-simple test version
import streamlit as st
import pandas as pd

def render_expenses_page():
    st.title("Expenses Page")
    st.write("This is the expenses page")
    st.success("Expenses page loaded successfully!")
    
    with st.form("expenses_test_form"):
        name = st.text_input("Test input")
        submit = st.form_submit_button("Submit")
    
    if submit and name:
        st.write(f"You entered: {name}")