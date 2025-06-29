# Ultra-simple test version
import streamlit as st
import pandas as pd

def render_income_page():
    st.title("Income Page")
    st.write("This is the income page")
    st.success("Income page loaded successfully!")
    
    with st.form("income_test_form"):
        name = st.text_input("Test input")
        submit = st.form_submit_button("Submit")
    
    if submit and name:
        st.write(f"You entered: {name}")