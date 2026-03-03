"""
Simple password-based authentication for Streamlit.
On Community Cloud, set APP_PASSWORD in the Secrets panel.
"""

import streamlit as st
import hmac
from config import APP_PASSWORD


def check_password() -> bool:
    """Show a login form and return True only after the correct password is entered."""

    if st.session_state.get("authenticated"):
        return True

    st.title("🔐 AI Travel Planner")
    st.caption("Enter the access password to continue.")

    with st.form("login_form"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        if hmac.compare_digest(password, APP_PASSWORD):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")

    return False
