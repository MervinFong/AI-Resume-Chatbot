# pages/Logout.py

import streamlit as st
import time

# --- Setup Page Config ---
try:
    st.set_page_config(page_title="Logout | JobConnect", 
                       page_icon="assets/favicon.png",
                       layout="wide")
except st.errors.StreamlitAPIException:
    pass

# --- Main Logout Logic ---

def logout():
    st.title("🔓 Logout")

    if "email" in st.session_state:
        email = st.session_state.get("email", "User")
        st.success(f"✅ You have been logged out, {email}. Redirecting to login page...")
        
        # Clear session after showing the message
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        time.sleep(2)  # Pause to let user see success message
        
        # Redirect back to Login page
        st.switch_page("pages/Login.py")
    
    else:
        st.info("ℹ️ You are already logged out. Redirecting...")
        time.sleep(2)
        st.switch_page("pages/Login.py")

# --- Run the logout ---
logout()
