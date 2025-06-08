import streamlit as st
from firebase_config import init_firebase
from sidebar import setup_sidebar

# --- Setup Page Config ---
try:
    st.set_page_config(
        page_title="Dashboard | JobConnect", 
        page_icon="assets/favicon.png",
        layout="wide")
except st.errors.StreamlitAPIException:
    pass

setup_sidebar()

# --- Initialize Firebase ---
db = init_firebase()

# --- Page Protection ---
if "user_uid" not in st.session_state or st.session_state.get("role") != "Candidate":
    st.error("🚫 Unauthorized access!")
    st.switch_page("pages/Login.py")
    st.stop()
    
# --- Main Candidate Dashboard Content ---
st.title("🏡 Welcome to Candidate Dashboard")
st.subheader(f"Hello, {st.session_state.get('email', 'Candidate')} 👋")
st.markdown("""
    <div style='background-color: #F0F8FF; padding: 20px; border-radius: 10px; margin-top: 20px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1);'>
        <h3 style='color: #2E86C1;'>🎯 What you can do here:</h3>
        <ul style='font-size: 16px; color: #555;'>
            <li>🔍 Browse and apply for job opportunities.</li>
            <li>📄 Upload your resume and receive job recommendations.</li>
            <li>🧠 Take the MBTI personality test to discover your strengths.</li>
            <li>💬 Chat with our AI Career Advisor for resume tips and career advice.</li>
            <li>📥 Save jobs you like and manage your applications easily.</li>
            <li>📩 Get support or report suspicious job posts.</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# --- Helpful quick shortcuts ---
st.divider()
st.subheader("🚀 Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🔍 Browse Jobs"):
        st.switch_page("pages/Job_Listings.py")

with col2:
    if st.button("📄 Upload Resume"):
        st.switch_page("pages/Resume_History.py")

with col3:
    if st.button("🧠 Take MBTI Test"):
        st.switch_page("pages/MBTI_Test_with_Firebase.py")

# Footer
st.markdown("---")
st.caption("🔔 Stay active — new jobs are posted daily. Your dream job is waiting! 🚀")
