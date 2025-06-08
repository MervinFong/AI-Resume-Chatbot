import streamlit as st
from sidebar import setup_sidebar  

# --- Setup Page Config ---
try:
    st.set_page_config(
        page_title="Resume Maker | JobConnect", 
        page_icon="assets/favicon.png",
        layout="wide")
except st.errors.StreamlitAPIException:
    pass

setup_sidebar()

# --- Page Protection ---
if "user_uid" not in st.session_state or st.session_state.get("role") != "Candidate":
    st.error("🚫 Unauthorized access!")
    st.switch_page("pages/Login.py")
    st.stop()

# --- Page Content ---
st.title("📝 Resume Maker")

st.markdown("""
<div style="background-color: #f0f2f6; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);">
    <h3 style="color: #333;">Welcome to the Resume Maker Section!</h3>
    <p style="font-size: 16px;">Here, you can create a professional, ATS-friendly resume using trusted online builders. 
    An ATS (Applicant Tracking System) friendly resume increases your chances of getting noticed by recruiters. 🚀</p>
    <hr style="margin: 10px 0;">
    <p style="font-size: 16px;">Recommended ATS Resume Builders:</p>
    <ul style="font-size: 16px;">
        <li><a href="https://novoresume.com/resume-templates/ats" target="_blank">Novoresume - ATS Optimized Templates</a></li>
        <li><a href="https://zety.com/resume-builder" target="_blank">Zety - Professional Resume Builder</a></li>
        <li><a href="https://resumegenius.com/resume-builder" target="_blank">Resume Genius - Fast Resume Builder</a></li>
    </ul>
    <hr style="margin: 10px 0;">
    <p style="font-size: 16px;">🛠️ After creating your resume, you can return to our platform to upload and get AI-powered feedback!</p>
</div>
""", unsafe_allow_html=True)

# --- Simple Footer ---
st.markdown("""
<br>
<center style="color: #888;">Need help writing your resume? Contact our support team for personalized assistance! ✉️</center>
""", unsafe_allow_html=True)
