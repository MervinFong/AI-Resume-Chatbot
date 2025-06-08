import streamlit as st
from firebase_config import init_firebase
from firebase_admin import auth
from sidebar import setup_sidebar

# --- Initialize Firebase ---
db = init_firebase()

# --- Page Config ---
try:
    st.set_page_config(page_title="Admin Job Approval", layout="wide")
except st.errors.StreamlitAPIException:
    pass

setup_sidebar()

# --- Protect Admin Access ---
if "user_uid" not in st.session_state or "role" not in st.session_state:
    st.warning("🚫 You must be logged in to access this page.")
    st.stop()

if st.session_state["role"] != "Admin":
    st.warning("🚫 You do not have admin rights to access this page.")
    st.stop()

# --- Main Title ---
st.title("🛡️ Admin - Job Listings Approval")

st.markdown("""
    <style>
    .job-card {
        background-color: #FAFAFA;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        padding: 20px;
        margin-bottom: 20px;
    }
    .job-card:hover {
        transform: scale(1.01);
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    .job-title {
        font-size: 22px;
        font-weight: bold;
        color: #2C3E50;
    }
    .company-name {
        font-size: 18px;
        font-weight: 500;
        color: #555;
        margin-bottom: 10px;
    }
    .job-details {
        font-size: 14px;
        color: #666;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Query Unapproved Jobs ---
unapproved_jobs = db.collection("job_listings").where("approved", "==", False).stream()

job_found = False

for job in unapproved_jobs:
    job_data = job.to_dict()
    job_found = True

    with st.container():
        st.markdown(f"""
        <div class="job-card">
            <div class="job-title">{job_data.get('job_title', 'N/A')}</div>
            <div class="company-name">{job_data.get('company_name', 'N/A')}</div>
            <div class="job-details">
                📍 <b>Location:</b> {job_data.get('location', 'N/A')}<br>
                📧 <b>Email:</b> {job_data.get('company_email', 'N/A')}<br>
                📞 <b>Phone:</b> {job_data.get('company_phone', 'N/A')}<br>
                💰 <b>Salary:</b> {job_data.get('salary', 'Not Provided')}<br>
                🗓️ <b>Date Posted:</b> {job_data.get('date_posted', 'N/A')}<br><br>
                <b>Description:</b> {job_data.get('job_description', 'N/A')}
            </div>
        """, unsafe_allow_html=True)

        # --- Approve Button ---
        if st.button(f"✅ Approve Job: {job_data.get('job_title', 'N/A')}", key=f"approve_{job.id}"):
            db.collection("job_listings").document(job.id).update({"approved": True})
            st.success(f"✅ Job listing '{job_data.get('job_title', 'N/A')}' has been approved.")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# --- No unapproved jobs ---
if not job_found:
    st.info("🎉 No pending job listings to approve.")
