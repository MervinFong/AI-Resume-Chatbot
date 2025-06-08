import streamlit as st
from firebase_config import init_firebase
from datetime import datetime
from sidebar import setup_sidebar

# --- Initialize Firebase ---
db = init_firebase()

# --- Setup Page Config ---
try:
    st.set_page_config(page_title="Manage Jobs", 
                       page_icon="assets/favicon.png",
                       layout="wide")
except st.errors.StreamlitAPIException:
    pass

setup_sidebar()

# --- Protect Admin Access ---
if "user_uid" not in st.session_state or "role" not in st.session_state:
    st.warning("🚫 You must be logged in to access this page.")
    st.stop()

# --- Protect Admin Access ---
if st.session_state.get("role") != "Admin":
    st.error("🚫 Only admins can access this page.")
    st.stop()

# --- Main Content ---
st.title("🛡️ Admin - Manage All Job Listings")

st.info("Here you can view, activate, or deactivate job postings.")

# --- Fetch all Jobs ---
all_jobs = db.collection("job_listings").stream()
jobs = [{**doc.to_dict(), "doc_id": doc.id} for doc in all_jobs]

if not jobs:
    st.warning("⚠️ No job listings found.")
else:
    for job in jobs:
        with st.container():
            active_status = job.get("active", True)
            approved_status = job.get("approved", False)

            st.markdown(f"""
            <div style='background-color: #FAFAFA; padding: 15px; border-radius: 10px; margin-bottom: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);'>
                <h5 style="color: #2C3E50;">{job.get('job_title', 'Unknown')} at {job.get('company_name', 'Unknown Company')}</h5>
                📍 <b>Location:</b> {job.get('location', 'N/A')}<br>
                📂 <b>Category:</b> {job.get('category', 'Others')}<br>
                🗓️ <b>Date Posted:</b> {job.get('date_posted', 'Unknown')}<br>
                ✅ <b>Approved:</b> {"Yes" if approved_status else "No"}<br>
                🚀 <b>Active:</b> {"Yes" if active_status else "No"}
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                if active_status:
                    if st.button(f"🔒 Deactivate Job", key=f"deactivate_{job['doc_id']}"):
                        db.collection("job_listings").document(job["doc_id"]).update({"active": False})
                        st.success(f"✅ Deactivated {job.get('job_title', 'Unknown')}")
                        st.rerun()
                else:
                    if st.button(f"🚀 Activate Job", key=f"activate_{job['doc_id']}"):
                        db.collection("job_listings").document(job["doc_id"]).update({"active": True})
                        st.success(f"✅ Activated {job.get('job_title', 'Unknown')}")
                        st.rerun()

            with col2:
                if st.button(f"🗑️ Delete Job", key=f"delete_{job['doc_id']}"):
                    db.collection("job_listings").document(job["doc_id"]).delete()
                    st.success(f"✅ Deleted job {job.get('job_title', 'Unknown')}")
                    st.rerun()

        st.markdown("---")
