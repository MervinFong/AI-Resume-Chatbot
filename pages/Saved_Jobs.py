import streamlit as st
from firebase_config import init_firebase
from datetime import datetime
from sidebar import setup_sidebar

# --- Initialize Firestore ---
db = init_firebase()

# --- Setup Page Config ---
try:
    st.set_page_config(
        page_title="Saved Jobs | JobConnect",
        page_icon="assets/favicon.png",
        layout="wide"
    )
except st.errors.StreamlitAPIException:
    pass

setup_sidebar()

# --- Page Protection ---
if "user_uid" not in st.session_state or st.session_state.get("role") != "Candidate":
    st.error("🚫 Unauthorized access!")
    st.switch_page("pages/Login.py")
    st.stop()

# --- Main Page ---
st.title("📥 Your Saved Favorite Jobs")

# --- Check Login ---
if "user_uid" not in st.session_state:
    st.error("🚫 You must log in to view saved jobs.")
    st.stop()

user_uid = st.session_state["user_uid"]

# --- Fetch Saved Jobs ---
saved_jobs_query = db.collection("saved_jobs").where("user_uid", "==", user_uid).stream()
saved_jobs = list(saved_jobs_query)

# --- Apply Function ---
def apply_saved_job(job):
    applications_ref = db.collection("applied_jobs")
    existing_application = applications_ref\
        .where("user_uid", "==", user_uid)\
        .where("job_title", "==", job.get('job_title'))\
        .where("company_name", "==", job.get('company_name'))\
        .stream()

    if any(existing_application):
        st.warning("⚠️ You already applied for this job!")
    else:
        applications_ref.add({
            "user_uid": user_uid,
            "user_email": st.session_state.get("email"),
            "job_title": job.get('job_title'),
            "company_name": job.get('company_name'),
            "location": job.get('location'),
            "timestamp": datetime.utcnow(),
            "status": "Applied",
            "resume_file_base64": st.session_state.get("resume_file_base64", ""),
            "resume_file_name": st.session_state.get("resume_file_name", "resume.pdf")
        })
        st.success(f"✅ Successfully applied for {job.get('job_title')}!")

# --- Remove Function ---
def remove_saved_job(doc_id):
    db.collection("saved_jobs").document(doc_id).delete()
    st.success("🗑️ Removed from saved jobs!")
    st.rerun()

# --- Show Saved Jobs ---
if not saved_jobs:
    st.info("You haven't saved any jobs yet. Go to Job Listings and save some! 🔖")
else:
    for idx, doc in enumerate(saved_jobs):
        job = doc.to_dict()

        with st.container():
            st.markdown(f"""
                <div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);'>
                    <h4 style='color: #333;'>{job.get('job_title', 'Unknown Position')}</h4>
                        <p style='color: #666;'><b>Company:</b> {job.get('company_name', 'Unknown Company')}</p>
                        <p style='color: #555;'><b>Location:</b> {job.get('location', 'Not specified')}</p>
                        <p style='color: #555;'><b>Category:</b> {job.get('category', 'Not specified')}</p>
                        <p style='color: #555;'><b>Description:</b><br>{job.get('job_description', 'No description available')[:300]}...</p>
                </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"📩 Apply Now", key=f"apply_saved_{idx}"):
                    apply_saved_job(job)

            with col2:
                if st.button(f"🗑️ Remove", key=f"remove_saved_{idx}"):
                    remove_saved_job(doc.id)
