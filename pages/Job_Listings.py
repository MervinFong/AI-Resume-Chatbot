import streamlit as st
from firebase_config import init_firebase
from math import ceil
from datetime import datetime
from sidebar import setup_sidebar

# --- Initialize Firestore ---
db = init_firebase()

# --- Setup Page Config ---
try:
    st.set_page_config(
        page_title="Job Listings | JobConnect", 
        page_icon="assets/favicon.png",
        layout="wide")
except st.errors.StreamlitAPIException:
    pass

setup_sidebar()

# --- Main Page Content ---
st.title("💼 Browse Job Listings")
st.markdown("Explore exciting job opportunities below. Apply or save jobs you like! 🚀")

# --- Search and Filter Section ---
st.subheader("🔍 Search and Filter Jobs")

col1, col2 = st.columns(2)

with col1:
    category_filter = st.selectbox(
        "Filter by Category",
        options=["All", "Engineering", "Information Technology (IT)", "Marketing", "Sales", "Finance",
                 "Human Resources (HR)", "Healthcare", "Education", "Operations", "Others"]
    )

with col2:
    title_search = st.text_input("Search by Job Title")

# --- Fetch and Filter Jobs ---
all_jobs = []
approved_jobs_ref = db.collection("job_listings").where("approved", "==", True).stream()

for job in approved_jobs_ref:
    job_data = job.to_dict()
    job_data['doc_id'] = job.id
    matches_category = (category_filter == "All") or (job_data.get("category") == category_filter)
    matches_title = (not title_search) or (title_search.lower() in job_data.get("job_title", "").lower())
    if matches_category and matches_title:
        all_jobs.append(job_data)

# --- Pagination ---
jobs_per_page = 6
total_pages = ceil(len(all_jobs) / jobs_per_page)

if "job_page" not in st.session_state:
    st.session_state.job_page = 0

start = st.session_state.job_page * jobs_per_page
end = start + jobs_per_page
current_jobs = all_jobs[start:end]

# --- Apply Function ---
def apply_for_job(job):
    applications_ref = db.collection("applied_jobs")
    query = applications_ref.where("user_uid", "==", st.session_state["user_uid"]).where("job_id", "==", job["doc_id"]).stream()
    if any(query):
        st.warning("⚠️ You already applied for this job.")
        return

    # Save application with recruiter_email field
    applications_ref.add({
        "user_uid": st.session_state["user_uid"],
        "user_email": st.session_state.get("email"),
        "job_id": job["doc_id"],
        "company_email": job.get("company_email"),
        "company_name": job.get("company_name"),
        "job_title": job.get("job_title"),
        "recruiter_email": job.get("poster_email"),   # ✅ Add this line
        "timestamp": datetime.utcnow(),
        "status": "Applied",
        "resume_file_base64": st.session_state.get("resume_file_base64", ""),
        "resume_file_name": st.session_state.get("resume_file_name", "resume.pdf")
    })

    st.success(f"✅ Successfully applied for {job.get('job_title', 'the job')}!")

def save_job(job):
    saved_jobs_ref = db.collection("saved_jobs")

    # Check if already saved
    existing_saved = saved_jobs_ref.where("user_uid", "==", st.session_state["user_uid"])\
                                   .where("job_id", "==", job['doc_id'])\
                                   .stream()

    if any(existing_saved):
        st.warning("⚠️ You already saved this job!")
        return

    # Save full details into Firestore
    saved_jobs_ref.add({
        "user_uid": st.session_state["user_uid"],
        "user_email": st.session_state.get("email"),
        "job_id": job['doc_id'],
        "job_title": job.get("job_title", "Unknown Position"),
        "company_name": job.get("company_name", "Unknown Company"),
        "location": job.get("location", "Not specified"),
        "category": job.get("category", "Not specified"),
        "job_description": job.get("job_description", "No description available"),
        "timestamp": datetime.utcnow()
    })

    st.success(f"✅ Saved {job.get('job_title', 'the job')} successfully!")

# --- Display Jobs 2 Per Row ---
if current_jobs:
    for i in range(0, len(current_jobs), 2):
        cols = st.columns(2)
        for idx, col in enumerate(cols):
            if i + idx < len(current_jobs):
                job = current_jobs[i + idx]
                with col:
                    st.markdown(f"""
                    <div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);'>
                        <h4 style='color: #333;'>{job.get('job_title', 'Unknown Position')}</h4>
                            📂 <b>Category:</b> {job.get('category', 'Not specified')}<br>
                            📍 <b>Location:</b> {job.get('location', 'Not specified')}<br>
                            📧 <b>Email:</b> {job.get('company_email', 'Not provided')}<br>
                            📞 <b>Phone:</b> {job.get('company_phone', 'Not provided')}<br>
                            <b>Description:</b><br>{job.get('job_description', 'No description available')}
                    </div>
                    """, unsafe_allow_html=True)

                    button_col1, button_col2 = st.columns(2)
                    with button_col1:
                        if st.button(f"📩 Apply Now", key=f"apply_{job['doc_id']}"):
                            if not st.session_state.get("resume_file_base64"):
                                st.error("🚫 Please upload your resume first!")
                            else:
                                apply_for_job(job)

                    with button_col2:
                        if st.button(f"💾 Save Job", key=f"save_{job['doc_id']}"):
                            save_job(job)
else:
    st.warning("⚠️ No matching job listings found.")

# --- Pagination Controls ---
st.divider()
col_prev, col_page, col_next = st.columns([1, 2, 1])

with col_prev:
    if st.button("⬅️ Previous") and st.session_state.job_page > 0:
        st.session_state.job_page -= 1
        st.rerun()

with col_page:
    st.write(f"Page {st.session_state.job_page + 1} of {total_pages}")

with col_next:
    if st.button("Next ➡️") and st.session_state.job_page < total_pages - 1:
        st.session_state.job_page += 1
        st.rerun()
