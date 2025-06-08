import streamlit as st
from firebase_config import init_firebase
from datetime import datetime
from sidebar import setup_sidebar
import re

# --- Initialize Firebase ---
db = init_firebase()

# --- Setup Page Config ---
try:
    st.set_page_config(
        page_title="Post a Job | JobConnect", 
        page_icon="assets/favicon.png",
        layout="wide"
    )
except st.errors.StreamlitAPIException:
    pass

# --- Sidebar ---
setup_sidebar()

# --- Session Check ---
if "user_uid" not in st.session_state or st.session_state.get("role") != "Recruiter":
    st.error("🚫 Unauthorized access. Recruiters only.")
    st.stop()

# --- Clear form if just submitted ---
if st.session_state.get("clear_form", False):
    st.session_state["job_title"] = ""
    st.session_state["location"] = "Pahang"
    st.session_state["company_name"] = ""
    st.session_state["company_phone"] = ""
    st.session_state["category"] = "Engineering"
    st.session_state["job_description"] = ""
    st.session_state["skills_input"] = ""
    st.session_state["clear_form"] = False

# --- Page Title ---
st.title("📢 Post a New Job Listing")
st.info("Fill in the details below to post a new job. Admin approval is required before it becomes public.")

# --- Job Posting Form ---
with st.form("job_post_form"):
    col1, col2 = st.columns(2)

    with col1:
        job_title = st.text_input("Job Title *", key="job_title")
        location = st.selectbox(
            "Location *",
            options=[
                "Pahang", "Perak", "Selangor", "Kuala Lumpur", "Penang", "Johor",
                "Melaka", "Negeri Sembilan", "Terengganu", "Kelantan", "Sabah",
                "Sarawak", "Labuan", "Putrajaya", "Perlis", "Kedah", "Langkawi", "Others"
            ],
            key="location"
        )
        company_name = st.text_input("Company Name *", key="company_name")

    with col2:
        company_phone = st.text_input("Company Phone Number *", key="company_phone")
        category = st.selectbox(
            "Job Category *",
            options=[
                "Engineering", "Information Technology (IT)", "Marketing", "Sales",
                "Finance", "Human Resources (HR)", "Healthcare", "Education", "Operations", "Others"
            ],
            key="category"
        )
        poster_email = st.session_state.get("email", "")
        poster_uid = st.session_state.get("user_uid", "")

    job_description = st.text_area("Job Description *", height=150, key="job_description")
    skills_input = st.text_input("Required Skills (comma separated) *", key="skills_input")

    submit_button = st.form_submit_button(label="Submit Job Listing")

# --- After form submission ---
if submit_button:
    if not (job_title and location and company_name and company_phone and category and job_description and skills_input):
        st.error("⚠️ Please complete all required fields marked with *.")
    elif not re.match(r"^\+?\d{9,15}$", company_phone.replace(" ", "")):
        st.warning("📞 Please enter a valid phone number (9–15 digits, optional '+').")
    else:
        # --- Prevent duplicate by same recruiter ---
        existing_jobs = db.collection("job_listings") \
            .where("poster_email", "==", poster_email) \
            .where("job_title", "==", job_title) \
            .stream()

        if any(existing_jobs):
            st.warning("⚠️ You’ve already posted a job with this title.")
        else:
            date_posted = datetime.utcnow().strftime('%Y-%m-%d')
            timestamp = datetime.utcnow()

            job_data = {
                "job_title": job_title,
                "job_description": job_description,
                "skills": [skill.strip() for skill in skills_input.split(",")],
                "location": location,
                "company_name": company_name,
                "company_phone": company_phone,
                "category": category,
                "poster_email": poster_email,
                "poster_uid": poster_uid,
                "date_posted": date_posted,
                "timestamp": timestamp,
                "approved": False,
                "active": True
            }

            db.collection("job_listings").add(job_data)

            st.success("🎉 Job listing submitted successfully! It will appear once approved by an admin.")
            st.session_state["clear_form"] = True

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Post Another Job"):
                    st.rerun()
            with col2:
                if st.button("🔙 Back to Dashboard"):
                    st.switch_page("pages/Recruiter_Dashboard.py")
