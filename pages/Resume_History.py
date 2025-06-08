import streamlit as st
import PyPDF2
from docx import Document
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from firebase_config import init_firebase
import base64
from sidebar import setup_sidebar
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# --- Initialize Firebase ---
db = init_firebase()

# --- Setup Page Config ---
try:
    st.set_page_config(
        page_title="Upload Resume | JobConnect", 
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

# --- Helper Functions ---
def suggest_fields(skills):
    field_map = {
        "python": "Software Development / Data Science",
        "machine learning": "AI Engineer / Data Scientist",
        "marketing": "Marketing Manager / Digital Marketer",
        "finance": "Financial Analyst / Finance Executive",
        "sales": "Sales Manager / Business Development",
        "engineering": "Engineer / Technical Specialist",
        "project management": "Project Manager",
        "healthcare": "Healthcare / Medical Staff",
        "education": "Teacher / Educational Trainer",
        "software": "Software Developer",
        "hardware": "Hardware Engineer",
        "cloud": "Cloud Engineer / Cloud Architect",
        "ai": "AI Specialist / Machine Learning Engineer",
        "data": "Data Analyst / Data Scientist"
    }
    suggested_fields = set()
    for skill in skills:
        field = field_map.get(skill.lower())
        if field:
            suggested_fields.add(field)
    return list(suggested_fields)

def save_job_to_firestore(job):
    saved_jobs_ref = db.collection("saved_jobs")
    existing_saved = (
        saved_jobs_ref
        .where("user_uid", "==", st.session_state["user_uid"])
        .where("job_title", "==", job.get('job_title'))
        .where("company_name", "==", job.get('company_name'))
        .stream()
    )
    
    if any(existing_saved):
        st.warning("⚠️ You already saved this job!")
        return

    saved_jobs_ref.add({
        "user_uid": st.session_state["user_uid"],
        "user_email": st.session_state.get("email"),
        "job_title": job.get('job_title', 'Unknown Position'),
        "company_name": job.get('company_name', 'Unknown Company'),
        "location": job.get('location', 'Not specified'),
        "category": job.get('category', 'Not specified'),
        "job_description": job.get('job_description', 'No description available'),
        "timestamp": datetime.now()
    })

    st.success(f"✅ Saved {job.get('job_title')} at {job.get('company_name')} successfully!")

# --- Session States ---
def initialize_resume_states():
    default_states = {
        "resume_uploaded": False,
        "resume_text": "",
        "resume_file_base64": "",
        "resume_file_name": "",
        "detected_skills": [],
        "suggested_fields": [],
        "favorites": []
    }
    for key, val in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = val

initialize_resume_states()

# --- Upload Resume Section ---
st.title("📜 Resume Upload and Recommendations")
st.subheader("📄 Upload Your Resume")

with st.form("upload_form", clear_on_submit=True):
    uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
    submit_upload = st.form_submit_button(label="Upload Resume")

if submit_upload and uploaded_file and not st.session_state["resume_uploaded"]:
    resume_text = ""
    resume_base64 = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
    resume_filename = uploaded_file.name

    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        resume_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        resume_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

    if resume_text.strip():
        timestamp = datetime.now()

        email = st.session_state.get("email")

        
        db.collection("resume_uploads").document(email).set({
            "resume_text": resume_text,
            "timestamp": timestamp
        })

        st.session_state.update({
            "resume_uploaded": True,
            "resume_text": resume_text,
            "resume_file_base64": resume_base64,
            "resume_file_name": resume_filename
        })

        keywords = ["python", "machine learning", "marketing", "finance", "sales", "engineering",
                    "project management", "healthcare", "education", "software", "hardware", "cloud", "ai", "data"]
        detected_skills = [kw.title() for kw in keywords if kw in resume_text.lower()]
        st.session_state["detected_skills"] = detected_skills
        st.session_state["suggested_fields"] = suggest_fields(detected_skills)

        st.success("✅ Resume uploaded and data extracted successfully!")
        st.rerun()
    else:
        st.warning("⚠️ Could not extract text. Please upload a clearer resume.")

# --- After Upload Success: Display Dashboard ---
elif st.session_state["resume_uploaded"]:
    st.markdown("""
        <div style='background-color: #DFF0D8; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h2 style='color: #3c763d;'>🎉 Resume Uploaded Successfully!</h2>
            <p style='color: #3c763d;'>Here’s your personalized career dashboard based on your resume.</p>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("🧠 Top Skills Detected")
    st.success(f"**{', '.join(st.session_state['detected_skills'])}**") if st.session_state["detected_skills"] else st.info("No major skills detected from your resume.")

    st.subheader("🛤️ Suggested Career Fields")
    st.success(f"**{', '.join(st.session_state['suggested_fields'])}**") if st.session_state["suggested_fields"] else st.info("No strong field suggestion detected.")

    if st.session_state["detected_skills"]:
        st.subheader("📊 Skill Strength")
        np.random.seed(42)
        skills = st.session_state["detected_skills"]
        scores = np.random.randint(50, 100, size=len(skills))

        angles = np.linspace(0, 2 * np.pi, len(skills), endpoint=False).tolist()
        angles += angles[:1]
        scores_radar = np.concatenate((scores, [scores[0]]))

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🧭 Skill Radar")
            fig1 = plt.figure(figsize=(3, 3))
            ax1 = fig1.add_subplot(111, polar=True)
            ax1.plot(angles, scores_radar, linewidth=1.5)
            ax1.fill(angles, scores_radar, alpha=0.3)
            ax1.set_xticks(angles[:-1])
            ax1.set_xticklabels(skills, fontsize=7)
            ax1.set_yticklabels([])
            ax1.set_title("Radar Chart", size=10, pad=10)
            st.pyplot(fig1)

        with col2:
            st.markdown("#### 📈 Skill Scores")
            fig2, ax2 = plt.subplots(figsize=(3, 3))
            ax2.barh(skills, scores, color="skyblue")
            ax2.invert_yaxis()
            ax2.set_xlim(0, 100)
            ax2.set_title("Skill Scores Bar Chart", size=10)
            st.pyplot(fig2)

    st.subheader("🎯 Recommended Jobs for You")

    latest_resume_text = st.session_state["resume_text"]
    approved_jobs = db.collection("job_listings").where("approved", "==", True).stream()
    job_matches = []

    keywords = ["python", "machine learning", "marketing", "finance", "sales", "engineering",
                "project management", "healthcare", "education", "software", "hardware", "cloud", "ai", "data"]

    for job in approved_jobs:
        job_data = job.to_dict()
        job_description = job_data.get("job_description", "").lower()
        resume_text_lower = latest_resume_text.lower()

        match_score = sum(kw in job_description for kw in keywords) + sum(kw in resume_text_lower for kw in keywords)

        if match_score >= 5:
            job_data["match_score"] = match_score
            job_matches.append(job_data)

    job_matches = sorted(job_matches, key=lambda x: x["match_score"], reverse=True)

    if job_matches:
        for idx, job in enumerate(job_matches):
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

                if st.button(f"💾 Save Job {idx}", key=f"save_job_{idx}"):
                    save_job_to_firestore(job)
    else:
        st.info("⚡ No strong matches found. Try uploading a more detailed resume!")

    if st.button("🔄 Upload Another Resume"):
        for key in ["resume_uploaded", "resume_text", "resume_file_base64", "resume_file_name", "detected_skills", "suggested_fields"]:
            st.session_state[key] = False if key == "resume_uploaded" else "" if "text" in key or "file" in key else []
        st.rerun()

else:
    st.info("Please upload your resume to get personalized recommendations.")