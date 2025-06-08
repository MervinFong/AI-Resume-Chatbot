import firebase_admin
from firebase_admin import credentials, firestore
from collections import defaultdict
import re

# --- Initialize Firebase ---
cred = credentials.Certificate('C:/Users/user/Documents/jobconnect_chatbot/firebase-adminsdk.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Fetch resumes from resume_uploads collection ---
resumes = db.collection("resume_uploads").stream()
resume_data = [
    {"email": doc.get("email"), "text": doc.get("resume_text")}
    for doc in resumes if doc.get("text") is not None
]

# --- Fetch jobs from job_listings collection (approved only) ---
jobs = db.collection("job_listings").where("approved", "==", True).stream()
job_data = [
    {
        "title": job.get("job_title"),
        "description": job.get("job_description", ""),
        "skills": job.get("skills", []),
        "company": job.get("company_name"),
        "location": job.get("location"),
    }
    for job in jobs
]

# --- Simple keyword matcher ---
def keyword_match(resume_text, job):
    resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
    job_keywords = set(re.findall(r'\b\w+\b', job["description"].lower()))
    job_keywords.update([skill.lower() for skill in job.get("skills", [])])
    return len(resume_words & job_keywords)

# --- Match each resume to top 3 jobs ---
for resume in resume_data:
    scored_jobs = []
    for job in job_data:
        score = keyword_match(resume["text"], job)
        scored_jobs.append((score, job))
    scored_jobs.sort(reverse=True, key=lambda x: x[0])

    print(f"\n🔍 Resume for {resume['email']}:")
    for score, job in scored_jobs[:3]:
        print(f"  ✅ {job['title']} at {job['company']} ({job['location']}) — Score: {score}")
