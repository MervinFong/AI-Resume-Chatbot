import streamlit as st
from firebase_config import init_firebase
from datetime import datetime
import base64
from utils.email_helper import send_email
from sidebar import setup_sidebar

# --- Initialize Firestore ---
db = init_firebase()

# --- Page Config ---
try:
    st.set_page_config(page_title="Recruiter Dashboard | JobConnect", 
                       page_icon="assets/favicon.png",
                       layout="wide")
except st.errors.StreamlitAPIException:
    pass

setup_sidebar()

# --- Session Check ---
if "user_uid" not in st.session_state or st.session_state.get("role") != "Recruiter":
    st.error("🚫 Unauthorized access. Recruiters only.")
    st.stop()

recruiter_email = st.session_state.get("email")

# --- Page Title ---
st.title("📋 Recruiter Dashboard")
st.write("Manage your posted jobs and candidate applications here.")

# --- Load Recruiter's Jobs ---
jobs_ref = db.collection("job_listings").where("poster_email", "==", recruiter_email).stream()

job_list = []
for job in jobs_ref:
    job_data = job.to_dict()
    job_data["doc_id"] = job.id
    job_list.append(job_data)

# --- Display Posted Jobs ---
if job_list:
    for job in job_list:
        with st.expander(f"📢 {job.get('job_title', 'No Title')} - {job.get('location', 'No Location')}"):
            st.markdown(f"**Posted On:** {job.get('date_posted', 'Unknown')}")
            st.markdown(f"**Description:** {job.get('job_description', 'No description available')[:300]}...")

            status_color = "green" if job.get("approved") else "red"
            st.markdown(f"**Status:** <span style='color:{status_color}; font-weight:bold;'>{'Approved' if job.get('approved') else 'Pending Approval'}</span>", unsafe_allow_html=True)

            # --- Divider ---
            st.divider()

            # --- Applications for This Job ---
            applications_ref = db.collection("applied_jobs").where("job_id", "==", job["doc_id"]).stream()
            applications = [app.to_dict() for app in applications_ref]

            if applications:
                st.subheader(f"📥 Applications ({len(applications)})")

                for idx, app in enumerate(applications):
                    with st.container():
                        st.markdown(f"**Candidate Email:** {app.get('user_email', 'Unknown')}")
                        st.markdown(f"**Applied On:** {app.get('timestamp').strftime('%Y-%m-%d %H:%M') if app.get('timestamp') else 'Unknown'}")
                        st.markdown(f"**Application Status:** {app.get('status', 'Pending')}")

                        if app.get("resume_file_base64"):
                            resume_bytes = base64.b64decode(app["resume_file_base64"])
                            st.download_button(
                                label="📄 Download Resume",
                                data=resume_bytes,
                                file_name=app.get("resume_file_name", "resume.pdf"),
                                mime="application/octet-stream",
                                key=f"resume_download_{idx}"
                            )
                        else:
                            st.warning("⚠️ No resume uploaded.")

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✅ Approve", key=f"approve_{idx}"):
                                db.collection("applied_jobs").document(app["doc_id"]).update({"status": "Approved"})

                                html_body = f"""
                                <html><body style="font-family: Arial, sans-serif;">
                                    <h2 style="color: #4CAF50;">🎉 Congratulations!</h2>
                                    <p>Your application for <b>{job.get('job_title', '')}</b> at <b>{job.get('company_name', '')}</b> has been <span style="color:green;"><b>Approved</b></span>!</p>
                                    <p>We look forward to connecting with you soon!</p>
                                </body></html>
                                """

                                send_email(
                                    to_email=app.get("user_email"),
                                    subject="🎉 Application Approved - JobConnect",
                                    body=html_body,
                                    is_html=True
                                )

                                st.success("✅ Application approved and candidate notified!")
                                st.experimental_rerun()

                        with col2:
                            if st.button("❌ Reject", key=f"reject_{idx}"):
                                db.collection("applied_jobs").document(app["doc_id"]).update({"status": "Rejected"})

                                html_body = f"""
                                <html><body style="font-family: Arial, sans-serif;">
                                    <h2 style="color: #E74C3C;">😞 Application Update</h2>
                                    <p>We regret to inform you that your application for <b>{job.get('job_title', '')}</b> at <b>{job.get('company_name', '')}</b> has been <span style="color:red;"><b>Rejected</b></span>.</p>
                                    <p>Thank you for applying. We wish you all the best!</p>
                                </body></html>
                                """

                                send_email(
                                    to_email=app.get("user_email"),
                                    subject="⚠️ Application Rejected - JobConnect",
                                    body=html_body,
                                    is_html=True
                                )

                                st.warning("❌ Application rejected and candidate notified.")
                                st.experimental_rerun()

            else:
                st.info("ℹ️ No candidates have applied for this job yet.")
else:
    st.info("ℹ️ You have not posted any jobs yet.")
