import streamlit as st
from firebase_config import init_firebase
from sidebar import setup_sidebar
import base64
from datetime import datetime

# --- Initialize Firestore ---
db = init_firebase()

# --- Setup Page Config ---
try:
    st.set_page_config(page_title="Manage Applications | JobConnect", 
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
st.title("📥 Manage Applications")

# --- Load Applications for This Recruiter ---
applications_ref = db.collection("applied_jobs").where("recruiter_email", "==", recruiter_email).stream()

applications = []
for app_doc in applications_ref:
    app_data = app_doc.to_dict()
    app_data["doc_id"] = app_doc.id
    applications.append(app_data)

# --- Display Applications ---
if applications:
    for app in applications:
        with st.expander(f"📝 {app.get('user_email', 'Unknown')} - {app.get('job_title', 'Unknown Job')}"):
            st.markdown(f"**Job Title:** {app.get('job_title', 'Unknown')}")
            st.markdown(f"**Applied On:** {app.get('timestamp').strftime('%Y-%m-%d %H:%M') if app.get('timestamp') else 'Unknown'}")
            st.markdown(f"**Status:** {app.get('status', 'Pending')}")

            if app.get("resume_file_base64"):
                resume_bytes = base64.b64decode(app["resume_file_base64"])
                st.download_button(
                    label="📄 Download Resume",
                    data=resume_bytes,
                    file_name=app.get("resume_file_name", "resume.pdf"),
                    mime="application/pdf",
                    key=f"resume_download_{app['doc_id']}"
                )
            else:
                st.warning("⚠️ No resume uploaded.")

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"✅ Approve Application", key=f"approve_{app['doc_id']}"):
                    db.collection("applied_jobs").document(app["doc_id"]).update({
                        "status": "Approved",
                        "status_updated": datetime.utcnow()
                    })
                    st.success("✅ Application approved.")
                    st.rerun()
            with col2:
                if st.button(f"❌ Reject Application", key=f"reject_{app['doc_id']}"):
                    db.collection("applied_jobs").document(app["doc_id"]).update({
                        "status": "Rejected",
                        "status_updated": datetime.utcnow()
                    })
                    st.warning("❌ Application rejected.")
                    st.rerun()
else:
    st.info("ℹ️ No applications received yet.")
