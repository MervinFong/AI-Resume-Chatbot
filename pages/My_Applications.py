import streamlit as st
from firebase_config import init_firebase
from sidebar import setup_sidebar

# --- Setup ---
db = init_firebase()

# --- Setup Page Config ---
try:
    st.set_page_config(
        page_title="My Applications | JobConnect", 
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

# --- Main Content ---
st.title("📋 My Job Applications")

if "user_uid" not in st.session_state:
    st.error("🚫 You must log in to view your applications.")
    st.stop()

user_uid = st.session_state["user_uid"]

# Fetch applications
applications_query = db.collection("applied_jobs").where("user_uid", "==", user_uid).stream()
applications = [{**doc.to_dict(), "doc_id": doc.id} for doc in applications_query]

if not applications:
    st.info("You haven't applied for any jobs yet.")
else:
    for app in applications:
        status = app.get('status', 'Applied')

        color = {
            "Applied": "#3498DB",     # Blue
            "Accepted": "#27AE60",    # Green
            "Rejected": "#E74C3C"     # Red
        }.get(status, "#7F8C8D")  # Default Gray

        with st.container():
            st.markdown(f"""
            <div style='background-color: #fdfdfd; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px;'>
                <h4 style='margin-bottom: 5px;'>{app.get('job_title', 'Unknown Position')}</h4>
                <p style='margin: 0 0 10px; color: #666;'>Company: {app.get('company_name', 'Unknown Company')}</p>
                <span style='background-color: {color}; color: white; padding: 5px 10px; border-radius: 20px; font-size: 13px;'>
                    {status}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Withdraw Button
            if status == "Applied":
                if st.button(f"Withdraw Application for {app.get('job_title')}", key=f"withdraw_{app['doc_id']}"):
                    db.collection("applied_jobs").document(app["doc_id"]).delete()
                    st.success("✅ Application withdrawn successfully!")
                    st.rerun()
