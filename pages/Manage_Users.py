# pages/Manage_Users.py

import streamlit as st
from firebase_config import init_firebase
from datetime import datetime
from sidebar import setup_sidebar

# --- Initialize Firestore ---
db = init_firebase()

# --- Setup Page Config ---
try:
    st.set_page_config(page_title="Manage Users | Admin Panel", 
                       page_icon="assets/favicon.png",
                       layout="wide")
except st.errors.StreamlitAPIException:
    pass
setup_sidebar()

# --- Access Control ---
if st.session_state.get("role") != "Admin":
    st.error("🚫 You are not authorized to view this page.")
    st.stop()

# --- Page Title ---
st.title("👥 Manage Users")

# --- Load Users ---
users_ref = db.collection("users").stream()
users_list = []

for user_doc in users_ref:
    user_data = user_doc.to_dict()
    user_data["doc_id"] = user_doc.id
    users_list.append(user_data)

# --- Display Users ---
if not users_list:
    st.info("ℹ️ No users found.")
else:
    for user in users_list:
        with st.container():
            status = user.get("status", "Active")  # Default is Active if not set
            color = "#2ecc71" if status == "Active" else "#e74c3c"

            st.markdown(f"""
                <div style='background-color: #f9f9f9; padding: 15px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                    <b>Email:</b> {user.get('email', 'Unknown')}<br>
                    <b>Role:</b> {user.get('role', 'Unknown')}<br>
                    <b>Status:</b> <span style='color:{color}; font-weight:bold;'>{status}</span>
                </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                if status == "Active":
                    if st.button(f"🛑 Deactivate", key=f"deactivate_{user['doc_id']}"):
                        db.collection("users").document(user["doc_id"]).update({
                            "status": "Deactivated",
                            "status_updated": datetime.utcnow()
                        })
                        st.success("✅ User deactivated successfully!")
                        st.rerun()

            with col2:
                if status == "Deactivated":
                    if st.button(f"♻️ Reactivate", key=f"reactivate_{user['doc_id']}"):
                        db.collection("users").document(user["doc_id"]).update({
                            "status": "Active",
                            "status_updated": datetime.utcnow()
                        })
                        st.success("✅ User reactivated successfully!")
                        st.rerun()
