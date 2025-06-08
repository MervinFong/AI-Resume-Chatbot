import streamlit as st
from firebase_config import init_firebase
from google.cloud import firestore
import base64
from datetime import datetime
from sidebar import setup_sidebar

# --- Initialize Firestore ---
db = init_firebase()

# --- Setup Page Config ---
try:
    st.set_page_config(
        page_title="My Support Tickets | JobConnect", 
        page_icon="assets/favicon.png",
        layout="wide"
    )
except st.errors.StreamlitAPIException:
    pass

setup_sidebar()

# --- Page Protection ---
if "user_uid" not in st.session_state or st.session_state.get("role") not in ["Candidate", "Recruiter"]:
    st.error("🚫 Unauthorized access!")
    st.switch_page("pages/Login.py")
    st.stop()

# --- Main Page ---
st.title("📩 My Support Tickets")

# --- Check Login ---
email = st.session_state.get("email")

if not email:
    st.error("🚫 You must be logged in to view your support tickets.")
    st.stop()

# --- Fetch User Tickets ---
tickets_ref = db.collection("support_tickets").where("email", "==", email)
tickets = tickets_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).stream()

ticket_list = []
new_reply_found = False

for ticket in tickets:
    ticket_data = ticket.to_dict()
    ticket_id = ticket.id
    ticket_data["doc_id"] = ticket_id
    ticket_list.append(ticket_data)

    if ticket_data.get("admin_reply") and not ticket_data.get("user_notified", True):
        new_reply_found = True

# --- Show notification if admin replied ---
if new_reply_found:
    st.success("📬 You have a new reply from Admin! Please check below.")

# --- Display Tickets ---
if ticket_list:
    for ticket in ticket_list:
        status = ticket.get("status", "Pending")
        status_color = "#2ecc71" if status == "Resolved" else "#f39c12"
        status_emoji = "🟢 Resolved" if status == "Resolved" else "🟠 Pending"

        with st.expander(f"📨 {ticket.get('ticket_id', 'No ID')} | {status_emoji}"):
            
            st.markdown(f"**Submitted On:** {ticket.get('timestamp').strftime('%Y-%m-%d %H:%M') if ticket.get('timestamp') else 'Unknown'}")
            st.markdown(f"**Your Message:** {ticket.get('message', '')}")

            if ticket.get("proof_file"):
                st.markdown("**Proof Uploaded:**")
                try:
                    proof_content = base64.b64decode(ticket["proof_file"])
                    if ticket.get("proof_filename", "").endswith(('.jpg', '.jpeg', '.png')):
                        st.image(proof_content, use_container_width=True)
                    elif ticket.get("proof_filename", "").endswith('.pdf'):
                        st.download_button("📄 Download Uploaded PDF", proof_content, file_name=ticket.get("proof_filename"))
                except Exception as e:
                    st.warning("⚠️ Error displaying uploaded proof.")

            st.markdown(f"**Admin Reply:** {ticket.get('admin_reply', 'No reply yet.')}")

            # Mark ticket as seen
            if not ticket.get("user_notified", True):
                db.collection("support_tickets").document(ticket["doc_id"]).update({
                    "user_notified": True
                })
else:
    st.info("ℹ️ You have not submitted any support tickets yet.")
