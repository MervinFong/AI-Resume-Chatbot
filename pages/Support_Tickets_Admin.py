import streamlit as st
from firebase_config import init_firebase
from datetime import datetime
from google.cloud import firestore
import base64
from utils.email_helper import send_email  # ✅ Import email helper!
from sidebar import setup_sidebar

# Initialize Firestore
db = init_firebase()

# --- Setup Page Config ---
try:
    st.set_page_config(
        page_title="Support Tickets | JobConnect", 
        page_icon="assets/favicon.png",
        layout="wide")
except st.errors.StreamlitAPIException:
    pass

setup_sidebar()

# --- Session Check ---
if "user_uid" not in st.session_state or st.session_state.get("role") != "Admin":
    st.error("🚫 Unauthorized access!")
    st.stop()

# Page Title
st.title("📩 Support Tickets Management")
st.info("View, reply, and manage user-submitted inquiries or scam reports.")

st.divider()

# Load Tickets
tickets_ref = db.collection("support_tickets")
tickets = tickets_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).stream()

ticket_list = []

for ticket in tickets:
    ticket_data = ticket.to_dict()
    ticket_id = ticket.id
    ticket_data["doc_id"] = ticket_id
    ticket_list.append(ticket_data)

# Badge Counts
pending_count = sum(1 for t in ticket_list if t.get("status") == "Pending")
resolved_count = sum(1 for t in ticket_list if t.get("status") == "Resolved")

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"<div style='background-color: #f1c40f; padding: 15px; border-radius: 8px; color: white; text-align: center;'>Pending Tickets: <b>{pending_count}</b></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div style='background-color: #2ecc71; padding: 15px; border-radius: 8px; color: white; text-align: center;'>Resolved Tickets: <b>{resolved_count}</b></div>", unsafe_allow_html=True)

st.divider()

# Display Tickets
if ticket_list:
    for ticket in ticket_list:
        status = ticket.get("status", "Pending")
        status_color = "#2ecc71" if status == "Resolved" else "#f39c12"
        status_emoji = "🟢 Resolved" if status == "Resolved" else "🟠 Pending"
        with st.expander(f"📨 {ticket.get('ticket_id', 'No ID')} | {status_emoji}"):
            st.markdown(f"**Submitted at:** {ticket.get('timestamp').strftime('%Y-%m-%d %H:%M') if ticket.get('timestamp') else 'Unknown'}", unsafe_allow_html=True)
            st.write(f"**Message:** {ticket.get('message', '')}")

            if ticket.get("proof_file"):
                st.write("**Proof Uploaded:**")
                st.image(base64.b64decode(ticket["proof_file"]), use_container_width=True)

            st.write(f"**Current Admin Reply:** {ticket.get('admin_reply', 'No reply yet.')}")

            # Reply Form
            with st.form(f"reply_form_{ticket['doc_id']}"):
                reply_text = st.text_area("✏️ Write a reply to user", value=ticket.get("admin_reply", ""), height=100)
                submitted_reply = st.form_submit_button("Send/Update Reply")

                if submitted_reply:
                    # Update Firestore
                    db.collection("support_tickets").document(ticket["doc_id"]).update({
                        "admin_reply": reply_text,
                        "user_notified": False
                    })

                    # Send Email Notification to User
                    html_body = f"""
                    <html>
                    <body style="font-family: Arial, sans-serif; color: #333;">
                        <h2 style="color: #4CAF50;">📩 Support Ticket Update</h2>
                        <p>Dear {ticket.get('name', 'User')},</p>
                        <p>Our admin has replied to your support ticket:</p>
                        <blockquote style="background-color: #f9f9f9; padding: 10px; border-left: 4px solid #4CAF50;">
                            {reply_text}
                        </blockquote>
                        <p>Please log in to your JobConnect account to view the full details.</p>
                        <p style="margin-top:30px;">Best Regards,<br>JobConnect Support Team</p>
                    </body>
                    </html>
                    """

                    send_email(
                        to_email=ticket.get("email", ""),
                        subject="📩 Support Ticket Reply - JobConnect",
                        body=html_body,
                        is_html=True
                    )

                    st.success("✅ Reply sent successfully and user notified!")
                    st.rerun()

            # Status Buttons
            col1, col2 = st.columns(2)
            with col1:
                if ticket.get("status") == "Pending":
                    if st.button(f"✅ Mark Resolved ({ticket['doc_id']})"):
                        db.collection("support_tickets").document(ticket["doc_id"]).update({"status": "Resolved"})
                        st.success(f"✅ Ticket {ticket['doc_id']} marked as Resolved.")
                        st.rerun()
            with col2:
                if ticket.get("status") == "Resolved":
                    if st.button(f"🔄 Mark Pending ({ticket['doc_id']})"):
                        db.collection("support_tickets").document(ticket["doc_id"]).update({"status": "Pending"})
                        st.info(f"🔄 Ticket {ticket['doc_id']} marked as Pending.")
                        st.rerun()
else:
    st.info("ℹ️ No support tickets yet.")
