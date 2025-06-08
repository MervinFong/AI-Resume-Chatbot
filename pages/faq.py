import streamlit as st
from firebase_config import init_firebase
from datetime import datetime
import base64
from google.cloud import firestore
from sidebar import setup_sidebar

# --- Initialize Firestore ---
db = init_firebase()

# --- Setup Page Config ---
try:
    st.set_page_config(
        page_title="FAQ | JobConnect", 
        page_icon="assets/favicon.png",
        layout="wide"
    )
except st.errors.StreamlitAPIException:
    pass

setup_sidebar()

# --- Main Page ---
st.title("❓ Frequently Asked Questions (FAQ)")

st.markdown("""
<div style="background-color: #f0f2f6; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);">
    <h3 style="color: #333;">Welcome to our FAQ Section!</h3>
    <p style="font-size: 16px;">Here you'll find answers to common questions about using JobConnect, uploading your resume, and more.  
    If you can't find your answer here, feel free to reach out to our support team. 🚀</p>
</div>
""", unsafe_allow_html=True)

# --- FAQ Section ---
st.header("🔍 General Information")

faq_list = [
    {
        "question": "How do I upload my resume?",
        "answer": """
        To upload your resume:
        1. Log in to your account.
        2. Navigate to the **Upload Resume** section.
        3. Follow the prompts to select and upload your file.  
        _Supported formats: PDF, DOCX._
        """
    },
    {
        "question": "How can I reset my password?",
        "answer": """
        If you forgot your password:
        - Go to the **Login** page.
        - Click **Forgot Password**.
        - Enter your email address to receive a reset link.
        """
    },
    {
        "question": "How can I apply for a job?",
        "answer": """
        After uploading your resume:
        - Visit the **Job Listings** page.
        - Browse available jobs.
        - Apply directly to roles matching your profile.
        """
    },
    {
        "question": "How do I update my profile or resume?",
        "answer": """
        To update your details:
        - Go to the **Profile Settings**.
        - Update your contact info, upload a new resume, etc.
        """
    },
    {
        "question": "What is the MBTI test and how does it help me?",
        "answer": """
        The **MBTI (Myers-Briggs Type Indicator)** test identifies your personality type and suggests career paths aligned with your strengths.
        """
    },
    {
        "question": "How can I contact support?",
        "answer": """
        Need help?
        - Click the **Contact Support** link in the sidebar.
        - Or email us directly at **support@jobconnect.com**.
        """
    }
]

for faq in faq_list:
    with st.expander(f"**{faq['question']}**"):
        st.markdown(faq['answer'])

st.markdown("---")

# --- Report Scam / Contact Admin Section ---
# --- Report Scam / Contact Admin Section ---
st.header("📩 Report an Issue or Contact Admin")

st.markdown("""
Encounter suspicious activities?  
Report it directly to our admin team — including screenshots if needed! 🚨
""")

# ✅ Standard inputs (no form)
user_name = st.text_input("Your Name", placeholder="Enter your name")

user_email = st.session_state.get("email", "")
st.text_input("Your Email", value=user_email, disabled=True)

user_message = st.text_area("Your Message or Scam Report", placeholder="Describe the issue...", height=150)

uploaded_file = st.file_uploader("Upload Screenshot Proof (optional)", type=["png", "jpg", "jpeg", "pdf"])

if st.button("📨 Submit Report"):  # ✅ Normal button, no form
    if user_name and user_email and user_message:
        try:
            proof_base64 = None
            if uploaded_file:
                proof_base64 = base64.b64encode(uploaded_file.read()).decode('utf-8')

            ticket_id = f"TICKET-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            user_uid = st.session_state.get("user_uid", "unknown")

            # ✅ Save ticket
            db.collection("support_tickets").add({
                "ticket_id": ticket_id,
                "user_uid": user_uid,
                "name": user_name,
                "email": user_email,
                "message": user_message,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "proof_file": proof_base64,
                "proof_filename": uploaded_file.name if uploaded_file else None,
                "status": "Pending",
                "admin_reply": "",
                "user_notified": True
            })

            st.success(f"✅ Report submitted! Ticket ID: **{ticket_id}** 🎉 \n\n Go to My Support Tickets to check the status.")

            st.markdown("""<br>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Failed to submit report: {e}")
    else:
        st.warning("⚠️ Please complete all fields before submitting.")

st.markdown("---")

# --- Additional Resources ---
st.header("📚 Additional Resources")
st.markdown("""
For more guidance, contact (https://www.jobconnect.com/help) for tutorials, best practices, and job search tips.
""")
