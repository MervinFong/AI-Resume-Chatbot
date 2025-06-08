import streamlit as st
import os
from dotenv import load_dotenv
from firebase_config import init_firebase
from chatbot.logger import log_chat_to_firestore
from chatbot.personalized_chain import get_personalized_chain
from datetime import datetime
from sidebar import setup_sidebar

# --- Load environment variables and Firebase ---
load_dotenv()
db = init_firebase()

# --- Setup Page Config ---
try:
    st.set_page_config(
        page_title="Chatbot | JobConnect",
        page_icon="assets/favicon.png",
        layout="wide"
    )
except st.errors.StreamlitAPIException:
    pass

setup_sidebar()

# --- Page Setup ---
st.title("\U0001F4AC JobConnect Chatbot")

# --- Get current user's email ---
email = st.session_state.get("email", "anonymous")

# --- Load personalized chatbot chain ---
try:
    personalized_chain = get_personalized_chain(db, email)
except Exception as e:
    st.error(f"❌ Failed to load assistant: {e}")
    st.stop()

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Render Existing Chat History ---
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])
    with st.chat_message("assistant"):
        st.markdown(chat["answer"], unsafe_allow_html=True)

# --- Display Welcome Message if No Chat Yet ---
if not st.session_state.chat_history:
    with st.chat_message("assistant"):
        st.markdown("""
**👋 Welcome to JobConnect!**  
I'm your smart assistant for job discovery, resume guidance, and career advice.

You can:
- 📄 Upload your resume and get recommendations  
- 🧠 Take the MBTI test and match careers to your personality  
- 💬 Ask me about how to apply, save jobs, or explore job listings

Try something like: *software engineer jobs in Kuala Lumpur*
""")

# --- Chat Input ---
user_input = st.chat_input("Ask about resume, MBTI, recruiter roles...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            response = personalized_chain(user_input)
            answer = response["output_text"]
            docs = response.get("input_documents", [])
            resume_pdf = response.get("resume_pdf")
        except Exception as e:
            answer = f"❌ Sorry, something went wrong: {e}"
            docs = []
            resume_pdf = None

        # === FIX Unicode issue ===
        safe_answer = answer.encode("utf-8", "ignore").decode("utf-8")

        # --- Smart resume uploader if no resume found ---
        if "no resume found" in answer.lower():
            st.markdown("📂 **No resume found. Please upload it here to continue:**")
            uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"], label_visibility="collapsed")

            if uploaded_file:
                from chatbot.resume_parser import extract_text_from_resume
                from chatbot.resume_analyzer import analyze_resume

                resume_text = extract_text_from_resume(uploaded_file)

                db.collection("resume_uploads").add({
                    "email": email,
                    "resume_text": resume_text,
                    "timestamp": datetime.utcnow()
                })

                st.success("✅ Resume uploaded successfully!")

                rerun_response = personalized_chain(user_input)
                rerun_answer = rerun_response["output_text"]
                rerun_docs = rerun_response.get("input_documents", [])
                rerun_pdf = rerun_response.get("resume_pdf")

                safe_rerun_answer = rerun_answer.encode("utf-8", "ignore").decode("utf-8")

                st.markdown("📄 **Resume uploaded. Here's my updated response:**")
                st.markdown(safe_rerun_answer, unsafe_allow_html=True)

                if rerun_pdf and isinstance(rerun_pdf, str) and rerun_pdf.startswith("Refined Content:"):
                    refined_text = rerun_pdf.replace("Refined Content:", "").strip()
                    with st.expander("📄 Refined Resume Preview"):
                        st.code(refined_text, language="markdown")

                st.session_state.chat_history.append({
                    "question": user_input,
                    "answer": safe_rerun_answer
                })

                clean_rerun_docs = [doc.page_content for doc in rerun_docs]
                log_chat_to_firestore(db, email, user_input, safe_rerun_answer, matched_chunks=clean_rerun_docs)

        else:
            st.markdown(safe_answer, unsafe_allow_html=True)

            if resume_pdf and isinstance(resume_pdf, str) and resume_pdf.startswith("Refined Content:"):
                refined_text = resume_pdf.replace("Refined Content:", "").strip()
                with st.expander("📄 Refined Resume Preview"):
                    st.code(refined_text, language="markdown")

            if docs:
                with st.expander("🔍 View relevant info retrieved from knowledge base"):
                    for doc in docs:
                        st.markdown(f"- {doc.page_content.strip()}")

            st.session_state.chat_history.append({
                "question": user_input,
                "answer": safe_answer
            })

            clean_docs = [doc.page_content for doc in docs]
            log_chat_to_firestore(db, email, user_input, safe_answer, matched_chunks=clean_docs)

    # --- Feedback Buttons ---
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("👍", key=f"thumbs_up_{user_input}"):
            db.collection("chat_feedback").add({
                "email": email,
                "question": user_input,
                "answer": safe_answer,
                "feedback": "positive",
                "timestamp": datetime.utcnow()
            })
            st.toast("Thanks for your feedback! 👍", icon="✅")

    with col2:
        if st.button("👎", key=f"thumbs_down_{user_input}"):
            db.collection("chat_feedback").add({
                "email": email,
                "question": user_input,
                "answer": safe_answer,
                "feedback": "negative",
                "timestamp": datetime.utcnow()
            })
            st.toast("We’ll work on improving this! 👎", icon="⚠️")
