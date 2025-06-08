import streamlit as st
from firebase_config import init_firebase
from datetime import datetime

# --- Initialize Firebase ---
db = init_firebase()

# --- Page Settings ---
st.set_page_config(page_title="Chat Log Viewer | Admin", layout="wide")
st.title("📊 Admin Chat Log Viewer")
st.write("View chatbot interactions submitted by users.")

# --- Role Check (optional) ---
if st.session_state.get("role") != "Admin":
    st.error("🚫 Unauthorized access. Admins only.")
    st.stop()

# --- Fetch Chat Logs ---
chat_logs_ref = db.collection("chat_logs").order_by("timestamp", direction="DESCENDING").stream()

# --- Display in Table ---
chat_data = []
for doc in chat_logs_ref:
    d = doc.to_dict()
    chat_data.append({
        "Timestamp": d.get("timestamp").strftime("%Y-%m-%d %H:%M:%S"),
        "Email": d.get("email"),
        "Question": d.get("question"),
        "Answer": d.get("answer"),
        "Context (snippet)": "\n".join(d.get("context", [])[:1])  # preview first chunk
    })

if chat_data:
    st.dataframe(chat_data, use_container_width=True)
else:
    st.info("No logs found.")
