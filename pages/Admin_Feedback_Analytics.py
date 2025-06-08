import streamlit as st
from firebase_config import init_firebase

db = init_firebase()
st.set_page_config(page_title="Feedback Analytics | Admin", layout="wide")
st.title("📈 Chatbot Feedback Analytics")

if st.session_state.get("role") != "Admin":
    st.error("🚫 Unauthorized access. Admins only.")
    st.stop()

# Fetch feedback
feedback = db.collection("chat_feedback").stream()
positive, negative = 0, 0
data = []

for f in feedback:
    d = f.to_dict()
    data.append(d)
    if d["feedback"] == "positive":
        positive += 1
    elif d["feedback"] == "negative":
        negative += 1

# Show metrics
st.metric("👍 Positive", positive)
st.metric("👎 Negative", negative)
st.write("")

# Show detailed table
if data:
    st.dataframe(data, use_container_width=True)
else:
    st.info("No feedback recorded yet.")
