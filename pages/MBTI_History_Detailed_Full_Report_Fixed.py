import streamlit as st
from firebase_config import init_firebase
from datetime import datetime
from sidebar import setup_sidebar

# --- Setup Page Config ---
try:
    st.set_page_config(
        page_title="MBTI Report | JobConnect", 
        page_icon="assets/favicon.png",
        layout="wide")
except st.errors.StreamlitAPIException:
    pass

setup_sidebar()

# --- Page Title ---
st.title("📜 Your MBTI Personality Report History")

# --- Page Protection (Login Check) ---
if "email" not in st.session_state:
    st.warning("🚫 Please log in to view your MBTI history.")
    st.switch_page("pages/Login.py")
    st.stop()

email = st.session_state["email"]
db = init_firebase()

# --- Fetch MBTI Results from Firestore ---
try:
    results_ref = db.collection("mbti_results").where("email", "==", email).order_by("timestamp", direction="DESCENDING")
    results = results_ref.stream()
except Exception as e:
    st.error(f"❌ Error fetching MBTI history: {e}")
    st.stop()

# --- Personality Information ---
personality_data = {
    "INTJ": {
        "name": "Introverted Intuitive Thinking Judging",
        "nameDescription": "Extraverted Thinking with Introverted Intuition",
        "epithet": "The Architect",
        "description": """INTJs are strategic thinkers. They are creative and can come up with unconventional ideas but are also analytical 
and driven to turn their ideas into reality. Often seen as independent, determined, and skeptical, they are natural planners and prefer structure."""
    },
    "ENFP": {
        "name": "Extraverted Intuitive Feeling Perceiving",
        "nameDescription": "Extraverted Feeling with Introverted Intuition",
        "epithet": "The Campaigner",
        "description": """ENFPs are known for their enthusiasm and creativity. They are deeply caring, empathetic, and enjoy connecting with people.
They are also curious and love exploring new possibilities, often juggling many ideas and interests at once."""
    },
    "ESTJ": {
        "name": "Extraverted Sensing Thinking Judging",
        "nameDescription": "Extraverted Thinking with Introverted Sensing",
        "epithet": "The Executive",
        "description": """ESTJs are strong-willed, efficient, and highly practical. They value tradition and stability and prefer to make decisions based on logic and facts. 
They are natural leaders, focused on getting things done and ensuring that rules and procedures are followed."""
    },
    # Reminder: Add remaining MBTI types for full version
}

# --- Career Mapping ---
career_mapping = {
    "INTJ": ["Engineer", "Strategist", "Software Developer"],
    "ENFP": ["Public Speaker", "Creative Director", "Writer"],
    "ESTJ": ["Project Manager", "Executive", "Military Officer"],
    # Add more MBTI career mappings
}

# --- Render MBTI Results ---
count = 0
for r in results:
    data = r.to_dict()
    mbti = data.get("mbti")
    timestamp = data.get("timestamp") or r.create_time

    # Safely handle Firestore timestamp
    if hasattr(timestamp, "to_datetime"):
        timestamp = timestamp.to_datetime()
    date_str = timestamp.strftime("%Y-%m-%d %H:%M")

    info = personality_data.get(mbti)

    with st.expander(f"🧬 {mbti} - {date_str}"):
        if info:
            st.markdown(f"### {info['epithet']}")
            st.markdown(f"**Type:** *{info['name']}*")
            st.markdown(f"**Cognitive Functions:** `{info['nameDescription']}`")
            st.markdown(f"**Personality Description:**\n{info['description']}")
        else:
            st.info("ℹ️ Description for this MBTI type is not available.")

        if mbti in career_mapping:
            st.markdown("**💼 Suggested Careers:**")
            st.success(", ".join(career_mapping[mbti]))

    count += 1

# --- No Results Fallback ---
if count == 0:
    st.info("⚠️ No MBTI results found. Please complete the MBTI test first!")
