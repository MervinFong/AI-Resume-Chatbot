import streamlit as st
import json
import math
import os
from firebase_config import init_firebase
from datetime import datetime
from sidebar import setup_sidebar
from google.cloud import firestore

# --- Setup Page Config ---
try:
    st.set_page_config(
        page_title="MBTI Test | JobConnect",
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

email = st.session_state.get("email")
uid = st.session_state.get("user_uid")

# --- Load MBTI Questions ---
file_path = os.path.join("data", "mbti_questions_full.json")
with open(file_path, "r", encoding="utf-8") as f:
    questions = json.load(f)

questions_per_page = 10
total_pages = math.ceil(len(questions) / questions_per_page)

# --- Initialize Session State ---
if "mbti_page" not in st.session_state:
    st.session_state.mbti_page = 0
if "mbti_answers" not in st.session_state:
    st.session_state.mbti_answers = [None] * len(questions)

st.title("🧠 MBTI Personality Test")

st.info("""
📜 **About the MBTI Test:**
- The MBTI (Myers-Briggs Type Indicator) helps you understand your personality type.
- This test has 70 carefully designed questions.
- It will take around 5–10 minutes to complete.
- Your result will suggest suitable careers and strengths.

✅ Please answer honestly for the best results!
""")

st.markdown(f"<h5 style='text-align:center;'>Page {st.session_state.mbti_page + 1} / {total_pages}</h5>", unsafe_allow_html=True)

start = st.session_state.mbti_page * questions_per_page
end = min(start + questions_per_page, len(questions))

with st.form("mbti_form"):
    for i in range(start, end):
        q = questions[i]
        st.markdown(f"**{i+1}. {q['question']}**")
        selected = st.radio(
            label="",
            options=q["options"],
            index=(0 if st.session_state.mbti_answers[i] is None else q["options"].index(st.session_state.mbti_answers[i])),
            key=f"q_{i}",
            horizontal=True
        )
        st.session_state.mbti_answers[i] = selected
        st.markdown("<hr style='margin-top:10px;margin-bottom:10px;'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.session_state.mbti_page > 0:
            if st.form_submit_button("⬅️ Previous"):
                st.session_state.mbti_page -= 1
                st.rerun()
    with col2:
        if st.session_state.mbti_page < total_pages - 1:
            if st.form_submit_button("Next ➡️"):
                st.session_state.mbti_page += 1
                st.rerun()
    with col3:
        if st.session_state.mbti_page == total_pages - 1:
            if st.form_submit_button("🎯 Submit Test"):
                scores = {"E":0, "I":0, "S":0, "N":0, "T":0, "F":0, "J":0, "P":0}
                for idx, answer in enumerate(st.session_state.mbti_answers):
                    if answer is not None:
                        q = questions[idx]
                        if answer == q["options"][0]:
                            scores[q["scores"][0]] += 1
                        else:
                            scores[q["scores"][1]] += 1

                mbti = ""
                mbti += "E" if scores["E"] >= scores["I"] else "I"
                mbti += "S" if scores["S"] >= scores["N"] else "N"
                mbti += "T" if scores["T"] >= scores["F"] else "F"
                mbti += "J" if scores["J"] >= scores["P"] else "P"

                st.success(f"🎉 Your MBTI type is: **{mbti}**")

                descriptions = {
                    "INTJ": "The Architect – Strategic, logical, and independent.",
                    "ENFP": "The Campaigner – Enthusiastic, imaginative, and sociable.",
                    "ISTJ": "The Logistician – Responsible, serious, and practical.",
                }
                careers = {
                    "INTJ": ["Engineer", "Strategist", "Developer"],
                    "ENFP": ["Public Speaker", "Marketer", "Writer"],
                    "ISTJ": ["Accountant", "Auditor", "Project Manager"],
                }

                if mbti in descriptions:
                    st.subheader("🧬 Personality Description")
                    st.info(descriptions[mbti])
                else:
                    st.warning("ℹ️ No description available for this type yet.")

                if mbti in careers:
                    st.subheader("💼 Career Suggestions")
                    for c in careers[mbti]:
                        st.markdown(f"- {c}")

                db = init_firebase()
                db.collection("mbti_results").document(uid).set({
                    "mbti": mbti,
                    "timestamp": datetime.now(),
                    "email": email
                })

                st.success("✅ Your MBTI result has been saved successfully!")
                st.info("""
🎯 **Next Step:**
You can now view your detailed personality report in the 📄 MBTI Report page!

_Thank you for completing the MBTI Test!_
""")
