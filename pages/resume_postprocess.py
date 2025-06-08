# pages/resume_postprocess.py

import streamlit as st
import pandas as pd

def show():
    st.title("🧠 Resume Analyzer")
    csv_path = r"C:\Users\user\Documents\t5-model\postprocessed_output.csv"

    try:
        df = pd.read_csv(csv_path)
        st.success("Resume postprocessing data loaded successfully!")

        st.dataframe(df[["input_text", "predicted", "keywords", "found_sections", "missing_sections"]], use_container_width=True)

        selected = st.selectbox("🔍 Select a resume to preview:", df["predicted"])
        st.markdown("#### ✨ Resume Preview")
        st.markdown(f"<div style='background-color:#f9f9f9;padding:10px;border-radius:10px;'>{selected}</div>", unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("Postprocessed resume file not found. Please ensure prediction with postprocessing has been run.")
