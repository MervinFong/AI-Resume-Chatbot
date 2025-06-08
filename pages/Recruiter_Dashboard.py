import streamlit as st
from firebase_config import init_firebase
import pandas as pd
import altair as alt
from sidebar import setup_sidebar

# --- Initialize Firestore ---
db = init_firebase()

# --- Setup Page Config ---
try:
    st.set_page_config(page_title="Recruiter Dashboard | JobConnect", 
                       page_icon="assets/favicon.png",
                       layout="wide")
except st.errors.StreamlitAPIException:
    pass

setup_sidebar()

# --- Session Check ---
if "user_uid" not in st.session_state or st.session_state.get("role") != "Recruiter":
    st.error("🚫 Unauthorized access. Recruiters only.")
    st.stop()

recruiter_email = st.session_state.get("email")

# --- Page Title ---
st.title("📊 Recruiter Dashboard")
st.write("Quick overview of your job postings and candidate activities.")

# --- Load Recruiter's Jobs ---
jobs = db.collection("job_listings").where("poster_email", "==", recruiter_email).stream()
job_list = [job.to_dict() for job in jobs]

# --- Load Applications ---
applications = db.collection("applied_jobs").where("recruiter_email", "==", recruiter_email).stream()
application_list = [app.to_dict() for app in applications]

# --- Prepare Metrics Data ---
total_jobs = len(job_list)
approved_jobs = sum(1 for j in job_list if j.get("approved"))
total_applications = len(application_list)

# --- Clean and Consistent Summary Cards ---
st.subheader("📋 Quick Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div style="background-color: #ffffff; padding: 20px; border-radius: 15px; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.05); text-align: center; height: 150px;">
            <div style="font-size: 18px; font-weight: bold; color: #6c757d;">📢 Jobs Posted</div>
            <div style="font-size: 40px; font-weight: bold; margin-top: 10px; color: #4F8BF9;">{total_jobs}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div style="background-color: #ffffff; padding: 20px; border-radius: 15px; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.05); text-align: center; height: 150px;">
            <div style="font-size: 18px; font-weight: bold; color: #6c757d;">✅ Jobs Approved</div>
            <div style="font-size: 40px; font-weight: bold; margin-top: 10px; color: #28a745;">{approved_jobs}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div style="background-color: #ffffff; padding: 20px; border-radius: 15px; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.05); text-align: center; height: 150px;">
            <div style="font-size: 18px; font-weight: bold; color: #6c757d;">📥 Applications Received</div>
            <div style="font-size: 40px; font-weight: bold; margin-top: 10px; color: #E67E22;">{total_applications}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Job Approval Status Overview ---
st.subheader("📈 Job Approval Status Overview")

if job_list:
    # Prepare data
    approved_count = approved_jobs
    pending_count = total_jobs - approved_jobs
    total_count = total_jobs

    # Create DataFrame
    status_df = pd.DataFrame({
        "Status": ["Approved", "Pending Approval"],
        "Count": [approved_count, pending_count]
    })
    status_df["Percentage"] = (status_df["Count"] / total_count * 100).round(2)

    # Layout: 2 columns
    col1, col2 = st.columns([2, 1])  # Wider chart, narrower summary

    with col1:
        pie_chart = alt.Chart(status_df).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            color=alt.Color(field="Status", type="nominal", legend=alt.Legend(title="Job Status")),
            tooltip=[
                alt.Tooltip("Status:N"),
                alt.Tooltip("Count:Q"),
                alt.Tooltip("Percentage:Q", format=".1f")
            ]
        ).properties(width=350, height=300)

        label_chart = alt.Chart(status_df).mark_text(radius=110, size=13).encode(
            theta=alt.Theta(field="Count", type="quantitative"),
            text=alt.Text(field="Percentage", type="quantitative", format=".1f"),
            color=alt.value("white")
        )

        st.altair_chart(pie_chart + label_chart, use_container_width=False)

    with col2:
        st.markdown("""
            <div style="background-color: #f9f9f9; padding: 20px 25px; border-radius: 12px; 
                        border: 1px solid #ddd; font-size: 15px; line-height: 1.7;">
                <h4 style="margin-bottom: 15px;">📄 Summary</h4>
                <p><strong>Total Jobs:</strong> {total}</p>
                <p style="color: green;"><strong>Approved:</strong> {approved}</p>
                <p style="color: orange;"><strong>Pending:</strong> {pending}</p>
                <p><strong>Approval Rate:</strong> {rate:.1f}%</p>
            </div>
        """.format(
            total=total_count,
            approved=approved_count,
            pending=pending_count,
            rate=(approved_count / total_count * 100) if total_count else 0
        ), unsafe_allow_html=True)

else:
    st.info("ℹ️ No jobs posted yet to display chart.")

# --- Optional: Bar Chart: Jobs per Category ---
st.subheader("🗂️ Jobs by Category")
if job_list:
    category_counts = {}
    for job in job_list:
        cat = job.get("category", "Others")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    cat_df = pd.DataFrame(list(category_counts.items()), columns=["Category", "Jobs"])

    bar_chart = alt.Chart(cat_df).mark_bar().encode(
        x=alt.X("Category:N", sort='-y'),
        y="Jobs:Q",
        color="Category:N",
        tooltip=["Category", "Jobs"]
    )
    st.altair_chart(bar_chart, use_container_width=True)
else:
    st.info("ℹ️ No jobs posted yet to display categories.")
