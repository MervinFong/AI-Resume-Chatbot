import streamlit as st

def setup_sidebar():
    with st.sidebar:
        st.image("assets/logo.png", width=250)

        role = st.session_state.get("role")

        if role == "Candidate":
            st.page_link("pages/Home.py", label="🏠 Home")
            st.page_link("pages/Resume_History.py", label="📝 Upload Resume")
            st.page_link("pages/MBTI_Test_with_Firebase.py", label="🧠 MBTI Test")
            st.page_link("pages/MBTI_History_Detailed_Full_Report_Fixed.py", label="📄 MBTI Report")
            st.page_link("pages/Chatbot.py", label="🤖 Chatbot")
            st.page_link("pages/Job_Listings.py", label="💼 Job Listings")
            st.page_link("pages/Saved_Jobs.py", label="📥 Saved Jobs")
            st.page_link("pages/resume_maker.py", label="🛠️ Resume Maker")
            st.page_link("pages/My_Applications.py", label="📄 My Applications")
            st.page_link("pages/faq.py", label="❓ FAQ")
            st.page_link("pages/My_Support_Tickets.py", label="📩 Support Tickets")
            st.page_link("pages/Logout.py", label="🔓 Logout")

        elif role == "Recruiter":
            st.page_link("pages/Recruiter_Dashboard.py", label="🏠 Dashboard")
            st.page_link("pages/Manage_Listings.py", label="📋 Manage Listings")
            st.page_link("pages/Post_Job.py", label="📢 Post New Job")
            st.page_link("pages/Job_Listings.py", label="💼 Job Listings")
            st.page_link("pages/Manage_Applications.py", label="📥 Manage Applications")
            st.page_link("pages/Chatbot.py", label="🤖 Chatbot")
            st.page_link("pages/faq.py", label="❓ FAQ")
            st.page_link("pages/My_Support_Tickets.py", label="📩 Support Tickets")
            st.page_link("pages/Logout.py", label="🔓 Logout")

        elif role == "Admin":
            st.page_link("pages/AdminHome.py", label="🛡️ Admin Dashboard")
            st.page_link("pages/Manage_Users.py", label="👥 Manage Users")
            st.page_link("pages/Job_Listings.py", label="💼 Job Listings")
            st.page_link("pages/admin_approve.py", label="✅ Approve Job Listings")
            st.page_link("pages/Admin_Manage_Jobs.py", label="📝 Manage Listings")
            st.page_link("pages/Chatbot.py", label="🤖 Chatbot")
            st.page_link("pages/Support_Tickets_Admin.py", label="📩 View Support Tickets")
            st.page_link("pages/Logout.py", label="🔓 Logout")

        else:
            st.info("⚠️ Unknown role. Please log in again.")

    # Hide Streamlit Default Sidebar Navigation
    st.markdown("""
        <style>
        [data-testid="stSidebarNav"] { display: none; }
        </style>
    """, unsafe_allow_html=True)
