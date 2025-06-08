# AdminHome.py

import streamlit as st
import matplotlib.pyplot as plt
from firebase_config import init_firebase
from google.cloud import firestore
from datetime import datetime
from sidebar import setup_sidebar

# Initialize Firestore
db = init_firebase()

# --- Page Config ---
try:
    st.set_page_config(page_title="Admin Dashboard | JobConnect", 
                       page_icon="assets/favicon.png",
                       layout="wide")
except st.errors.StreamlitAPIException:
    pass

setup_sidebar()

# --- Session Check ---
if "user_uid" not in st.session_state or st.session_state.get("role") != "Admin":
    st.error("🚫 Unauthorized access!")
    st.stop()


# --- Page Title ---
st.title("🏢 Admin Dashboard")

st.info("Quick overview of system status and user activities.")

st.divider()

# --- Fetch Data ---

# Users
users_ref = db.collection("users")
users = list(users_ref.stream())
total_users = len(users)
total_candidates = sum(1 for user in users if user.to_dict().get("role") == "Candidate")
total_recruiters = sum(1 for user in users if user.to_dict().get("role") == "Recruiter")

# Support Tickets
tickets_ref = db.collection("support_tickets")
tickets = list(tickets_ref.stream())
pending_tickets = sum(1 for ticket in tickets if ticket.to_dict().get("status") == "Pending")
resolved_tickets = sum(1 for ticket in tickets if ticket.to_dict().get("status") == "Resolved")

# --- Display Stats ---

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div style='background-color: #3498db; padding: 20px; border-radius: 10px; color: white; text-align: center;'>👥 Total Users<br><b style='font-size:24px;'>{total_users}</b></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div style='background-color: #1abc9c; padding: 20px; border-radius: 10px; color: white; text-align: center;'>🧑‍💻 Candidates<br><b style='font-size:24px;'>{total_candidates}</b></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div style='background-color: #9b59b6; padding: 20px; border-radius: 10px; color: white; text-align: center;'>🧑‍💼 Recruiters<br><b style='font-size:24px;'>{total_recruiters}</b></div>", unsafe_allow_html=True)

st.divider()

col4, col5 = st.columns(2)
with col4:
    st.markdown(f"<div style='background-color: #f1c40f; padding: 20px; border-radius: 10px; color: white; text-align: center;'>📩 Pending Tickets<br><b style='font-size:24px;'>{pending_tickets}</b></div>", unsafe_allow_html=True)
with col5:
    st.markdown(f"<div style='background-color: #2ecc71; padding: 20px; border-radius: 10px; color: white; text-align: center;'>✅ Resolved Tickets<br><b style='font-size:24px;'>{resolved_tickets}</b></div>", unsafe_allow_html=True)

st.divider()

# --- Graph Section 🎨 ---

st.header("📊 System Analytics")

col6, col7 = st.columns(2)

# --- Pie Chart: Candidates vs Recruiters
with col6:
    fig1, ax1 = plt.subplots(figsize=(3, 3))  # Smaller size for web
    labels = ['Candidates', 'Recruiters']
    sizes = [total_candidates, total_recruiters]
    explode = (0.05, 0.05)
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=140)
    ax1.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
    st.pyplot(fig1)

# --- Bar Chart: Ticket Status
with col7:
    fig2, ax2 = plt.subplots(figsize=(4, 3))  # Smaller width
    ticket_labels = ['Pending', 'Resolved']
    ticket_counts = [pending_tickets, resolved_tickets]
    ax2.bar(ticket_labels, ticket_counts, color=['orange', 'green'])
    ax2.set_ylabel('Number of Tickets')
    ax2.set_title('Support Ticket Status')
    st.pyplot(fig2)

st.divider()

# --- Latest Activity ---

st.header("🕒 Latest Activity")

# Latest registered user
if users:
    latest_user = max(users, key=lambda x: x.update_time)
    user_data = latest_user.to_dict()
    st.write(f"**Latest User Registered:** {user_data.get('email', 'Unknown')} ({user_data.get('role', 'Unknown Role')})")

# Latest support ticket
if tickets:
    latest_ticket = max(tickets, key=lambda x: x.update_time)
    ticket_data = latest_ticket.to_dict()
    st.write(f"**Latest Support Ticket:** {ticket_data.get('ticket_id', 'Unknown')} submitted by {ticket_data.get('name', 'Unknown')}")
