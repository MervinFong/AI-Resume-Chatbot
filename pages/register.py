from firebase_config import init_firebase
from firebase_admin import auth 
import streamlit as st
import re
import time
import base64
import requests
import json

# Initialize Firebase
db = init_firebase()

# Firebase Web API Key
FIREBASE_API_KEY = st.secrets["FIREBASE_API_KEY"]  

# Helper functions
def firebase_login(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    response = requests.post(url, json=payload)
    return response.json()

def send_verification_email(id_token):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={FIREBASE_API_KEY}"
    payload = {"requestType": "VERIFY_EMAIL", "idToken": id_token}
    return requests.post(url, json=payload)

# Page config
try:
    st.set_page_config(page_title="Register | JobConnect", 
                       page_icon="assets/favicon.png",
                       layout="wide")
except st.errors.StreamlitAPIException:
    pass

# Hide sidebar and header/footer
st.markdown("""
    <style>
    [data-testid="stSidebar"], [data-testid="stSidebarNav"], header, footer {display: none;}
    </style>
""", unsafe_allow_html=True)

# Load logo safely
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

logo_base64 = get_base64_image("connect_logo2.svg")

# Layout
with st.container():
    if logo_base64:
        st.image(f"data:image/svg+xml;base64,{logo_base64}", width=150)

    st.title("📝 Register")
    st.write("Create your account below")

    email = st.text_input("Email", placeholder="Enter your email")
    password = st.text_input("Password", type='password', placeholder="Create a password")
    user_type = st.selectbox("Role", ["Candidate", "Recruiter"])

    register_clicked = st.button("Register")

    # Validate email and password
    def is_valid_email(email):
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(email_regex, email) is not None

    def is_valid_password(password):
        return len(password) >= 6

    def check_email_exists(email):
        try:
            user = auth.get_user_by_email(email)
            return True
        except auth.UserNotFoundError:
            return False

    if register_clicked:
        if not is_valid_email(email):
            st.error("Invalid email format.")
        elif not is_valid_password(password):
            st.error("Password must be at least 6 characters.")
        elif check_email_exists(email):
            st.error("Email already registered. Please log in.")
        else:
            try:
                with st.spinner('Creating account...'):
                    user = auth.create_user(email=email, password=password)

                    # Save to Firestore
                    db.collection("users").document(user.uid).set({
                        "email": email,
                        "role": user_type
                    })

                    # Login to get idToken
                    login_response = firebase_login(email, password)
                    if "idToken" in login_response:
                        id_token = login_response["idToken"]
                        verify_response = send_verification_email(id_token)

                        if verify_response.status_code == 200:
                            st.success("✅ Registered! A verification email has been sent. Please check your inbox!")
                        else:
                            st.warning("⚠️ Registered, but failed to send verification email.")

                        time.sleep(2)
                        st.query_params.update(page="/Login")
                        st.rerun()

                    else:
                        st.warning("⚠️ Registered, but failed to login automatically. Please try login manually.")

            except Exception as e:
                st.error(f"Registration failed: {e}")

    st.markdown("""
        <div style='text-align:center; margin-top:20px;'>
        Already have an account? <a href="/Login" target='_self'>Login here</a>
        </div>
    """, unsafe_allow_html=True)
