import streamlit as st
import time
import requests
from firebase_admin import firestore
from firebase_config import init_firebase

# Initialize Firebase Admin
db = init_firebase()

# --- Setup Page Config ---
try:
    st.set_page_config(page_title="Login | JobConnect", 
                       page_icon="assets/favicon.png",
                       layout="wide")
except st.errors.StreamlitAPIException:
    pass

# Firebase Web API Key
FIREBASE_API_KEY = "AIzaSyB6g00OOTaV2Eri0yPwvBBGJQF_pTYE7p0"

# --- Firebase REST API Functions ---
def firebase_login(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    response = requests.post(url, json=payload)
    return response.json()

def firebase_reset_password(email):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={FIREBASE_API_KEY}"
    payload = {"requestType": "PASSWORD_RESET", "email": email}
    response = requests.post(url, json=payload)
    return response.json()

def firebase_lookup_user(id_token):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:lookup?key={FIREBASE_API_KEY}"
    payload = {"idToken": id_token}
    response = requests.post(url, json=payload)
    return response.json()

def send_verification_email(id_token):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={FIREBASE_API_KEY}"
    payload = {"requestType": "VERIFY_EMAIL", "idToken": id_token}
    return requests.post(url, json=payload)

# --- Main Page ---
def show():
    if "user_uid" not in st.session_state:
        st.markdown("""
            <style>
                [data-testid="stSidebar"], header, footer {display: none;}
            </style>
        """, unsafe_allow_html=True)

    if "user_uid" in st.session_state and "email" in st.session_state:
        st.success("✅ You are already logged in!")
        st.query_params.update(page="home")
        st.rerun()

    st.title("🔐 Welcome Back")
    st.subheader("Please log in to continue")

    tab1, tab2 = st.tabs(["🔑 Login", "🔒 Forgot Password?"])

    with tab1:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email", placeholder="Enter your email", key="login_email")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")
            submitted = st.form_submit_button("Login")

        if submitted:
            if email and password:
                with st.spinner('Authenticating...'):
                    login_response = firebase_login(email, password)

                    if "idToken" in login_response:
                        id_token = login_response["idToken"]

                        # --- Lookup fresh user info after login ---
                        user_info = firebase_lookup_user(id_token)

                        if "users" in user_info:
                            email_verified = user_info["users"][0].get("emailVerified", False)

                            if not email_verified:
                                st.error("🚫 Your email is not verified yet. Please check your inbox.")

                                if st.button("🔁 Resend Verification Email"):
                                    with st.spinner('Sending verification email...'):
                                        resend_response = send_verification_email(id_token)

                                        if resend_response.status_code == 200:
                                            st.success("✅ Verification email sent again! Please check your inbox.")
                                        else:
                                            st.error("🚫 Failed to resend verification email. Please try again later.")

                                st.stop()

                            user_doc = db.collection("users").document(login_response["localId"]).get()

                            if user_doc.exists:
                                user_data = user_doc.to_dict()

                                if user_data.get("status", "Active") != "Active":
                                    st.error('🚫 Your account is deactivated. Contact our support team at [jobconnect_support@email.com](mailto:jobconnect_support@email.com).')
                                    st.stop()

                                st.session_state["user_uid"] = login_response["localId"]
                                st.session_state["email"] = email
                                st.session_state["idToken"] = login_response["idToken"]                                   
                                st.session_state["role"] = user_data.get("role", "Candidate")

                                st.success(f"✅ Welcome, {email}!")
                                time.sleep(1)

                                # --- Redirect based on role ---
                                if st.session_state["role"] == "Admin":
                                    st.switch_page("pages/AdminHome.py")
                                elif st.session_state["role"] == "Recruiter":
                                    st.switch_page("pages/Recruiter_Dashboard.py")
                                else:
                                    st.switch_page("pages/Home.py")
                            else:
                                st.warning("⚠️ User profile not found. Please contact admin.")
                        else:
                            st.error("🚫 Failed to retrieve user info. Please try again.")
                            st.stop()

                    else:
                        error_message = login_response.get("error", {}).get("message", "Unknown error")
                        if error_message == "INVALID_PASSWORD":
                            st.error("🚫 Incorrect password.")
                        elif error_message == "EMAIL_NOT_FOUND":
                            st.error("🚫 No account found with this email.")
                        else:
                            st.error(f"🚫 Login failed: {error_message}")
            else:
                st.warning("⚠️ Please fill in both fields.")

        # Register link
        st.markdown("""
            <div style='text-align:center; margin-top:10px;'>
            Don't have an account? <a href="/register" target='_self'>Register here</a>
            </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.subheader("🔒 Reset Your Password")
        reset_email = st.text_input("Enter your registered email", placeholder="you@example.com")

        if st.button("Send Reset Link"):
            if reset_email:
                reset_response = firebase_reset_password(reset_email)
                if "email" in reset_response:
                    st.success("✅ Password reset email sent! Please check your inbox.")
                else:
                    st.error(f"🚫 Failed to send reset link. {reset_response.get('error', {}).get('message', 'Unknown error')}")
            else:
                st.warning("⚠️ Please enter your email.")

show()
