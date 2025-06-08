import streamlit as st
import base64
import os 
from model_loader import ensure_models_downloaded
import nltk

NLTK_PATH = "/tmp/nltk_data"
nltk.data.path.append(NLTK_PATH)

# Ensure punkt is downloaded before use
nltk.download("punkt", download_dir=NLTK_PATH, quiet=True)

# Download models if not already present
ensure_models_downloaded()

if not os.path.exists("t5model_v5") or not os.path.exists("distilbert_resume_classifier_v2"):
    st.error("❌ Model folders not found. Please ensure 't5model_v5' and 'distilbert_resume_classifier_v2' exist in the project root.")

# --- SETUP ---

# Page config
try:
    st.set_page_config(page_title="JobConnect", 
                       page_icon="assets/favicon.png",
                       layout="wide")
except st.errors.StreamlitAPIException:
    pass

# --- PAGE DETECTION ---
current_page = st.query_params.get("page", ["landing"])[0].lower()

# 🔥 Force rerun if query changes
if "last_page" not in st.session_state:
    st.session_state["last_page"] = current_page
elif st.session_state["last_page"] != current_page:
    st.session_state["last_page"] = current_page
    st.rerun()

is_logged_in = st.session_state.get("email") is not None

# Function to encode logo
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load logo
logo_base64 = get_base64_image("assets/logo.png")  # Adjust path if needed

# --- PAGE DETECTION ---

current_page = st.query_params.get("page", ["landing"])[0].lower()
is_logged_in = st.session_state.get("email") is not None

# --- CONDITIONAL SIDEBAR HIDE ---

if current_page in ["landing", "login", "register", "faq"]:
    st.markdown("""
        <style>
        [data-testid="stSidebar"], [data-testid="stSidebarNav"], header, footer {
            display: none;
        }
        .css-18e3th9 {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

# --- CUSTOM CSS ---

st.markdown(
    f"""
    <style>
    body {{
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    .header {{
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        color: #2C3E50;
        margin-top: 50px;
    }}
    .subheader {{
        font-size: 20px;
        color: #7F8C8D;
        margin-top: 10px;
        text-align: center;
    }}
    .section {{
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        align-items: center;
        margin: 40px 0;
        background-color: #f9f9f9;
        padding: 40px 20px;
        border-radius: 12px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
        animation: fadeIn 1s ease;
    }}
    .section img {{
        max-height: 150px;
        margin: 20px;
    }}
    .section-text {{
        max-width: 600px;
        text-align: left;
        margin: 20px;
    }}
    .cta-button {{
        background-color: #3498DB;
        color: white;
        padding: 15px 30px;
        font-size: 18px;
        border-radius: 8px;
        font-weight: bold;
        text-decoration: none;
        text-align: center;
        display: inline-block;
        margin: 40px auto 20px;
        transition: background-color 0.3s ease;
    }}
    .cta-button:hover {{
        background-color: #2980B9;
    }}
    .footer {{
        text-align: center;
        margin-top: 60px;
        font-size: 15px;
        color: #BDC3C7;
    }}
    .footer a {{
        color: #3498DB;
        text-decoration: none;
    }}
    .footer a:hover {{
        text-decoration: underline;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- PAGE ROUTING ---

if is_logged_in and current_page == "home":
    import pages.Home as Home
    Home.show()

elif current_page == "login":
    import pages.Login as Login
    Login.show()

elif current_page == "register":
    import pages.register as Register
    Register.show()

elif current_page == "faq":
    import pages.faq as faq
    faq.show()

else:
    # --- LANDING PAGE CONTENT ---

    st.markdown("<div class='header'>Welcome to the AI Resume Screening Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Enhance Your Job Search with AI</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="section">
            <img src="data:image/png;base64,{logo_base64}" alt="Company Logo" />
            <div class="section-text">
                <p>Our AI-powered resume screening chatbot assists you in discovering ideal job opportunities and refining your resume. 
                Navigate using the options above to explore features and receive personalized recommendations.</p>
                <p>Whether you're a <strong>candidate</strong> searching for jobs or a <strong>recruiter</strong> seeking top talent, 
                our platform is designed to empower your success.</p>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        ## 📋 How It Works
        <ol>
            <li><strong>Upload your resume</strong> to receive improvement suggestions.</li>
            <li><strong>Browse job listings</strong> curated to your profile.</li>
            <li><strong>Get career guidance</strong> through our smart AI chatbot.</li>
        </ol>
        <p>Begin your journey today by logging in or registering. Experience a smarter, faster job search!</p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="text-align: center;">
            <a href="/Login" target='_self' class="cta-button" style="color: white; text-decoration: none; font-size: 18px; background-color: #2776EA; padding: 10px 20px; border-radius: 8px;">Get Started</a>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="footer">
            <p>Need help? Contact our support team at <a href="mailto:jobconnect_support@email.com">jobconnect_support@email.com</a>.</p>
            <p>Or visit the <a href="/faq" target='_self'>FAQ</a> page for more information.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
