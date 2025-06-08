import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

# Try importing Streamlit secrets if deployed on Streamlit Cloud
try:
    import streamlit as st
    STREAMLIT_DEPLOY = True
except ImportError:
    from dotenv import load_dotenv
    load_dotenv()
    STREAMLIT_DEPLOY = False

def init_firebase():
    if not firebase_admin._apps:
        if STREAMLIT_DEPLOY:
            cred_dict = json.loads(st.secrets["FIREBASE_KEY_JSON"])
            cred = credentials.Certificate(cred_dict)
        else:
            cred_path = os.getenv("FIREBASE_CRED_PATH")
            cred = credentials.Certificate(cred_path)
        
        firebase_admin.initialize_app(cred)
    
    return firestore.client()
