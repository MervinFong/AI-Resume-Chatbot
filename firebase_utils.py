'''
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()
'''

# chatbot/firebase_utils.py

import firebase_admin
from firebase_admin import credentials, firestore
import os

# Only initialize Firebase once
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

def fetch_resume_from_firebase(email: str) -> str:
    """Retrieve the resume text uploaded by the user from Firestore"""
    doc_ref = db.collection("resume_uploads").document(email)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict().get("resume_text", "")
    return ""

def fetch_mbti_from_firebase(email: str) -> str:
    doc_ref = db.collection("mbti_results").document(email)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict().get("mbti", "")
    return ""

