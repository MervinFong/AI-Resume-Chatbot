import firebase_admin
from firebase_admin import auth, credentials, firestore
import os
import sys

# 🔧 Add parent directory to path before importing custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from firebase_config import init_firebase

# ✅ Initialize Firebase Admin
init_firebase()
db = firestore.client()

# ✅ 1. Create User
email = "ali@mail.com"
password = "123123"

user = auth.create_user(
    email=email,
    password=password,
    email_verified=True  # Mark as verified immediately
)

print(f"✅ Successfully created user: {user.uid}")

# ✅ 2. Save role to Firestore
db.collection("users").document(user.uid).set({
    "email": email,
    "role": "Candidate",  # Change to "Recruiter" if needed
})

print("✅ Email verified and user role saved.")
