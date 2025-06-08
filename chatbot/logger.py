from firebase_admin import firestore
from datetime import datetime
import uuid

def log_chat_to_firestore(db, email, question, answer, matched_chunks=[]):
    timestamp = datetime.utcnow()
    doc_id = f"log_{timestamp.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"
    log_entry = {
        "email": email or "anonymous",
        "question": question,
        "answer": answer,
        "context": matched_chunks,  # Now it's only plain strings
        "timestamp": timestamp
    }
    db.collection("chat_logs").document(doc_id).set(log_entry)
