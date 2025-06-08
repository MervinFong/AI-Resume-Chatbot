from firebase_admin import firestore
from langchain.schema import Document

def fetch_user_context_documents(db, email):
    context_docs = []
    context_docs += fetch_mbti(db, email)
    context_docs += fetch_resume_summary(db, email)
    context_docs += fetch_job_listings(db)
    context_docs += fetch_user_open_tickets(db, email)
    return context_docs

def fetch_mbti(db, email):
    results = db.collection("mbti_results").where("email", "==", email).order_by("timestamp", direction=firestore.Query.DESCENDING).limit(1).stream()
    for doc in results:
        mbti = doc.to_dict().get("result_type")
        return [Document(page_content=f"The user's MBTI type is {mbti}.", metadata={"source": "mbti"})]
    return []

def fetch_resume_summary(db, email):
    doc = db.collection("resumes").document(email).get()
    if doc.exists:
        summary = doc.to_dict().get("summary", "")
        return [Document(page_content=f"The user's resume summary is: {summary}", metadata={"source": "resume"})]
    return []

def fetch_job_listings(db):
    jobs = db.collection("job_listings").stream()
    docs = []
    for j in jobs:
        data = j.to_dict()
        title = data.get("job_title", "Unknown")
        company = data.get("company_name", "Unknown")
        desc = data.get("description", "")
        docs.append(Document(page_content=f"Job Posting: {title} at {company}. {desc}", metadata={"source": "job"}))
    return docs

def fetch_user_open_tickets(db, email):
    tickets = db.collection("support_tickets").where("email", "==", email).stream()
    return [Document(page_content=f"Ticket: {t.to_dict().get('message')} | Status: {t.to_dict().get('status')}", metadata={"source": "ticket"}) for t in tickets]
