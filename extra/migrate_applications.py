import firebase_admin
from firebase_admin import credentials, firestore
from tqdm import tqdm  # Optional: For progress bar

# Initialize Firebase Admin SDK
cred = credentials.Certificate('C:/Users/user/Documents/AI-Resume-Chatbot/firebase-adminsdk.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

def migrate_applications():
    applications_ref = db.collection("applied_jobs")
    applications = list(applications_ref.stream())

    updated_count = 0
    skipped_count = 0

    for app_doc in tqdm(applications, desc="Processing applications"):
        app_data = app_doc.to_dict()
        doc_id = app_doc.id

        if "recruiter_email" in app_data:
            skipped_count += 1
            continue  # Already migrated

        job_id = app_data.get("job_id")
        if not job_id:
            print(f"⚠️ Skipping application {doc_id}: Missing job_id.")
            skipped_count += 1
            continue

        job_doc = db.collection("job_listings").document(job_id).get()
        if not job_doc.exists:
            print(f"⚠️ Skipping application {doc_id}: Job {job_id} not found.")
            skipped_count += 1
            continue

        job_data = job_doc.to_dict()
        recruiter_email = job_data.get("poster_email") or job_data.get("company_email")
        if not recruiter_email:
            print(f"⚠️ Skipping application {doc_id}: Recruiter email not found in job {job_id}.")
            skipped_count += 1
            continue

        # Update the application document
        applications_ref.document(doc_id).update({"recruiter_email": recruiter_email})
        updated_count += 1

    print(f"\n✅ Migration completed: {updated_count} applications updated, {skipped_count} skipped.")

if __name__ == "__main__":
    migrate_applications()
