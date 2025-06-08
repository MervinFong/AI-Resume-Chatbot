import firebase_admin
from firebase_admin import credentials, firestore
from job_scrapers.jobstreet_scraper_cloud import scrape_jobstreet_jobs_cloud
import time

# === Step 1: Category Keywords Map ===
CATEGORY_KEYWORDS = {
    "Education": ["teacher", "tutor", "lecturer", "instructor"],
    "Engineering": ["engineer", "technician", "mechanical", "electrical"],
    "Finance": ["accountant", "auditor", "financial analyst", "banker"],
    "Healthcare": ["nurse", "doctor", "medical", "pharmacist"],
    "Human Resources (HR)": ["hr", "recruiter", "talent acquisition"],
    "Information Technology (IT)": ["software engineer", "developer", "programmer", "it support"],
    "Marketing": ["marketing", "seo", "content creator", "advertiser"],
    "Operations": ["logistics", "operations", "warehouse", "supply chain"],
    "Others": ["admin", "clerk", "assistant"],
    "Sales": ["sales", "business development", "account manager"]
}

# === Step 2: Firebase Init ===
cred = credentials.Certificate("C:/Users/user/Documents/AI-Resume-Chatbot/firebase-adminsdk.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# === Step 3: Scraping and Uploading ===
def push_jobs_to_firestore():
    total_uploaded = 0

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            print(f"[INFO] Scraping '{keyword}' for category '{category}'")
            try:
                job_links = scrape_jobstreet_jobs_cloud(keyword, location="Malaysia", max_pages=1)
            except Exception as e:
                print(f"❌ Failed to scrape '{keyword}': {e}")
                continue

            for job in job_links:
                link = job.get("link")
                if not link:
                    continue

                db.collection("job_listings").add({
                    "job_title": keyword.title(),  # Approximated job title
                    "company_name": "Unknown",     # Can't extract company name
                    "link": link,
                    "category": category,
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    "approved": True,
                    "active": True,
                    "source": "jobstreet"
                })
                total_uploaded += 1

            time.sleep(2)  # polite delay to avoid 403

    print(f"\n✅ Finished uploading {total_uploaded} JobStreet jobs to Firestore.")

if __name__ == "__main__":
    push_jobs_to_firestore()
