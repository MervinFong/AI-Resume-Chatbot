import os
import json
from typing import List, Dict
import spacy
from textblob import TextBlob
from job_scrapers.jobstreet_scraper_cloud import scrape_jobstreet_jobs_cloud

# ✅ Load spaCy model only when needed
def get_spacy_model():
    try:
        return spacy.load("en_core_web_md")
    except OSError:
        from spacy.cli import download
        download("en_core_web_md")
        return spacy.load("en_core_web_md")

# ✅ Synonym map for fallback matching
SYNONYM_MAP = {
    "software": ["developer", "engineer", "coder", "programmer"],
    "analyst": ["data analyst", "researcher", "insight specialist"],
    "technician": ["operator", "mechanic", "machinist", "repairman"],
    "marketing": ["advertiser", "promoter", "campaign manager", "brand manager"],
    "finance": ["accountant", "auditor", "banker", "financial analyst"],
    "sales": ["sales rep", "salesperson", "associate", "consultant"],
    "education": ["teacher", "tutor", "lecturer", "instructor"],
    "healthcare": ["nurse", "doctor", "medical officer", "therapist"],
    "admin": ["clerk", "secretary", "administrator", "office assistant"]
}

KNOWN_JOBS = list(SYNONYM_MAP.keys()) + [j for subs in SYNONYM_MAP.values() for j in subs]

# ✅ Call scraper safely with cloud error handling
def run_scraper(keyword: str, location: str) -> List[Dict]:
    try:
        print(f"[SCRAPER] Running for keyword: {keyword}")
        return scrape_jobstreet_jobs_cloud(keyword, location)
    except Exception as e:
        print(f"❌ Error during run_scraper({keyword}): {e}")
        return []

# ✅ Main smart fetch function
def fetch_scraped_jobs(keyword: str = "Software Engineer", location: str = "Malaysia") -> List[Dict]:
    tried = set()
    sanitized = keyword.strip().lower()

    # ✅ Step 1: Spell correction
    corrected = str(TextBlob(sanitized).correct())
    print(f"[SPELLCHECK] '{keyword}' → '{corrected}'")
    if corrected.lower() not in tried:
        tried.add(corrected.lower())
        jobs = run_scraper(corrected, location)
        if jobs:
            return jobs

    # ✅ Step 2: Fallback nouns via spaCy
    nlp = get_spacy_model()
    doc = nlp(corrected)
    fallback_nouns = [token.text.lower() for token in doc if token.pos_ in {"NOUN", "PROPN"}]

    for noun in fallback_nouns:
        if noun not in tried:
            tried.add(noun)
            print(f"[NLP RETRY] Retrying with noun: {noun}")
            jobs = run_scraper(noun, location)
            if jobs:
                return jobs

    # ✅ Step 3: Semantic similarity check
    keyword_doc = nlp(corrected)
    best_match, best_score = None, 0

    for job_term in KNOWN_JOBS:
        sim = keyword_doc.similarity(nlp(job_term))
        if sim > best_score:
            best_score = sim
            best_match = job_term

    if best_match and best_match not in tried and best_score > 0.75:
        tried.add(best_match)
        print(f"[SIMILARITY] '{corrected}' ≈ '{best_match}' (score={best_score:.2f})")
        jobs = run_scraper(best_match, location)
        if jobs:
            return jobs

    # ✅ Step 4: Try synonyms in same category
    for key, synonyms in SYNONYM_MAP.items():
        if key in corrected:
            for alt in synonyms:
                if alt not in tried:
                    tried.add(alt)
                    print(f"[SYNONYM RETRY] Trying synonym: {alt}")
                    jobs = run_scraper(alt, location)
                    if jobs:
                        return jobs

    print("[FAILURE] No jobs found after all fallback attempts.")
    return []
