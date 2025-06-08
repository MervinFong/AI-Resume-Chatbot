# job_scrapers/test_scraper_runner.py

from jobstreet_wrapper import fetch_scraped_jobs

if __name__ == "__main__":
    keyword = input("Enter job title: ")
    location = input("Enter location (default: Malaysia): ") or "Malaysia"
    
    jobs = fetch_scraped_jobs(keyword, location)
    print("\n--- Scraped Jobs ---")
    for job in jobs:
        print(job["link"])
