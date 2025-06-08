import cloudscraper
from bs4 import BeautifulSoup
import sys
import json
import time
import random

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Version/16.3 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0"
]

def scrape_jobstreet_jobs_cloud(keyword: str, location: str = "Malaysia", max_pages: int = 1):
    results = []
    keyword_slug = keyword.lower().replace(" ", "-")
    location_slug = location.lower().replace(" ", "-")

    for page in range(max_pages):
        start = page * 20
        url = f"https://www.jobstreet.com.my/en/job-search/{keyword_slug}-jobs/in-{location_slug}?start={start}"
        print(f"[DEBUG] Fetching page {page+1}: {url}", file=sys.stderr, flush=True)

        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

        scraper = cloudscraper.create_scraper()
        try:
            time.sleep(random.uniform(1.5, 3.5))  # polite scraping
            response = scraper.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                print(f"[DEBUG] ❌ Failed to fetch page {page+1}: HTTP {response.status_code}", file=sys.stderr, flush=True)
                continue

            soup = BeautifulSoup(response.content, "html.parser")
            job_cards = soup.select("a[data-automation='job-list-view-job-link']")
            print(f"[DEBUG] ✅ Found {len(job_cards)} job cards", file=sys.stderr, flush=True)

            for job in job_cards:
                href = job.get("href", "")
                if not href:
                    continue
                full_link = href if href.startswith("http") else "https://www.jobstreet.com.my" + href
                results.append({"link": full_link})

        except Exception as e:
            print(f"❌ Error during scraping: {e}", file=sys.stderr, flush=True)
            continue

    print(f"[DEBUG] 🔄 Final total job links scraped: {len(results)}", file=sys.stderr, flush=True)
    return results

# CLI entry point for subprocess use
if __name__ == "__main__":
    keyword = sys.argv[1] if len(sys.argv) > 1 else "Software Engineer"
    location = sys.argv[2] if len(sys.argv) > 2 else "Malaysia"
    scraped = scrape_jobstreet_jobs_cloud(keyword, location)
    sys.stdout.write(json.dumps(scraped))
    sys.stdout.flush()
