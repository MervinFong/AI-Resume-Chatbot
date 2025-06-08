import sys
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


def scrape_jobstreet_jobs(keyword: str, location: str = "Malaysia", max_pages: int = 1):
    results = []

    # Setup Chrome options for headless mode if needed
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        for page in range(max_pages):
            start = page * 20
            query = keyword.replace(" ", "-")
            url = f"https://www.jobstreet.com.my/en/job-search/{query}-jobs/in-{location}?start={start}"
            driver.get(url)
            time.sleep(5)  # wait for JS to load job cards

            job_cards = driver.find_elements(By.CSS_SELECTOR, "a[class*='_1ubeeig5j']")

            if not job_cards:
                print("⚠️ No job cards found — page may have changed.")
                break

            for job in job_cards:
                try:
                    link = job.get_attribute("href")
                    title = job.text.strip() or "No title available"
                    results.append({
                        "title": title,
                        "link": link
                    })
                except Exception as e:
                    print("⚠️ Error reading job data:", e)

    finally:
        driver.quit()

    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python jobstreet_scraper.py <keyword> [location]")
        sys.exit(1)

    keyword = sys.argv[1]
    location = sys.argv[2] if len(sys.argv) > 2 else "Malaysia"

    jobs = scrape_jobstreet_jobs(keyword, location)
    print(json.dumps(jobs, indent=2, ensure_ascii=False))
