import re

def extract_job_query(user_input: str):
    """
    Extract job title and location from user query.
    e.g. 'marketing jobs in Kuala Lumpur' → ('marketing', 'Kuala Lumpur')
    """
    job_title = ""
    location = "Malaysia"  # Default fallback

    # Normalize spacing
    text = user_input.lower().strip()

    # Match patterns like 'X jobs in Y'
    match = re.search(r"(.*?)\s*jobs?\s*(?:in\s+(.*))?", text)
    if match:
        job_title = match.group(1).strip()
        if match.group(2):
            location = match.group(2).strip()

    if not job_title:
        job_title = "software engineer"

    return job_title, location
