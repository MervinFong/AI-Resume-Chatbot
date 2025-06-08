def match_resume_with_jobs(resume_text, job_docs):
    matches = []

    for job in job_docs:
        job_text = job.get("job_description", "").lower()
        resume_text_lower = resume_text.lower()

        keywords = job.get("skills", [])  # Assumes job listings have a 'skills' list
        matched_keywords = [kw for kw in keywords if kw.lower() in resume_text_lower]

        score = len(matched_keywords)
        if score > 0:
            matches.append({
                "job_title": job.get("job_title", "Unknown"),
                "company": job.get("company_name", "Unknown"),
                "location": job.get("location", "Unknown"),
                "score": score,
                "matched_keywords": matched_keywords
            })

    return sorted(matches, key=lambda x: x["score"], reverse=True)
