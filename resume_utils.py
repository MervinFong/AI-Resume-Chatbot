import fitz  # PyMuPDF
import os
from datetime import datetime

# Extract text from a PDF resume
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Save the uploaded resume to a temp folder (optional)
def save_resume_to_disk(uploaded_file, upload_folder="uploaded_resumes"):
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uploaded_file.name}"
    save_path = os.path.join(upload_folder, filename)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return save_path
