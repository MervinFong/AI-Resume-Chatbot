import fitz  # PyMuPDF
import docx
from io import BytesIO

def extract_text_from_resume(uploaded_file):
    try:
        file_ext = uploaded_file.name.lower()

        if file_ext.endswith(".pdf"):
            file_bytes = uploaded_file.read()
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                return "\n".join(page.get_text() for page in doc)

        elif file_ext.endswith(".docx"):
            doc_stream = BytesIO(uploaded_file.read())
            document = docx.Document(doc_stream)
            return "\n".join(para.text for para in document.paragraphs)

        else:
            return "❌ Unsupported file type. Please upload a PDF or DOCX resume."

    except Exception as e:
        return f"❌ Error extracting text from resume: {e}"

        return ""
