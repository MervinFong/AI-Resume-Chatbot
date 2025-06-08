import torch
import spacy
import nltk
import os
from nltk.tokenize import sent_tokenize
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    T5Tokenizer, T5ForConditionalGeneration
)

# === Model Paths ===
model_path = "distilbert_resume_classifier_v2"
t5_path = "./t5model_v5"

# === Load Tokenizers ===
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_path)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_path)

# === Load Models with safetensors ===
distilbert_model = DistilBertForSequenceClassification.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float32
)

t5_model = T5ForConditionalGeneration.from_pretrained(
    t5_path,
    local_files_only=True,
    torch_dtype=torch.float32
)

# === Download required NLTK data ===
nltk_data_dir = "/tmp/nltk_data"
nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)
nltk.data.path.append(nltk_data_dir)

# === Load spaCy model ===
nlp = spacy.load("en_core_web_md")

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
distilbert_model.to(device)
t5_model.to(device)

# === Label Map for DistilBERT Classification ===
label_map = {
    0: "Education",
    1: "Engineering",
    2: "Finance",
    3: "Healthcare",
    4: "Human Resources (HR)",
    5: "Information Technology (IT)",
    6: "Marketing",
    7: "Operations",
    8: "Others",
    9: "Sales"
}

# === Inference Function: Predict Resume Category ===
def predict_category(text: str) -> str:
    if not text.strip():
        return "Unknown"

    tokens = distilbert_tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        output = distilbert_model(**tokens)
        pred_id = torch.argmax(output.logits, dim=1).item()
    return label_map.get(pred_id, "Unknown")

# === Inference Function: Refine Resume Text ===
def refine_resume(text: str) -> str:
    if not text.strip():
        return "❌ Resume content is empty. Please upload a valid resume."

    input_ids = t5_tokenizer.encode(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        output = t5_model.generate(
            input_ids=input_ids,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )

    return t5_tokenizer.decode(output[0], skip_special_tokens=True)

# === Post-Analysis Function ===
def analyze_resume(resume_text: str):
    doc = nlp(resume_text)
    keywords = list(set([
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop and token.is_alpha
    ]))

    sentences = sent_tokenize(resume_text)
    missing_sections = []

    text_lower = resume_text.lower()
    if "summary" not in text_lower:
        missing_sections.append("summary")
    if "skills" not in text_lower:
        missing_sections.append("skills")
    if "certification" not in text_lower:
        missing_sections.append("certifications")
    if "project" not in text_lower:
        missing_sections.append("projects")

    return {
        "keywords": keywords[:20],  # top 20 keywords
        "sentence_count": len(sentences),
        "missing_sections": missing_sections
    }

def analyze_refinement(refined_text):
    sections = {
        "Executive Profile": "Consider adding specific achievements or quantifiable results to showcase your impact in previous roles.",
        "Skill Highlights": "Organize skills into categories such as technical skills, soft skills, and certifications for better readability.",
        "Professional Experience": "Provide more specific details about your responsibilities. Use strong action verbs.",
        "Education": "Include relevant coursework, certifications, or honors if applicable.",
        "Skills": "Highlight your most relevant skills clearly and concisely."
    }

    keywords_to_refine = []
    suggestions = {}
    doc = nlp(refined_text.lower())

    for section, suggestion in sections.items():
        if section.lower() not in refined_text.lower():
            keywords_to_refine.append(section)
            suggestions[section] = suggestion

    summary = (
        "The resume has been refined for structure and clarity. However, some sections could still be improved. "
        "Focus on adding quantifiable achievements, organizing skills more clearly, and expanding on your experience."
    )

    return {
        "keywords_to_refine": keywords_to_refine,
        "suggestions": suggestions,
        "analysis_summary": summary
    }
