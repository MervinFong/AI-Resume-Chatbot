import torch
import spacy
import json
import datetime
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from firebase_admin import firestore
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    DistilBertTokenizer, DistilBertForSequenceClassification
)
from job_scrapers.jobstreet_wrapper import fetch_scraped_jobs
from chatbot.vector_store import load_vector_store, load_mbti_vector_store
from chatbot.resume_analyzer import analyze_resume
from fpdf import FPDF
from sentence_transformers import SentenceTransformer, util
import nltk

nltk.download("punkt", quiet=False)
nltk.download("punkt", download_dir="/tmp/nltk_data", quiet=False)
nltk.data.path.append("/tmp/nltk_data")

# === Device Setup ===
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# === NLP and Embedding Models ===
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    import importlib.util
    import sys
    from pathlib import Path
    spec = importlib.util.find_spec("en_core_web_md")
    if spec is None:
        raise OSError("Model en_core_web_md not found in runtime.")
    sys.path.append(str(Path(spec.origin).parent))
    nlp = spacy.load("en_core_web_md") 
retriever = load_vector_store().as_retriever()
chat_model = ChatOpenAI(temperature=0)
conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    retriever=retriever,
    return_source_documents=True
)

# === MBTI RAG Chain ===
mbti_retriever = load_mbti_vector_store().as_retriever()
mbti_chain = RetrievalQA.from_chain_type(llm=chat_model, retriever=mbti_retriever)

# === Resume Models ===
T5_MODEL_PATH = "t5model_v5"
t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_PATH)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_PATH).to(torch_device)

DISTILBERT_PATH = "distilbert_resume_classifier_v2"
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_PATH)
distilbert_model = DistilBertForSequenceClassification.from_pretrained(
    DISTILBERT_PATH,
    local_files_only=True,
    trust_remote_code=True
).to(torch_device)

label_map = {
    0: "Education", 1: "Engineering", 2: "Finance", 3: "Healthcare", 4: "Human Resources (HR)",
    5: "Information Technology (IT)", 6: "Marketing", 7: "Operations", 8: "Others", 9: "Sales"
}

session_memory = {}
intent_model = SentenceTransformer("all-MiniLM-L6-v2")
intent_examples = {
    "refine_resume": ["refine my resume", "rewrite my CV"],
    "analyze_resume": ["analyze my resume", "review my resume"],
    "mbti": ["what is my mbti"],
    "interview_tips": ["give me interview tips"],
    "help_request": ["what can you do"],
    "greeting": ["hello"]
}
intent_embeddings = {
    label: intent_model.encode(samples, convert_to_tensor=True)
    for label, samples in intent_examples.items()
}

def detect_intent(user_input):
    query_vec = intent_model.encode(user_input, convert_to_tensor=True)
    best_intent = "unknown"
    max_score = 0
    for label, examples in intent_embeddings.items():
        sim = util.cos_sim(query_vec, examples).max().item()
        if sim > max_score:
            max_score = sim
            best_intent = label
    return best_intent if max_score > 0.65 else "unknown"

def predict_category(text):
    tokens = distilbert_tokenizer(text, truncation=True, padding=True, return_tensors="pt").to(torch_device)
    with torch.no_grad():
        outputs = distilbert_model(**tokens)
        pred_id = outputs.logits.argmax(dim=1).item()
    return label_map[pred_id]

def run_t5_refiner(text):
    input_text = "Refine this resume: " + text
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(torch_device)
    with torch.no_grad():
        outputs = t5_model.generate(input_ids, max_new_tokens=512)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

def format_t5_output_to_chatbot(text):
    text = "{" + text.strip().strip(",") + "}"
    try:
        data = json.loads(text)
        suggestions = data.get("suggestions", [])
        summary = data.get("analysis_summary", "")
        bullets = "\n".join(f"- {s}" for s in suggestions)
        return f"📋 **Suggestions:**\n{bullets}\n\n📝 **Summary:**\n{summary}"
    except Exception:
        return f"⚠️ Could not format the output.\n\nRaw Output:\n{text}"

def get_internal_jobs(db, user_query):
    from spacy.lang.en.stop_words import STOP_WORDS

    nlp_query = nlp(user_query.lower())
    query_keywords = [token.text for token in nlp_query if not token.is_stop and not token.is_punct]

    job_matches = []
    internal_docs = db.collection("job_listings") \
        .where("approved", "==", True) \
        .where("active", "==", True) \
        .stream()

    for doc in internal_docs:
        data = doc.to_dict()
        if data.get("source", "") == "jobstreet":
            continue

        title = data.get("job_title", "").lower()
        company = data.get("company_name", "")
        description = data.get("job_description", "").lower()

        match_score = sum(1 for kw in query_keywords if kw in title or kw in description)

        if match_score > 0:
            job_matches.append((match_score, f"- {data['job_title']} at {company}"))

    job_matches.sort(reverse=True)
    return [match[1] for match in job_matches[:5]]

def get_jobstreet_jobs_from_firestore(db, user_query):
    jobs = []
    user_doc = nlp(user_query.lower())
    jobstreet_docs = db.collection("job_listings") \
        .where("approved", "==", True) \
        .where("active", "==", True) \
        .where("source", "==", "jobstreet") \
        .stream()
    for doc in jobstreet_docs:
        data = doc.to_dict()
        title = data.get("job_title", "")
        title_doc = nlp(title.lower())
        if user_doc.similarity(title_doc) > 0.75:
            link = data.get("link")
            if link:
                jobs.append(f"- {link}")
    return jobs[:5]

def mbti_rag_response(mbti_type):
    return mbti_chain.run(f"What are the traits and career recommendations for MBTI type {mbti_type}?")

def get_personalized_chain(db, email):
    if email not in session_memory:
        session_memory[email] = {
            "jobstreet_links": [],
            "chat_history": []
        }

    def run_chain(user_input: str):
        resume_text = ""
        resume_docs = db.collection("resume_uploads") \
            .where("email", "==", email) \
            .order_by("timestamp", direction=firestore.Query.DESCENDING) \
            .limit(1).stream()
        for doc in resume_docs:
            resume_text = doc.to_dict().get("resume_text", "")
            break

        if not user_input.strip() and resume_text:
            category = predict_category(resume_text)
            refined_raw = run_t5_refiner(resume_text)
            refined = format_t5_output_to_chatbot(refined_raw)
            return {
                "output_text": f"📄 I’ve analyzed your uploaded resume.\n\nPredicted Category: **{category}**\n\nHere’s a suggested refinement:\n\n{refined}",
                "input_documents": [],
                "resume_pdf": None
            }

        intent = detect_intent(user_input)
        resume_category = predict_category(resume_text) if resume_text else None

        if intent == "mbti":
            mbti_doc = db.collection("mbti_results").document(email).get()
            if mbti_doc.exists:
                mbti_type = mbti_doc.to_dict().get("mbti")
                if mbti_type:
                    rag_info = mbti_rag_response(mbti_type)
                    return {
                        "output_text": f"🧠 Your MBTI type is **{mbti_type}**.\n\n{rag_info}",
                        "input_documents": [],
                        "resume_pdf": None
                    }
            return {
                "output_text": "❌ No MBTI result found. Please take the MBTI test in the system first.",
                "input_documents": [],
                "resume_pdf": None
            }

        if intent == "refine_resume" and resume_text:
            refined_raw = run_t5_refiner(resume_text)
            refined = format_t5_output_to_chatbot(refined_raw)
            return {
                "output_text": f"✅ Your resume has been refined.\n\n{refined}",
                "input_documents": [],
                "resume_pdf": None
            }

        if intent == "analyze_resume" and resume_text:
            analysis = analyze_resume(resume_text)
            return {
                "output_text": f"🔍 Resume Analysis Result:\n{analysis}",
                "input_documents": [],
                "resume_pdf": None
            }

        if any(kw in user_input.lower() for kw in ["job", "career", "vacancy", "opening", "apply"]):
            internal_jobs = get_internal_jobs(db, user_input)
            scraped_links = get_jobstreet_jobs_from_firestore(db, user_input)

            response_text = """🗂️ Internal Job Postings:
{}

🔗 Additional JobStreet Listings:
{}""".format(
                "\n".join(internal_jobs) or "- No internal listings available.",
                "\n".join(scraped_links) or "- No JobStreet results found."
            )

            return {
                "output_text": response_text,
                "input_documents": [],
                "resume_pdf": None
            }

        history = session_memory[email]["chat_history"]
        response = conversational_chain.invoke({
            "question": user_input,
            "chat_history": history
        })
        session_memory[email]["chat_history"].append((user_input, response.get("answer", "")))
        response["output_text"] = response.get("answer", "")
        response.setdefault("resume_pdf", None)

        return response

    return run_chain
