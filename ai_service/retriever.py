# =========================================
# hybrid_rag_extractor.py — Full Hybrid RAG Retriever + LLM Integration
# =========================================

import os
import time
import hashlib
import chromadb
import numpy as np
from pymongo import MongoClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from chromadb.utils import embedding_functions

# OCR imports
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# Text processing
import re
import nltk
import nltk

# Download necessary tokenizers if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

# =========================================
# CONFIGURATION
# =========================================
CHROMA_DB_PATH = "chroma_store"
DOCS_PATH = "docs"
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "insurance_bot"
MONGO_COLLECTION = "policies"
MAX_MODEL_TOKENS = 1800  # max tokens per prompt for the model

# =========================================
# EMBEDDINGS + MODELS
# =========================================
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

semantic_encoder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# =========================================
# CHROMA SETUP
# =========================================
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(
    name="insurance_docs",
    embedding_function=embedder
)

# =========================================
# MONGODB CONNECTION
# =========================================
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[MONGO_DB]
mongo_collection = mongo_db[MONGO_COLLECTION]

# =========================================
# SIMPLE CACHE
# =========================================
_CACHE = {}

def cache_key(query):
    return hashlib.md5(query.encode()).hexdigest()

# =========================================
# CLEAN + CHUNK TEXT
# =========================================
def clean_text(text):
    # Remove zero-width spaces, weird control chars
    text = text.replace("\u200b", "").replace("\xa0", " ").replace("\ufeff", "")
    # Replace multiple whitespaces with single space
    text = re.sub(r"\s+", " ", text)
    # Remove any non-printable chars
    text = ''.join(c for c in text if c.isprintable())
    return text.strip()

def chunk_text(text, max_words=200):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = ""
    for sent in sentences:
        if len(chunk.split()) + len(sent.split()) <= max_words:
            chunk += " " + sent
        else:
            chunks.append(chunk.strip())
            chunk = sent
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# =========================================
# QUERY EXPANSION
# =========================================
def expand_query(query: str):
    q = query.lower()
    variants = [query]

    synonyms = {
        "pre-existing": ["prior illnesses", "conditions before policy purchase"],
        "hospital": ["hospitalization", "inpatient care", "hospital stay"],
        "ambulance": ["emergency transport", "medical transport"],
        "premium": ["monthly payment", "insurance cost", "installment"],
        "claim": ["file a claim", "claim process", "reimbursement"],
        "coverage": ["insurance cover", "benefits", "policy protection"],
        "policy": ["insurance policy number", "policy ID", "plan number"],
        "invoice": ["invoice number", "invoice no", "invoice #", "bill number"]
    }

    for key, vals in synonyms.items():
        if key in q:
            variants.extend(vals)

    return list(set(variants))

# =========================================
# LOCAL DOCS LOADER (PDF/TXT + OCR)
# =========================================
def load_local_docs():
    documents = []
    if not os.path.exists(DOCS_PATH):
        print(f"[WARNING] Docs path not found: {DOCS_PATH}")
        return documents

    for file_name in os.listdir(DOCS_PATH):
        path = os.path.join(DOCS_PATH, file_name)
        text = ""
        try:
            if file_name.lower().endswith(".pdf"):
                # PDF text extraction
                try:
                    reader = PdfReader(path)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                except Exception as e:
                    print(f"[ERROR] PDF parsing failed for {file_name}: {e}")

                # OCR fallback
                if len(text.strip()) < 30:
                    print(f"[INFO] Running OCR for scanned PDF: {file_name}")
                    images = convert_from_path(path)
                    for img in images:
                        gray = img.convert("L")
                        bw = gray.point(lambda x: 0 if x < 128 else 255, '1')
                        ocr_text = pytesseract.image_to_string(bw, config="--psm 11")
                        text += ocr_text + "\n"

            elif file_name.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

            if text.strip():
                text = clean_text(text)
                for chunk in chunk_text(text, max_words=300):
                    documents.append({"text": chunk, "meta": {"source": file_name}})
                print(f"[LOADED DOC] {file_name}: {text[:2000]}...")

        except Exception as e:
            print(f"[ERROR] Failed to load {file_name}: {e}")

    print(f"[INFO] Loaded {len(documents)} local docs (OCR enabled).")
    return documents

# =========================================
# BM25 KEYWORD SEARCH
# =========================================
def get_text_corpus():
    try:
        all_docs = collection.get()
        if not all_docs or "documents" not in all_docs:
            return [], []
        return all_docs["documents"], all_docs["ids"]
    except Exception as e:
        print(f"[WARNING] BM25 corpus unavailable: {e}")
        return [], []

def keyword_search(query, corpus_texts, corpus_ids, top_k=5):
    if not corpus_texts:
        return []
    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {"id": corpus_ids[i], "text": corpus_texts[i], "meta": {"source": "bm25", "score": float(scores[i])}}
        for i in top_indices
    ]

# =========================================
# MONGO RETRIEVAL
# =========================================
def get_mongo_results(query):
    search_fields = ["policy_name", "coverage", "description", "terms", "exclusions"]
    regex_query = {"$or": [{f: {"$regex": query, "$options": "i"}} for f in search_fields]}
    results = mongo_collection.find(regex_query).limit(3)
    docs = []
    for r in results:
        text = (
            f"Policy: {r.get('policy_name', '')}\n"
            f"Coverage: {r.get('coverage', '')}\n"
            f"Description: {r.get('description', '')}\n"
            f"Terms: {r.get('terms', '')}\n"
            f"Exclusions: {r.get('exclusions', '')}"
        ).strip()
        text = clean_text(text)
        for chunk in chunk_text(text, max_words=300):
            docs.append({
                "text": chunk,
                "meta": {"source": "MongoDB", "id": str(r.get('_id'))}
            })
    return docs

# =========================================
# RERANK + HYBRID SCORING
# =========================================
def rerank_docs(query, docs, top_k=5):
    if not docs:
        return []
    pairs = [(query, d["text"]) for d in docs]
    scores = reranker.predict(pairs)
    for d, s in zip(docs, scores):
        d["meta"]["rerank_score"] = float(s)
    ranked = sorted(docs, key=lambda x: x["meta"]["rerank_score"], reverse=True)
    return ranked[:top_k]

def hybrid_rank(query_emb, docs):
    if not docs:
        return docs
    doc_embeddings = semantic_encoder.encode([d["text"] for d in docs])
    sim_scores = np.dot(doc_embeddings, query_emb.T).squeeze()
    for i, d in enumerate(docs):
        rerank_score = d["meta"].get("rerank_score", 0.0)
        d["meta"]["final_score"] = float(0.7 * sim_scores[i] + 0.3 * rerank_score)
    return sorted(docs, key=lambda x: x["meta"]["final_score"], reverse=True)

# =========================================
# MAIN RAG PIPELINE
# =========================================
def get_relevant_docs(query: str, k: int = 5):
    start = time.time()
    ck = cache_key(query)
    if ck in _CACHE:
        return _CACHE[ck]

    queries = expand_query(query)
    all_results = []

    # 1️⃣ Chroma Semantic Search
    for q in queries:
        try:
            results = collection.query(query_texts=[q], n_results=k)
            if results and "documents" in results:
                for i in range(len(results["documents"][0])):
                    all_results.append({
                        "text": results["documents"][0][i],
                        "meta": results["metadatas"][0][i] if results["metadatas"] else {"source": "chroma"},
                        "id": results["ids"][0][i],
                    })
        except Exception as e:
            print(f"[ERROR] Chroma retrieval failed: {e}")

    # 2️⃣ Local Docs
    all_results.extend(load_local_docs())

    # 3️⃣ BM25
    corpus_texts, corpus_ids = get_text_corpus()
    all_results.extend(keyword_search(query, corpus_texts, corpus_ids, top_k=k))

    # 4️⃣ MongoDB
    all_results.extend(get_mongo_results(query))

    # Deduplicate
    seen = set()
    unique_docs = []
    for doc in all_results:
        doc_text_norm = clean_text(doc["text"]).lower()
        if doc_text_norm not in seen:
            unique_docs.append(doc)
            seen.add(doc_text_norm)

    # Rerank + Hybrid Scoring
    reranked = rerank_docs(query, unique_docs, top_k=min(k*2, 10))
    query_emb = semantic_encoder.encode([query])
    final_docs = hybrid_rank(query_emb, reranked)[:k]

    if not final_docs:
        final_docs = [{"text": "I couldn’t find relevant details in the available documents.",
                       "meta": {"source": "general"}}]

    _CACHE[ck] = final_docs
    return final_docs

# =========================================
# PROMPT PREPARATION (TRUNCATION)
# =========================================
def prepare_prompt(docs, query, max_tokens=MAX_MODEL_TOKENS):
    combined_text = ""
    total_tokens = 0
    for d in docs:
        doc_tokens = len(d["text"].split())
        if total_tokens + doc_tokens > max_tokens:
            remaining = max_tokens - total_tokens
            combined_text += " ".join(d["text"].split()[:remaining]) + "\n---\n"
            break
        combined_text += d["text"] + "\n---\n"
        total_tokens += doc_tokens

    prompt = f"""
You are an insurance assistant. Extract the answer to the following question from the documents below:

Documents:
{combined_text}

Question: {query}
Answer:
"""
    return prompt

# =========================================
# LLM INTEGRATION (example pseudo-code)
# =========================================
def query_llm(docs, query):
    prompt = prepare_prompt(docs, query)
    # Replace this with your actual LLM call
    # Example: response = llm.generate(prompt, max_tokens=200)
    print("[PROMPT SENT TO LLM]:")
    print(prompt[:1000] + "...\n")  # show first 1k chars
    return "LLM response placeholder (replace with your model call)"

# =========================================
# EXAMPLE USAGE
# =========================================
if __name__ == "__main__":
    user_query = "What is the policy number?"
    docs = get_relevant_docs(user_query, k=5)
    answer = query_llm(docs, user_query)
    print("ANSWER:", answer)
