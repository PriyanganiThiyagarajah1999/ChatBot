# =========================================
# TinyLlama Chat API using FastAPI
# =========================================

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import time
import uvicorn
import asyncio
import sys
from pymongo import MongoClient
from retriever import get_relevant_docs


# =========================================
# LOGGING UTILITIES
# =========================================
class Log:
    @staticmethod
    def info(msg): print(f"\033[94m[INFO]\033[0m {msg}")
    @staticmethod
    def success(msg): print(f"\033[92m[SUCCESS]\033[0m {msg}")
    @staticmethod
    def warning(msg): print(f"\033[93m[WARNING]\033[0m {msg}")
    @staticmethod
    def error(msg): print(f"\033[91m[ERROR]\033[0m {msg}")


# =========================================
# DATABASE SETUP
# =========================================
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "insurance_bot"
COLLECTION_NAME = "qa_cache"

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
qa_collection = db[COLLECTION_NAME]

Log.info("MongoDB connected ✅")


# =========================================
# MODEL CONFIGURATION
# =========================================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = None
model = None

Log.info("Initializing TinyLlama model setup...")

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
Log.info(f"Device selected: {device.upper()}")

try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    Log.success("Tokenizer loaded successfully ✅")

    # Optional quantization (much faster + less VRAM)
    quant_config = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    model.to(device)
    Log.success("Model loaded successfully ✅")

except Exception as e:
    Log.error(f"Model setup failed ❌: {e}")
    sys.exit(1)


# =========================================
# RESPONSE GENERATION FUNCTION
# =========================================
async def generate_response(query: str, context: str = "", max_new_tokens: int = 150) -> str:
    """Generate TinyLlama response using retrieved context."""
    try:
        prompt = (
            f"<|system|>You are an expert insurance assistant. Use the given context to answer clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"<|user|>{query}<|endoftext|><|assistant|>"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        end = time.time()

        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        assistant_tag = "<|assistant|>"
        generated_text = (
            full_response.split(assistant_tag, 1)[-1].strip()
            if assistant_tag in full_response
            else full_response.strip()
        )

        Log.success(f"Response generated in {end - start:.2f}s ✅")
        return generated_text

    except Exception as e:
        Log.error(f"Error generating response: {e}")
        return "Sorry, I couldn’t generate a proper response."


# =========================================
# FASTAPI INITIALIZATION
# =========================================
app = FastAPI(title="TinyLlama Chat API")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================
# ROUTES
# =========================================
@app.get("/")
async def root():
    return {"status": "TinyLlama RAG API is running!"}


@app.post("/chat")
async def chat(request: Request):
    """RAG + Mongo-cached chatbot endpoint"""
    try:
        data = await request.json()
        query = data.get("message", "").strip()

        if not query:
            return {"error": "Missing 'message' field."}

        Log.info(f"User query: {query}")

        # 1️⃣ Check if question already cached
        cached = qa_collection.find_one({"query": query})
        if cached:
            Log.info("Returning cached response ✅")
            return {
                "reply": cached["answer"],
                "cached": True,
                "source": "mongodb"
            }

        # 2️⃣ Retrieve relevant documents
        docs = get_relevant_docs(query, k=5)
        context = "\n".join([d["text"] for d in docs])

        if not context.strip():
            context = "No relevant information found in knowledge base."

        # 3️⃣ Generate model response
        answer = await generate_response(query, context)

        # 4️⃣ Save Q&A to MongoDB
        qa_collection.insert_one({
            "query": query,
            "answer": answer,
            "context": context,
            "timestamp": time.time()
        })

        return {
            "reply": answer,
            "cached": False,
            "source": "tinyllama",
            "context_docs": len(docs)
        }

    except Exception as e:
        Log.error(f"Error in /chat endpoint: {e}")
        return {"error": str(e)}


# =========================================
# MAIN ENTRY POINT
# =========================================
if __name__ == "__main__":
    test_prompt = "What is the policy number on the tax invoice?"
    test_output = asyncio.run(generate_response(test_prompt, "Pre-existing conditions refer to illnesses..."))
    print(f"\n🧠 Test Prompt: {test_prompt}")
    print(f"🤖 Model Reply: {test_output}\n")
    Log.success("All setup complete. Starting server... 🎉")

    uvicorn.run("app:app", host="127.0.0.1", port=5050, reload=False)