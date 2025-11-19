from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import sys

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

def progress(msg):
    # Displays a small animation to show progress without clogging the console
    for dot in "....":
        sys.stdout.write(f"\r{msg}{dot}")
        sys.stdout.flush()
        time.sleep(0.2)
    print() # newline

# =========================================
# CONFIGURATION
# =========================================
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = None
model = None
device = None

# =========================================
# PHASE 1 & 2: LOAD TOKENIZER & SELECT DEVICE
# =========================================
Log.info("Initializing model setup...")
try:
    progress("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    Log.success("Tokenizer loaded successfully ✅")
    
    # Determine the best device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Log.info(f"Device selected: {device.upper()}")

    # =========================================
    # PHASE 3: LOAD MODEL
    # =========================================
    Log.info("Loading model weights (this may take several minutes)...")
    progress("Downloading / Initializing model")
    
    # Check if a GPU is available and configure torch_dtype accordingly
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        # Use torch.bfloat16 for better memory usage on modern NVIDIA GPUs (Ampere architecture and later)
        # Use float16 as a fallback, or float32 for CPU
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_properties(device).major >= 8 else (torch.float16 if torch.cuda.is_available() else torch.float32),
        device_map="auto" if device == "cuda" else None
    )
    model.to(device)
    Log.success("Model loaded successfully ✅")

except Exception as e:
    # This is the primary exit point if model loading fails due to resources or configuration
    Log.error(f"Setup failed. Check dependencies and resources (especially VRAM/RAM). Detailed error: {e}")
    sys.exit(1)


# =========================================
# PHASE 4: GENERATION FUNCTION
# =========================================
def generate_response(user_input):
    Log.info(f"Generating response for: '{user_input[:40]}...'")
    try:
        # The prompt template for TinyLlama Chat uses this format:
        # <|system|>You are a helpful assistant.<|endoftext|><|user|>...<|endoftext|><|assistant|>
        prompt = f"<|user|>{user_input}<|endoftext|><|assistant|>"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        start = time.time()
        
        # Adjusting generation parameters for better chat quality and preventing infinite loops
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100,
            do_sample=True, # Recommended for chat models
            temperature=0.7, # Adds creativity
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id # Prevents warnings
        ) 
        end = time.time()
        
        # Decode the output and strip the original prompt if present (common in generation)
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Find the start of the assistant's reply
        assistant_tag = "<|assistant|>"
        if assistant_tag in full_response:
            generated_text = full_response.split(assistant_tag, 1)[-1].strip()
        else:
            generated_text = full_response.strip()

        Log.success(f"Response generated in {end - start:.2f} seconds ✅")
        return generated_text
        
    except Exception as e:
        Log.error(f"Error during generation ❌: {e}")
        return "Error generating response."


# =========================================
# FLASK APP INITIALIZATION AND ROUTES
# =========================================
app = Flask(__name__)
# CORS must be enabled so Express (on port 3000) can talk to Flask (on port 5050)
CORS(app) 

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        print("📩 Received Request Data:", data)

        if not data or "message" not in data:
            print("⚠️ Invalid request format")
            return jsonify({"error": "Invalid request format"}), 400

        user_message = data["message"]
        print(f"🧠 User said: {user_message}")

        bot_reply = generate_response(user_message)
        print(f"🤖 Bot replied: {bot_reply[:100]}...")

        response = jsonify({"reply": bot_reply})
        response.headers.add("Access-Control-Allow-Origin", "*")  # ✅ allow all origins
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response, 200

    except Exception as e:
        print(f"❌ Error in /chat: {e}")
        return jsonify({"error": str(e)}), 500

# =========================================
# SERVER START (ONLY executed when run directly)
# =========================================
if __name__ == "__main__":
    # Test run for verification
    Log.info("Starting initial test prompt to ensure model is functional...")
    test_text = "What is the capital of France?"
    response = generate_response(test_text)

    print("\n==============================")
    print(f"User Input   : {test_text}")
    print(f"Model Output : {response.strip()}")
    print("==============================\n")

    Log.success("All setup steps completed successfully. Starting Flask server. 🎉")
    
    # IMPORTANT FIX: Setting use_reloader=False prevents Flask from automatically
    # restarting the process if it detects file changes or encounters stability issues
    # during the heavy model initialization phase.
    app.run(host="127.0.0.1", port=5050, debug=True, use_reloader=False)
