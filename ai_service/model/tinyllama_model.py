from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import sys

# Optional: color printing (works on most terminals)
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
    for dot in "....":
        sys.stdout.write(f"\r{msg}{dot}")
        sys.stdout.flush()
        time.sleep(0.2)
    print()  # newline

# -----------------------------
# CONFIG
# -----------------------------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# -----------------------------
# PHASE 1: Load Tokenizer
# -----------------------------
Log.info("Initializing tokenizer loading...")
try:
    progress("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    Log.success("Tokenizer loaded successfully ✅")
except Exception as e:
    Log.error(f"Tokenizer loading failed ❌: {e}")
    sys.exit(1)

# -----------------------------
# PHASE 2: Determine Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
Log.info(f"Device selected: {device.upper()}")

# -----------------------------
# PHASE 3: Load Model
# -----------------------------
Log.info("Loading model weights (this may take several minutes)...")
try:
    progress("Downloading / Initializing model")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model.to(device)
    Log.success("Model loaded successfully ✅")
except Exception as e:
    Log.error(f"Model loading failed ❌: {e}")
    sys.exit(1)

# -----------------------------
# PHASE 4: Define Generation Function
# -----------------------------
def generate_response(user_input):
    Log.info(f"Generating response for: '{user_input}'")
    try:
        inputs = tokenizer(user_input, return_tensors="pt").to(device)
        start = time.time()
        outputs = model.generate(**inputs, max_new_tokens=100)
        end = time.time()
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        Log.success(f"Response generated in {end - start:.2f} seconds ✅")
        return response
    except Exception as e:
        Log.error(f"Error during generation ❌: {e}")
        return "Error generating response."

# -----------------------------
# PHASE 5: Test Run
# -----------------------------
Log.info("Starting test prompt...")
test_text = "Hello, how are you?"
response = generate_response(test_text)

print("\n==============================")
print(f"User Input   : {test_text}")
print(f"Model Output : {response}")
print("==============================\n")

Log.success("All steps completed successfully 🎉")
