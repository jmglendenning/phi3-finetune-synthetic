from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# === Initialize FastAPI ===
app = FastAPI()

# === Paths ===
base_model_path = "/workspace/models/Phi-3-mini-4k-instruct"
adapter_path = "/workspace/phi3-finetune-synthetic/checkpoints"

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# === Load base model ===
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# === Load adapter ===
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# === Input schema ===
class Prompt(BaseModel):
    prompt: str
    max_tokens: int = 256

# === Endpoint ===
@app.post("/generate")
async def generate_text(data: Prompt):
    inputs = tokenizer(data.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=data.max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    # Trim prompt prefix if needed
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": generated}