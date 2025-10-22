from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Optional, List

# === Initialize FastAPI ===
app = FastAPI()

# === Ping Endpoint ===
@app.get("/ping")
async def ping():
    return {"status": "ready", "model": "phi3-mini"}

# === Paths ===
base_model_path = "/workspace/models/Phi-3-mini-4k-instruct"
adapter_path = "/workspace/phi3-finetune-synthetic/checkpoints"

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(adapter_path)

# === Load base model + adapter ===
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# === Request Schema ===
class Prompt(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = False
    stop: Optional[List[str]] = None
    
# === /generate Endpoint ===
@app.post("/generate")
async def generate_text(data: Prompt):
    inputs = tokenizer(data.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():

        stop_sequences = ["]", "}"]  # optional stop tokens if your tokenizer supports it

        output = model.generate(
            **inputs,
            max_new_tokens=data.max_tokens,
            do_sample=data.do_sample,
            temperature=data.temperature,
            top_p=data.top_p,
            eos_token_id=tokenizer.eos_token_id,  # ensures it can stop early
            pad_token_id=tokenizer.eos_token_id   # avoids pad-token warnings
        )

    # === Decode full output ===
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # === Stop token simulation (manual truncation) ===
    if data.stop:
        for token in data.stop:
            if token in decoded:
                decoded = decoded.split(token)[0]
                break

    return {"response": decoded}