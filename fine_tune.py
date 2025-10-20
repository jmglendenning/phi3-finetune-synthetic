from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import os

model_name = "./models/phi-3-instruct"
output_dir = "./checkpoints/phi3-ft-2025-10"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

dataset = load_dataset("json", data_files="./train_data/2025-10-voter_profiles.jsonl")

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=100,
    save_total_limit=2,
    logging_steps=20,
    fp16=True,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="text"
)

trainer.train()