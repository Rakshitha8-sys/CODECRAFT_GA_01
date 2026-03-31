from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token   # Important fix
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load dataset
dataset = load_dataset('text', data_files='data_text.txt')

# Tokenization with labels (IMPORTANT)
def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=50
    )
    tokens["labels"] = tokens["input_ids"].copy()  # 🔥 required for training
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    dataloader_pin_memory=False   # avoids warning
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Train model
trainer.train()

# Generate text
input_text = "Machine learning is"
inputs = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(
    inputs,
    max_length=50,
    num_return_sequences=1
)

print("\nGenerated Text:\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))