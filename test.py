import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from model.model import config, Model

# --- Set device ---
# This ensures the code works on either CPU or GPU without changes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Tokenizer and Data ---
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "once upon a time"
ids = tokenizer.encode(text)

model = Model(config)

# --- Load State Dict with map_location ---
# This is the line that fixes the error.
# It tells PyTorch to load the model weights onto the CPU.
state_dict = torch.load('best_model.pth', map_location=device, weights_only=True)

model.load_state_dict(state_dict)
model.to(device) # Move the model to the selected device
model.eval()

# --- Generation ---
# Convert input ids to a tensor and add a batch dimension [1, sequence_length]
input_ids = torch.tensor([ids], dtype=torch.long, device=device)

# Generate output token IDs
print("Generating text...")
output_ids = model.generate(input_ids, max_new_tokens=100) # Assuming generate takes max_new_tokens

# Decode the generated IDs back to text
# output_ids[0] selects the first (and only) sequence in the batch
generated_text = tokenizer.decode(output_ids[0])

print("\n--- Generated Text ---")
print(generated_text)
