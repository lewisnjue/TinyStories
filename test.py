"""
Simple test script to verify model architecture and basic functionality.
Run this to ensure the model is set up correctly before training.
"""

import torch
from transformers import GPT2Tokenizer
from model.model import Config, Model

def test_model():
    """Test the model architecture and basic forward pass."""
    print("=" * 80)
    print("TinyStories Model Test")
    print("=" * 80)
    
    # Initialize config and model
    config = Config()
    print(f"\nModel Configuration:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Hidden Size: {config.n_embd}")
    print(f"  Attention Heads: {config.n_head}")
    print(f"  Context Length: {config.block_size}")
    print(f"  Vocabulary Size: {config.vocab_size}")
    print(f"  Dropout: {config.dropout}")
    print(f"  Device: {config.device}")
    
    # Create model
    model = Model(config).to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params / 1e6:.2f}M")
    print(f"  Trainable: {trainable_params / 1e6:.2f}M")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 128
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long).to(config.device)
    y = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long).to(config.device)
    
    model.eval()
    with torch.no_grad():
        logits, loss = model(x, targets=y)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Expected loss (random): ~{-torch.log(torch.tensor(1.0 / config.vocab_size)):.4f}")
    
    # Test generation
    print("\nTesting text generation...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
    print(f"  Prompt: '{prompt}'")
    print(f"  Prompt tokens: {input_ids.shape[1]}")
    
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(input_ids, max_new_tokens=20, temperature=1.0)
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"  Generated: '{generated_text}'")
    
    print("\n" + "=" * 80)
    print("All tests passed! Model is ready for training.")
    print("=" * 80)

if __name__ == "__main__":
    test_model()
