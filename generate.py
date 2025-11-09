"""
Text generation script for TinyStories model.
Loads a trained model and generates stories from prompts.
"""

import torch
from transformers import GPT2Tokenizer
from model.model import Config, Model
import argparse
from pathlib import Path

def generate_text(
    prompt,
    model_path="results/checkpoints/best_model.pth",
    max_new_tokens=256,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    device=None
):
    """
    Generate text from a prompt using the trained model.
    
    Args:
        prompt: Input text prompt
        model_path: Path to the trained model checkpoint
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        device: Device to run on (None for auto-detection)
    """
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    config = Config()
    config.vocab_size = tokenizer.vocab_size
    model = Model(config).to(device)
    
    # Load model weights
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    print(f"\nPrompt: {prompt}")
    print(f"Prompt tokens: {input_ids.shape[1]}")
    print("\nGenerating text...")
    print("-" * 80)
    
    # Generate
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(generated_text)
    print("-" * 80)
    
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Generate text using TinyStories model")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Input prompt for text generation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="results/checkpoints/best_model.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0.0 to 2.0)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter"
    )
    
    args = parser.parse_args()
    
    generate_text(
        prompt=args.prompt,
        model_path=args.model,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

if __name__ == "__main__":
    main()

