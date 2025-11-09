"""
Evaluation script for TinyStories model.
Computes perplexity and other metrics on validation/test sets.
"""

import torch
from transformers import GPT2Tokenizer
from datasets import load_dataset
from model.model import Config, Model
from pathlib import Path
import argparse
import json

def evaluate_model(
    model_path,
    dataset_split="validation",
    batch_size=32,
    max_samples=None,
    device=None
):
    """
    Evaluate model on dataset.
    
    Args:
        model_path: Path to model checkpoint
        dataset_split: Dataset split to evaluate on
        batch_size: Batch size for evaluation
        max_samples: Maximum number of samples to evaluate (None for all)
        device: Device to run on
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
    
    # Load dataset
    print(f"\nLoading {dataset_split} dataset...")
    ds = load_dataset("roneneldan/TinyStories", split="all")
    
    # Split dataset
    total_size = len(ds)
    if dataset_split == "validation":
        start_idx = int(0.95 * total_size)
        eval_ds = ds.select(range(start_idx, total_size))
    elif dataset_split == "train":
        end_idx = int(0.95 * total_size)
        eval_ds = ds.select(range(end_idx))
    else:
        eval_ds = ds
    
    if max_samples is not None:
        eval_ds = eval_ds.select(range(min(max_samples, len(eval_ds))))
    
    print(f"Evaluating on {len(eval_ds)} examples...")
    
    # Tokenize and prepare data
    all_tokens = []
    for example in eval_ds:
        tokens = tokenizer.encode(example['text'], add_special_tokens=True)
        all_tokens.extend(tokens)
    
    data = torch.tensor(all_tokens, dtype=torch.long)
    print(f"Total tokens: {len(data):,}")
    
    # Evaluation
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    block_size = config.block_size
    num_eval_batches = (len(data) - 1) // (batch_size * block_size)
    
    print(f"\nEvaluating {num_eval_batches} batches...")
    
    with torch.no_grad():
        for batch_idx in range(num_eval_batches):
            start = batch_idx * batch_size * block_size
            end = start + batch_size * block_size + 1
            
            if end > len(data):
                break
            
            chunk = data[start:end]
            x = chunk[:-1].view(batch_size, block_size).to(device)
            y = chunk[1:].view(batch_size, block_size).to(device)
            
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float32):
                _, loss = model(x, targets=y)
            
            total_loss += loss.item() * (batch_size * block_size)
            total_tokens += batch_size * block_size
            num_batches += 1
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1}/{num_eval_batches} batches...")
    
    # Compute metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    results = {
        'dataset_split': dataset_split,
        'num_examples': len(eval_ds),
        'num_tokens': total_tokens,
        'num_batches': num_batches,
        'average_loss': avg_loss,
        'perplexity': perplexity
    }
    
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Dataset: {dataset_split}")
    print(f"Examples evaluated: {len(eval_ds):,}")
    print(f"Tokens evaluated: {total_tokens:,}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print("=" * 80)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyStories model")
    parser.add_argument(
        "--model",
        type=str,
        default="results/checkpoints/best_model.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "all"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results JSON"
    )
    
    args = parser.parse_args()
    
    results = evaluate_model(
        model_path=args.model,
        dataset_split=args.split,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Save results if output path specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()

