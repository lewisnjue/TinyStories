"""
Training script for TinyStories model.
Based on the paper: "How Small Can Language Models Be and Still Speak Coherent English?"
by Ronen Eldan and Yuanzhi Li (2023)
"""

import torch
import torch.nn as nn
from transformers import GPT2Tokenizer
from datasets import load_dataset
from model.model import Config, Model
import time
import os
import json
from datetime import datetime
from pathlib import Path

# Create results directory
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
checkpoints_dir = results_dir / "checkpoints"
checkpoints_dir.mkdir(exist_ok=True)
logs_dir = results_dir / "logs"
logs_dir.mkdir(exist_ok=True)

# Initialize tokenizer (GPT-2 tokenizer as per TinyStories paper)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Model configuration
config = Config()
config.vocab_size = tokenizer.vocab_size  # 50257

print("=" * 80)
print("TinyStories Model Training")
print("=" * 80)
print(f"Model Architecture:")
print(f"  - Layers: {config.n_layer}")
print(f"  - Hidden Size: {config.n_embd}")
print(f"  - Attention Heads: {config.n_head}")
print(f"  - Context Length: {config.block_size}")
print(f"  - Vocabulary Size: {config.vocab_size}")
print(f"  - Dropout: {config.dropout}")
print("=" * 80)

# Load TinyStories dataset
print("\nLoading TinyStories dataset...")
ds = load_dataset("roneneldan/TinyStories", split="all")

def preprocess(dataset, split_name):
    """Preprocess dataset by tokenizing and concatenating all texts."""
    data = []
    print(f"Processing {split_name} split...")
    for idx, example in enumerate(dataset):
        if idx % 10000 == 0:
            print(f"  Processed {idx} examples...")
        text = example['text']
        # Tokenize and add EOS token
        encoded = tokenizer.encode(text, add_special_tokens=True)
        data.extend(encoded)
    print(f"  Total tokens in {split_name}: {len(data):,}")
    return torch.tensor(data, dtype=torch.long)

# Split dataset into train and validation
print("\nSplitting dataset...")
total_size = len(ds)
train_size = int(0.95 * total_size)
val_size = total_size - train_size

train_ds = ds.select(range(train_size))
val_ds = ds.select(range(train_size, total_size))

print(f"Train examples: {len(train_ds):,}")
print(f"Validation examples: {len(val_ds):,}")

# Preprocess data
train_data = preprocess(train_ds, "train")
val_data = preprocess(val_ds, "validation")

def get_batch(data, batch_size, block_size, device):
    """Get a batch of data for training."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

def estimate_loss(model, data, eval_iters, batch_size, block_size, device):
    """Estimate loss on validation set."""
    model.eval()
    losses = torch.zeros(eval_iters)
    with torch.no_grad():
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16 if device == 'cuda' else torch.float32):
                _, loss = model(X, targets=Y)
            losses[k] = loss.item()
    model.train()
    return losses.mean().item()

def train(
    epochs=20,
    batch_size=32,
    learning_rate=1e-4,
    weight_decay=0.1,
    eval_interval=500,
    eval_iters=200,
    gradient_accumulation_steps=32,
    max_grad_norm=0.5,
    warmup_steps=1000,
    save_interval=5000,
):
    """
    Train the TinyStories model.
    
    Hyperparameters based on TinyStories paper:
    - Learning rate: 1e-4
    - Weight decay: 0.1
    - Batch size: 32
    - Gradient accumulation: 32 (effective batch size: 1024)
    - Gradient clipping: 0.5
    - Warmup steps: 1000
    """
    model = Model(config).to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params / 1e6:.2f}M")
    print(f"  Trainable: {trainable_params / 1e6:.2f}M")
    
    # Optimizer (AdamW as per paper)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-9
    )
    
    # Learning rate scheduler (linear warmup + cosine annealing)
    def get_lr(it):
        if it < warmup_steps:
            return learning_rate * (it / warmup_steps)
        else:
            # Cosine annealing
            progress = (it - warmup_steps) / max(1, (max_iters - warmup_steps))
            return learning_rate * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    # Mixed precision training
    if config.device == 'cuda' and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        use_scaler = False
        print("Using bfloat16 mixed precision (no scaler needed).")
    else:
        dtype = torch.float16
        use_scaler = True
        print("Using float16 mixed precision with scaler.")
    
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    
    # Training setup
    max_iters = len(train_data) // (batch_size * config.block_size * gradient_accumulation_steps) * epochs
    print(f"\nTraining Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max iterations: {max_iters:,}")
    print(f"  Warmup steps: {warmup_steps}")
    print("=" * 80)
    
    # Checkpointing
    best_val_loss = float('inf')
    start_iter = 0
    checkpoint_path = checkpoints_dir / "checkpoint.pth"
    best_model_path = checkpoints_dir / "best_model.pth"
    
    if checkpoint_path.exists():
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint.get('iter', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if use_scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Resuming from iteration {start_iter} with best val loss {best_val_loss:.4f}")
    
    # Compile model for faster training (PyTorch 2.0+)
    try:
        model = torch.compile(model)
        print("Model compiled for faster training.")
    except Exception as e:
        print(f"Model compilation not available: {e}")
    
    # Training log
    log_file = logs_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    training_log = []
    
    print("\nStarting training...")
    print("=" * 80)
    
    model.train()
    iter_num = start_iter
    accum_loss = 0.0
    accum_count = 0
    current_lr = learning_rate  # Initialize learning rate
    
    while iter_num < max_iters:
        # Get batch
        X, Y = get_batch(train_data, batch_size, config.block_size, config.device)
        
        # Forward pass with mixed precision
        with torch.amp.autocast(device_type=config.device, dtype=dtype):
            _, loss = model(X, targets=Y)
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        accum_loss += loss.item() * gradient_accumulation_steps
        accum_count += 1
        
        # Update weights after accumulation
        if (iter_num + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            if use_scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            # Learning rate scheduling
            current_lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            optimizer.zero_grad(set_to_none=True)
        
        # Evaluation and logging
        if (iter_num + 1) % eval_interval == 0:
            train_loss = accum_loss / accum_count if accum_count > 0 else 0.0
            val_loss = estimate_loss(model, val_data, eval_iters, batch_size, config.block_size, config.device)
            
            # Get current learning rate
            eval_lr = current_lr if (iter_num + 1) % gradient_accumulation_steps == 0 else optimizer.param_groups[0]['lr']
            
            log_entry = {
                'iter': iter_num + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': eval_lr,
                'time': datetime.now().isoformat()
            }
            training_log.append(log_entry)
            
            print(f"Iter {iter_num + 1:6d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {log_entry['lr']:.2e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  -> New best validation loss: {best_val_loss:.4f}. Saving model...")
                torch.save(model.state_dict(), best_model_path)
            
            # Reset accumulation
            accum_loss = 0.0
            accum_count = 0
        
        # Save checkpoint
        if (iter_num + 1) % save_interval == 0:
            checkpoint = {
                'iter': iter_num + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config.__dict__,
                'scaler_state_dict': scaler.state_dict() if use_scaler else None
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  -> Checkpoint saved at iteration {iter_num + 1}")
        
        iter_num += 1
    
    # Final save
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved in: {results_dir}")
    print(f"  - Best model: {best_model_path}")
    print(f"  - Checkpoint: {checkpoint_path}")
    print(f"  - Training log: {log_file}")
    
    # Save training log
    with open(log_file, 'w') as f:
        json.dump(training_log, f, indent=2)

if __name__ == "__main__":
    train(
        epochs=20,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=0.1,
        eval_interval=500,
        eval_iters=200,
        gradient_accumulation_steps=32,
        max_grad_norm=0.5,
        warmup_steps=1000,
        save_interval=5000,
    )
