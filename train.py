import torch
from transformers import AutoTokenizer #no qa
from datasets import load_dataset
from model.model import config, Model # Assuming your model and config are in ./model/model.py

# --- INITIAL SETUP ---
tokenizer = AutoTokenizer.from_pretrained("gpt2")
vocab_size = tokenizer.vocab_size
ds = load_dataset("roneneldan/TinyStories", trust_remote_code=True) # Added trust_remote_code for new datasets versions


# --- DATA LOADING FUNCTIONS ---
def get_data_item(idx, split='train'):
    assert split in ['train','validation'], "Invalid split"
    # ## <-- FIX 1: Changed `idx > 0` to `idx >= 0` to allow the first element (index 0).
    assert idx >= 0 and idx < len(ds[split]), "Index out of range"
    example = ds[split][idx]['text']

    # Your encoding logic is fine, but slightly inefficient. A direct encode is faster.
    encoded = tokenizer.encode(example) + [tokenizer.eos_token_id]
    xs = torch.tensor(encoded[:-1], dtype=torch.long)
    ys = torch.tensor(encoded[1:], dtype=torch.long)
    return xs.view(1, -1), ys.view(1, -1)

def get_batch(split='train', batch_size=32):
    B = batch_size
    T = config.block_size
    x = torch.zeros((B, T), dtype=torch.long)
    y = torch.zeros((B, T), dtype=torch.long)

    for i in range(B):
        idx = torch.randint(0, len(ds[split]), (1,)).item()
        x_, y_ = get_data_item(idx, split)

        # Ensure there's something to sample from
        if x_.shape[1] <= T:
            seq_len = x_.shape[1]
            x[i, :seq_len] = x_
            y[i, :seq_len] = y_
        else:
            start_idx = torch.randint(0, x_.shape[1] - T, (1,)).item()
            x_chunk = x_[:, start_idx:start_idx + T]
            y_chunk = y_[:, start_idx:start_idx + T]
            x[i] = x_chunk
            y[i] = y_chunk

    x, y = x.to(config.device), y.to(config.device)
    return x, y

# --- TRAINING LOOP ---
def train(epochs=10, batch_size=64, lr=3e-4, eval_interval=250, eval_iters=200):
    model = Model(config).to(config.device)
    print(f"Using device: {config.device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # ## <-- IMPROVEMENT 1: Use Automatic Mixed Precision for A100 performance
    scaler = torch.cuda.amp.GradScaler(enabled=(config.device == 'cuda'))

    # ## <-- IMPROVEMENT 2: Track the best validation loss for robust model saving
    best_val_loss = float('inf')

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    print('=====================================================================')

    for epoch in range(epochs):
        model.train()
        x, y = get_batch('train', batch_size=batch_size)

        # ## <-- IMPROVEMENT 1 (cont.): Forward pass with autocast
        with torch.autocast(device_type=config.device, dtype=torch.float16, enabled=(config.device == 'cuda')):
            logits, loss = model(x, targets=y)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if epoch % eval_interval == 0 or epoch == epochs - 1: # Also eval on the last step
            model.eval()
            with torch.no_grad():
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    # Use a larger batch size for validation as we don't store gradients
                    x_val, y_val = get_batch('validation', batch_size=batch_size * 2)
                    _, val_loss = model(x_val, targets=y_val)
                    losses[k] = val_loss.item()

                mean_val_loss = losses.mean()
                print(f"Epoch {epoch}: train loss {loss.item():.4f}, validation loss {mean_val_loss:.4f}")

                # ## <-- IMPROVEMENT 2 (cont.): Save model only if validation loss improves
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
                    torch.save(model.state_dict(), 'best_model.pth')

            print('=====================================================================')

# --- SCRIPT EXECUTION ---
if __name__ == "__main__":
    # ## <-- GPU UTILIZATION FIX: Increased batch_size and epochs, sensible eval_interval
    train(epochs=10000, batch_size=64, lr=3e-4, eval_interval=250, eval_iters=100)
