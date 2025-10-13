import torch
import tiktoken
from datasets import load_dataset
from model.model import Config, Model
import time
import torch.amp
import os

tokenizer = tiktoken.encoding_for_model('gpt-4')
config = Config()
config.vocab_size = tokenizer.n_vocab

ds = load_dataset("roneneldan/TinyStories")

def preprocess(split):
    data = []
    for example in ds[split]:
        encoded = tokenizer.encode(example['text']) + [tokenizer.eot_token]
        data.extend(encoded)
    return torch.tensor(data, dtype=torch.long)

train_data = preprocess('train')
val_data = preprocess('validation')

def train(epochs=10, batch_size=64, lr=3e-4, eval_interval=250, target_tokens_per_update=1500000):
    model = Model(config).to(config.device)
    
    print(f"Using device: {config.device}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    if config.device == 'cuda' and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        use_scaler = False
        print("Using bfloat16 mixed precision without scaler.")
    else:
        dtype = torch.float16
        use_scaler = True
        print("Falling back to float16 mixed precision with scaler.")
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    
    tokens_per_step = batch_size * config.block_size
    accum_steps = max(1, round(target_tokens_per_update / tokens_per_step))
    print(f"Number of accumulation steps before updating weights: {accum_steps}")
    
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    print('=====================================================================')
    
    num_batches = (len(train_data) - 1) // (batch_size * config.block_size)
    
    eval_batch_size = batch_size * 4
    val_num_batches = (len(val_data) - 1) // (eval_batch_size * config.block_size)
    
    best_val_loss = float('inf')
    start_epoch = 0
    checkpoint_path = 'checkpoint.pth'
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        if use_scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Resuming training from epoch {start_epoch} with best val loss {best_val_loss:.4f}")
    
    model = torch.compile(model)
    
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        
        model.train()
        sum_train_loss = 0.0
        accum_count = 0
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size * config.block_size
            end = start + batch_size * config.block_size + 1
            chunk = train_data[start:end]
            x = chunk[:-1].view(batch_size, config.block_size).to(config.device)
            y = chunk[1:].view(batch_size, config.block_size).to(config.device)
            
            with torch.amp.autocast(device_type=config.device, dtype=dtype):
                logits, loss = model(x, targets=y)
            
            sum_train_loss += loss.item()
            
            if use_scaler:
                scaler.scale(loss / accum_steps).backward()
            else:
                (loss / accum_steps).backward()
            
            accum_count += 1
            
            if accum_count % accum_steps == 0:
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
        
        if accum_count > 0:
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        avg_train_loss = sum_train_loss / num_batches
        
        end_time = time.time()
        duration = end_time - start_time
        tokens_processed = num_batches * batch_size * config.block_size
        tps = tokens_processed / duration
        
        val_loss = estimate_loss(model, val_data, eval_batch_size, config.block_size, config.device)
        
        print(f"Epoch {epoch}: train loss {avg_train_loss:.4f}, validation loss {val_loss:.4f}, tokens per second {tps:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best validation loss: {best_val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), 'best_model.pth')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'scaler_state_dict': scaler.state_dict() if use_scaler else None
        }, checkpoint_path)
        
        print('=====================================================================')

def estimate_loss(model, data, batch_size, block_size, device):
    sum_loss = 0.0
    total_tokens = 0
    num_batches = (len(data) - 1) // (batch_size * block_size)
    
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size * block_size
        end = start + batch_size * block_size + 1
        chunk = data[start:end]
        x = chunk[:-1].view(batch_size, block_size).to(device)
        y = chunk[1:].view(batch_size, block_size).to(device)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                _, loss = model(x, targets=y)
        
        sum_loss += loss.item() * (batch_size * block_size)
        total_tokens += batch_size * block_size
    
    if total_tokens == 0:
        return float('inf')
    
    return sum_loss / total_tokens

if __name__ == "__main__":
    train(epochs=10000, batch_size=64, lr=3e-4, eval_interval=250, target_tokens_per_update=1500000)
