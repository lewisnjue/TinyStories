import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# Configuration
class Config:
    block_size = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2
    vocab_size = 131072  # Rounded up from 100277 to nearest power of 2 (2^17)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.block_size = config.block_size
        
        # Use PyTorch's MultiheadAttention for fused attention
        self.mha = nn.MultiheadAttention(
            embed_dim=self.n_embd,
            num_heads=self.n_head,
            dropout=self.dropout,
            bias=False,
            batch_first=True
        )
        
        # Projection layer
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Causal mask
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(self.block_size, self.block_size)).bool()
        )

    def forward(self, x):
        B, T, C = x.shape
        
        # Apply causal mask for autoregressive attention
        attn_mask = self.mask[:T, :T]
        
        # Multihead attention
        out, _ = self.mha(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            need_weights=False
        )
        
        # Projection and dropout
        out = self.dropout_layer(self.proj(out))
        return out  # shape -> (B, T, C)

class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ff = FFN(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
    
    def forward(self, x):
        # Residual connections with layer normalization
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.device = config.device
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.n_layer = config.n_layer
        self.dropout = config.dropout
        self.vocab_size = config.vocab_size
        
        # Embeddings
        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        self.dropout_emb = nn.Dropout(self.dropout)
        self.ln_emb = nn.LayerNorm(self.n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(self.n_layer)])
        
        # Final layer norm and head
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape
        device = x.device
        
        # Token and position embeddings
        tok_emb = self.token_embedding_table(x)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        x = self.ln_emb(x)  # Layer norm after embeddings
        x = self.dropout_emb(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and head
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        idx = idx.to(self.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # Crop to block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == "__main__":
    config = Config()
    tokenizer = tiktoken.encoding_for_model('gpt-4')
    m = Model(config).to(config.device)
    x = torch.randint(0, tokenizer.n_vocab, (1, config.block_size), dtype=torch.long).to(config.device)
    logits, loss = m(x, targets=x)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item()}")
    # Expected loss ~ -ln(1/vocab_size) ≈ 11.49 for vocab_size=131072
