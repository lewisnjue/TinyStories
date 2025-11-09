import torch
import torch.nn as nn
import torch.nn.functional as F

# Configuration matching TinyStories-33M architecture
# Based on: "How Small Can Language Models Be and Still Speak Coherent English?"
# by Ronen Eldan and Yuanzhi Li (2023)
class Config:
    block_size = 512  # Context length
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd = 384  # Hidden size
    n_head = 6  # Number of attention heads
    n_layer = 6  # Number of transformer layers
    dropout = 0.1  # Dropout rate (0.1 for training, 0.0 for inference)
    vocab_size = 50257  # GPT-2 tokenizer vocabulary size
    bias = False  # No bias in attention and MLP layers (following GPT-2)

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
        
        # Projection layer (no bias as per GPT-2 style)
        self.proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
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
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),  # GELU activation as per TinyStories paper
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
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
        # Weight tying: share weights between token embedding and output projection
        # This reduces parameters and improves performance
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 style initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

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
        # Weight tying: use token embedding weights for output projection
        logits = F.linear(x, self.token_embedding_table.weight)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Generate text using the model.
        
        Args:
            idx: Input token indices (B, T)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random, 1.0 = default)
            top_k: Top-k sampling (keep only top k tokens)
            top_p: Nucleus sampling (keep tokens with cumulative probability <= top_p)
        """
        idx = idx.to(self.device)
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop to block_size if needed
                idx_cond = idx[:, -self.block_size:] if idx.size(1) > self.block_size else idx
                
                # Forward pass
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature  # (B, vocab_size)
                
                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')
                
                # Sample from the distribution
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

if __name__ == "__main__":
    config = Config()
    m = Model(config).to(config.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in m.parameters())
    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Test forward pass
    x = torch.randint(0, config.vocab_size, (1, config.block_size), dtype=torch.long).to(config.device)
    logits, loss = m(x, targets=x)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    # Expected loss ~ -ln(1/vocab_size) ≈ 10.82 for vocab_size=50257
