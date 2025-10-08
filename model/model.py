import torch
from transformers import AutoTokenizer #no qa
tokenizer = AutoTokenizer.from_pretrained("gpt2") #bpe tokenizer

class config():
    block_size = 512
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2
    vocab_size = tokenizer.vocab_size




class Head(torch.nn.Module):
    def __init__(self,config:config):
        super().__init__()
        self.n_emd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.key = torch.nn.Linear(self.n_emd,self.head_size,bias=False)
        self.query = torch.nn.Linear(self.n_emd,self.head_size,bias=False)
        self.value = torch.nn.Linear(self.n_emd,self.head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(config.block_size,config.block_size)))
        self.dropout = torch.nn.Dropout(self.dropout)

    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei = torch.nn.functional.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(torch.nn.Module):
    def __init__(self,config:config):
        super().__init__()
        self.n_head = config.n_head
        self.heads = torch.nn.ModuleList([Head(config) for _ in range(self.n_head)])
        self.proj = torch.nn.Linear(config.n_embd,config.n_embd)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out # shape -> (B,T,C)

class FFN(torch.nn.Module):
    def __init__(self,config:config):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(config.n_embd,4*config.n_embd),
            torch.nn.ReLU(),
            torch.nn.Linear(4*config.n_embd,config.n_embd),
            torch.nn.Dropout(config.dropout)
        )

    def forward(self,x):
        return self.net(x)

class Model(torch.nn.Module):

    def __init__(self,config:config):
        super().__init__()
        self.block_size = config.block_size
        self.device = config.device
        self.n_emd = config.n_embd
        self.n_head = config.n_head
        self.n_layer = config.n_layer
        self.dropout = config.dropout
        self.vocab_size = config.vocab_size
        self.token_embedding_table = torch.nn.Embedding(self.vocab_size,self.n_emd)
        self.position_embedding_table = torch.nn.Embedding(self.block_size,self.n_emd)
        self.sa = MultiHeadAttention(config)
        self.ln_f = torch.nn.LayerNorm(self.n_emd)
        self.ff = FFN(config)
        self.lm_head = torch.nn.Linear(self.n_emd,self.vocab_size)

    def forward(self,x,targets=None):
        B,T = x.shape
        tok_emb = self.token_embedding_table(x)
        pos_emb = self.position_embedding_table(torch.arange(T,device=self.device))
        x = tok_emb + pos_emb
        x = self.ln_f(x) # layer norm before self-attention
        x = x + self.sa(x)  # residual connection
        x = self.ln_f(x)
        x = x + self.ff(x)  # residual connection
        logits = self.lm_head(x) # i have my logits

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits,targets)

        return logits,loss


    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-self.block_size:]
            logits,loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = torch.nn.functional.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx



if __name__ == "__main__":
    m = Model(config()).to(config().device)
    x = torch.randint(0,config().vocab_size,(1,config().block_size),dtype=torch.long).to(config().device)
    logits,loss = m(x,targets=x)
    print(logits.shape)
    print(loss)  # am expecting loss of  -(1/vocab_size).log() which is around 10.83 and am getting 10.94 which is close enough
