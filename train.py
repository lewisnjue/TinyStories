import torch 
from transformers import AutoTokenizer #no qa 
tokenizer = AutoTokenizer.from_pretrained("gpt2") #bpe tokenizer 
from datasets import load_dataset
from model.model import config, Model


vocab_size = tokenizer.vocab_size 
ds = load_dataset("roneneldan/TinyStories")
tokenizer = AutoTokenizer.from_pretrained("gpt2")



def get_data_item(idx,split='train'): 
    assert split in ['train','validation'] ,"Invalid split"
    assert idx > 0 and idx < len(ds[split]),"Index out of range"
    example = ds[split][idx]['text']
    xs , ys = [], [] 
    for x,y in zip(tokenizer.encode(example),tokenizer.encode(example)[1:]+[tokenizer.eos_token_id]):
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs).view(1,-1),torch.tensor(ys).view(1,-1)  




def get_batch(split='train',batch_size=32):
    B = batch_size
    T = config.block_size 
    x = torch.zeros((B,T),dtype=torch.long)
    y = torch.zeros((B,T),dtype=torch.long)
    for i in range(B):
        idx = torch.randint(0,len(ds[split]),(1,)).item()
        x_,y_ = get_data_item(idx,split)
        if x_.shape[1] > T:
            start_idx = torch.randint(0,x_.shape[1]-T,(1,)).item()
            x_ = x_[:,start_idx:start_idx+T]
            y_ = y_[:,start_idx:start_idx+T]
        x[i,:x_.shape[1]] = x_
        y[i,:y_.shape[1]] = y_
    x,y = x.to(config.device),y.to(config.device)
    return x,y



training_losses = []
validation_losses = [] 

def train(epochs = 10,batch_size=32, lr=3e-4, eval_interval=200, eval_iters=200):
    model = Model(config).to(config.device)   
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr) 
    print("Nuber of parameters:",sum(p.numel() for p in model.parameters())/1e6,"M")
    print('=====================================================================')

    for epoch in range(epochs):
        model.train() 
        x,y = get_batch('train',batch_size) 
        logits,loss = model(x,targets=y) 
        optimizer.zero_grad(set_to_none=True) 
        loss.backward() 
        optimizer.step() 

        if epoch % eval_interval == 0:
            model.eval() 
            with torch.no_grad():
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    x,y = get_batch('validation') 
                    logits,loss = model(x,targets=y) 
                    losses[k] = loss.item()
                print(f"Epoch {epoch}: train loss {loss.item():.4f}, validation loss {losses.mean():.4f}")
                # i will also want to print the norm of gradients 
                print("Gradient norms:",[p.grad.norm().item() for p in model.parameters() if p.grad is not None])
                training_losses.append(loss.item())
                validation_losses.append(losses.mean().item()) 
                # compare the last two validation losses and save the model if the latest one is lower
                if len(validation_losses) > 1 and validation_losses[-1] < validation_losses[-2]:
                    torch.save(model.state_dict(),'model.pth')
                
            print('=====================================================================')





if __name__ == "__main__":
    train(epochs=4,eval_interval=1)