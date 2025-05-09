
import torch
torch.__version__
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm




#hyperparameters
lr = 1e-3
block_size = 8
batch_size = 32
eval_iters = 200
max_iters = 3000
eval_interval = 300
n_embed = 32






device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device : {device}")
# get data
with open("./data/input.txt" , "r" , encoding = "utf-8") as f:
    text = f.read()


#get vocabulary
chars = sorted(list(set(text)))



vocab_size = len(chars)


stoi = {ch : i for i , ch in enumerate(chars)}
itos = {i : ch for i , ch in enumerate(chars)}

encode = lambda s : [stoi[c] for c in s]
decode = lambda l : "".join([itos[i] for i in l])

dataset = torch.tensor(encode(text) , dtype = torch.long)



n = int(0.9 * len(dataset))

train = dataset[:n]
val = dataset[n:]


# Dataloader
torch.manual_seed(42)

def get_batch(split : str):
    data = train if split == 'train' else  val  


    idx = torch.randint(len(data) - block_size , (batch_size , ))


    x = torch.stack( [data[i : i + block_size] for i in idx] )
    y = torch.stack( [data[i + 1 : i + block_size +1 ] for i in idx] )
    # print(idx)
    
    x = x.to(device)
    y = y.to(device)
    return x , y


@torch.no_grad

def estimate_loss():
    
    out = {}

    model.eval()

    for split in [ 'train' , 'val']:

        losses = torch.zeros(eval_iters)
        
        for k in range(eval_iters):
            
            x , y = get_batch(split = split)
            logits , loss = model(x , y)
            
            losses[k] = loss
        
        out[split] = losses.mean()
    
    return out


## single self attention head

class Head(nn.Module):
    
    def __init__(self , head_size):
        super().__init__()
        
        self.key = nn.Linear(n_embed , head_size , bias  = False)
        self.query = nn.Linear(n_embed, head_size , bias  = False)
        self.value = nn.Linear(n_embed , head_size , bias  = False)

        self.register_buffer('tril' , torch.tril(torch.ones(block_size , block_size)) ) 

    def forward(self , x):
        
        B ,T , C =  x.shape      
        
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        wei = q @ k.transpose(-2 , -1) * C** -0.5
        
        wei = wei.masked_fill(self.tril[:T , :T] == 0 , float('-inf'))

        wei = F.softmax(wei , dim =-1)

        out = wei @ v
        
        return out
    
    
    
## Mullti head attention


class MultiHeadAttention(nn.Module):
    
    
    def __init__(self, num_heads , head_size):
        super().__init__()
        
        
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
        
        self.proj = nn.Linear(n_embed , n_embed) #projection 
        
    def forward(self , x):
        out = torch.cat([h(x) for h in self.heads] , dim = -1)
        
        out = self.proj(out)
        
        return out  
    





## FFN

class FeedForwardNN(nn.Module):
    
    def __init__(self, n_embed):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embed , 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed , n_embed) # projection
        )

    def forward(self , x):
        
        return self.net(x)




## LayerNorm



class Layernorm(nn.Module):

    def __init__(self , dim , eps = 1e-5 ):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))   # <-- Fix
        self.beta  = nn.Parameter(torch.zeros(dim))  
        
    def forward(self , x):
        
        xmean = x.mean(dim = -1 , keepdim = True)
        
        xvar = x.var(dim = -1 , keepdim = True)
        
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        
        self.out = self.gamma * xhat + self.beta
        
        return self.out
    
    def parameters(self):
        return [self.gamma , self.beta]
    
    
    

## Block

class Block(nn.Module):
    
    def __init__(self,n_embed , n_head):
        super().__init__()
        
        
        head_size = n_embed // n_head
        
        
        self.sa_heads = MultiHeadAttention(num_heads = n_head ,head_size = head_size)
    
        self.ffn = FeedForwardNN(n_embed)
        
        self.ln1 = Layernorm(n_embed)
        self.ln2 = Layernorm(n_embed)
            
        

    def forward(self , x):
        
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        
        return x

## bigram language model


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size , n_embed) # (V , n_embed) 

        self.position_embedding_table = nn.Embedding(block_size , n_embed) # (T , n_embed)
        
        self.lm_head = nn.Linear(n_embed , vocab_size) # (T , C)
        
        self.blocks = nn.Sequential(
            Block(n_embed , n_head= 4),
            Block(n_embed , n_head= 4),
            Block(n_embed , n_head= 4),
            Layernorm(n_embed),
        )
        

    def forward(self, idx , targets = None):
        """
        idx @ token_embedding_table == (B, t)  @  (c , n) ---> (b , t , n) token_embeds
        
        For every b in the batch and every t in the sequence,
        Get the embedding vector of dimension N corresponding to the idx[b, t]-th row in the table.
        So token_embeds[b, t] gives you a vector of shape (N,).
        
        
        
        position_embedding_table @ arange(T) == (t , n) @ (t) --> (t , n ) pos_embeds 
        x = token_embeds + pos_embed == (B , t , n) + ( t , n) ---> (B , t, , n)  
        
        x @ lm_head  == (B, t, n )  @  (n , c) ---> (b , t , c) logits 
        
        
        
        """
        
        # print(idx)
        B , T  = idx.shape
        
        # print(B , T )
        
        # breakpoint()
        
        tok_embeds = self.token_embedding_table(idx) #(B,T,C)

        # print(f"T : {T}")
        pos_embeds = self.position_embedding_table(torch.arange(T , device = device))



        x = tok_embeds + pos_embeds


        x = self.blocks(x)
        
        
        logits = self.lm_head(x) # (B  , T , Vocab_size)
        
        
        

        if targets is None:
            
            loss =  None

        else:

            B , T , C = logits.shape

            # print(f"B : {B} | T : {T} | C : {C}")

            logits  = logits.view(B*T , C)
            # print(f"logits shape : {logits.shape}")

            targets = targets.view(B*T)
            # print(f"targets shape : {targets.shape}")


            loss = F.cross_entropy(logits , targets)

        
        return logits , loss

    def generate(self , idx , max_new_tokens):

        for _ in range(max_new_tokens):
              
             
            idx_cond = idx[: , -block_size:]    
            
            # print(idx_cond.shape)
            
            logits , loss = self(idx_cond)

            logits = logits[ : , -1 ,:] # (B , C)

            probs = F.softmax(logits , dim = 1)
            probs = torch.clamp(probs, min=1e-9, max=1.0)  # make sure no inf value is passed onto the multimonial func
            
            
            idx_next = torch.multinomial(probs , num_samples = 1)


            idx = torch.cat((idx , idx_next) , dim = 1) # (B , T +1)


        # idx = self.decode(idx)
        return idx

    def decode(self , logits):

        return [decode(i)for i in logits.tolist()]  

model = BigramLanguageModel()
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters() , lr = lr)





for iter in tqdm(range(max_iters)):
    
    
    if iter % eval_interval == 0 :
        losses = estimate_loss()
        
        print(f"step : {iter} | train loss : {losses['train']} | val loss : {losses['val']}")
    
    xb , yb = get_batch('train')

    logits , loss = model(xb , yb)

    optimizer.zero_grad(set_to_none = True)

    loss.backward()

    optimizer.step()



context = torch.zeros((1,1) , dtype = torch.long , device= device)

print(model.decode(model.generate(idx =  context, max_new_tokens = 500 )))




