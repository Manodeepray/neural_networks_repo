import torch
import torch.nn as nn
import torch.nn.functional as F




class SelfAttention(nn.Module):
    
    def __init__(self , d_model = 2,
                row_dim = 0,
                col_dim = 1):

        super().__init__()

        # weight matrices
        
        self.W_q = nn.Linear(in_features = d_model,
                            out_features = d_model,
                            bias = False) 
        self.W_k = nn.Linear(in_features = d_model,
                            out_features = d_model,
                            bias = False)
        
        self.W_v = nn.Linear(in_features = d_model,
                            out_features = d_model,
                            bias = False)

        self.row_dim = row_dim
        self.col_dim = col_dim
        self.d_model = d_model


    
    def forward(self , token_encodings):
        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)
        
        sims = torch.matmul(q , k.transpose(dim0 = self.row_dim,
                                           dim1 = self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)
        # scaled_sims = sims / torch.tensor(self.d_model**0.5)

        attention_percents = F.softmax(scaled_sims , dim = self.col_dim)
        attention_scores = torch.matmul(attention_percents , v)

        return attention_scores

    
    
    
    
    ################################## Masked SA ##############################################
    
    
    
    
    
    
    
        
class MaskedSelfAttention(nn.Module):
    
    def __init__(self , d_model = 2,
                row_dim = 0,
                col_dim = 1):

        super().__init__()

        # weight matrices
        
        self.W_q = nn.Linear(in_features = d_model,
                            out_features = d_model,
                            bias = False) 
        self.W_k = nn.Linear(in_features = d_model,
                            out_features = d_model,
                            bias = False)
        
        self.W_v = nn.Linear(in_features = d_model,
                            out_features = d_model,
                            bias = False)
        

        self.row_dim = row_dim
        self.col_dim = col_dim
        self.d_model = d_model
        

    
    def forward(self , token_encodings , mask = None):
        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)
        
        sims = torch.matmul(q , k.transpose(dim0 = self.row_dim,
                                           dim1 = self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)
        # scaled_sims = sims / torch.tensor(self.d_model**0.5)

        if mask is not None:
            # mask = torch.tril(torch.ones(k.size(self.row_dim),k.size(self.row_dim)))
            # mask = mask == 0
            scaled_sims = scaled_sims.masked_fill(mask = mask,
                                                 value = -1e-9)       

        
     

        
        attention_percents = F.softmax(scaled_sims , dim = self.col_dim)
        attention_scores = torch.matmul(attention_percents , v)
        
        return attention_scores

    
        






########################### MHSA ####################################




class Attention(nn.Module):
    
    def __init__(self ,
                 d_model = 2,
                 row_dim = 0,
                 col_dim = 1):

        super().__init__()

        # weight matrices
        
        self.W_q = nn.Linear(in_features = d_model,
                            out_features = d_model,
                            bias = False) 
        self.W_k = nn.Linear(in_features = d_model,
                            out_features = d_model,
                            bias = False)
        
        self.W_v = nn.Linear(in_features = d_model,
                            out_features = d_model,
                            bias = False)

        self.row_dim = row_dim
        self.col_dim = col_dim
        self.d_model = d_model


    
    def forward(self , 
                encodings_q ,
                encodings_k ,
                encodings_v,
               mask = None):
        
        q = self.W_q(encodings_q)
        k = self.W_k(encodings_k)
        v = self.W_v(encodings_v)
        
        sims = torch.matmul(q , k.transpose(dim0 = self.row_dim,
                                           dim1 = self.col_dim))
        
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)
        # scaled_sims = sims / torch.tensor(self.d_model**0.5)

        if mask is not None:
            # mask = torch.tril(torch.ones(k.size(self.row_dim),k.size(self.row_dim)))
            # mask = mask == 0
            scaled_sims = scaled_sims.masked_fill(mask = mask,
                                                 value = -1e-9)       
       
        attention_percents = F.softmax(scaled_sims , dim = self.col_dim)
        attention_scores = torch.matmul(attention_percents , v)

        return attention_scores

    
class MultiHeadAttention(nn.Module):
    
    def __init__(self ,
                 d_model = 2,
                 row_dim = 0,
                 col_dim = 1,
                num_heads = 1):
        super().__init__()


        self.heads = nn.ModuleList(
            [Attention(d_model = 2,row_dim = 0,col_dim = 1) 
             for _ in range(num_heads)]
        )

        self.col_dim = col_dim


    def forward(self , 
                encodings_q ,
                encodings_k ,
                encodings_v):

        return torch.cat(
            [head(encodings_q ,
                encodings_k ,
                encodings_v)
             for head in self.heads] , dim = self.col_dim
        )