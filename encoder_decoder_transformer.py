import torch
import torch.nn as nn # error 1

import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_v):
        super().__init__() #error 2
    
        assert d_model%n_heads ==0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model//n_heads # // for int
        self.d_v = d_v 

        self.W_q = nn.Linear(d_model, d_model, bias = False)
        self.W_k = nn.Linear(d_model, d_model, bias = False)
        self.W_v = nn.Linear(d_model, d_v*n_heads, bias = False)
        self.W_o = nn.Linear(d_v*n_heads, d_model, bias = False)

    def forward(self, q_input, k_input, v_input, mask = None):
        # x: [B, T, d_model]
        B, T_q, _ = q_input.shape
        _, T_k, _ = k_input.shape
        _, T_v,_ = v_input.shape

        assert T_k == T_v

        Q = self.W_q(q_input) #  [B, T, d_model]
        K = self.W_k(k_input)
        V = self.W_v(v_input) # [B,T,d_v*n_heads]


        Q = Q.view(B, T_q, self.n_heads, self.d_head).transpose(1,2) # [B, n_heads, T_q, d_head]
        K = K.view(B, T_k, self.n_heads, self.d_head).transpose(1,2)# [B, n_heads, T_k, d_head]
        V = V.view(B, T_v, self.n_heads, self.d_v).transpose(1,2)# [B, n_heads, T_k, d_v]

        scores = Q@K.transpose(-2,-1)  # [B, n_heads, T_q, d_head] @[B, n_heads, d_head,T_k] -> [B, n_heads, T_q,T_k]
        scores = scores/math.sqrt(self.d_head)

        if mask is not None:
            scores = scores.masked_fill(mask ==0, float("-inf"))

        attention = F.softmax(scores, dim =-1) #[B, n_heads, T_q,T_k]


        # [B, n_heads, T_q,T_k] @ [B, n_heads, T_k, d_v] ->  [B, n_heads, T_q, d_v]
        out = attention @ V
        out = out.transpose(1,2).contiguous() #[B, T_q, n_heads, d_v]
        out = out.view(B,T_q, self.n_heads * self.d_v)

        return self.W_o(out)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True, unbiased = False) # variance unbiased  divided by N

        x_hat = (x - mean)/torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads,d_v, d_ff):
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, d_ff) # default bias= true
        self.fc2 = nn.Linear(d_ff, d_model)

        self.attention = MultiHeadAttention(d_model, n_heads,d_v)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)


    def forward(self, x, mask = None): # x is the embedding input
        attention_out = self.attention(x,x,x,mask)
        x = self.norm1(x + attention_out)

        ffn_out = self.fc2(F.relu(self.fc1(x)))

        x = self.norm2(x + ffn_out)

        return x  #[B,T,d_model]


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads,d_v, d_ff):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

        self.masked_self_attention =MultiHeadAttention(d_model, n_heads, d_v)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, d_v)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, x, encoder_output, tgt_mask =None, src_mask = None):
        self_attention_out = self.masked_self_attention(x,x,x,tgt_mask)
        x = self.norm1(x + self_attention_out)

        cross_attention_out = self.cross_attention(x, encoder_output, encoder_output, src_mask)

        x = self.norm2(x + cross_attention_out)

        ffn_out = self.fc2(F.relu(self.fc1(x)))

        x = self.norm3(x + ffn_out)
        return x 

#sinusoidal postional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1) # [max_len, 1]

        #a^b =exp(bloga)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()/d_model*(-math.log(10000.0))) #[d_model/2,]
    
        pe[:, 0::2] = torch.sin(position*div_term) # [max_len, d_model/2]

        pe[:, 1::2] = torch.cos(position*div_term) 

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe) # pe is unlearnabe parameter

    def forward (self, x): 
        # x : [B, T, d_model]

        T = x.shape[1]

        x = x + self.pe[:, :T]

        return x 



class Encoder(nn.Module):
    def __init__(self, d_model, vocab_size, n_heads, d_v, d_ff, n_layers, max_len = 5000):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.position = PositionalEncoding(d_model, max_len)

        self.layers= nn.ModuleList([EncoderBlock(d_model, n_heads,d_v, d_ff) for _ in range(n_layers)])

    def forward(self, x, src_mask = None):
        # x: [B,T]

        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.position(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x


class Decoder(nn.Module):
    def __init__(self, d_model, vocab_size, n_heads, d_v, d_ff, n_layers, max_len = 5000):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.position = PositionalEncoding(d_model, max_len)

        self.layers= nn.ModuleList([DecoderBlock(d_model, n_heads,d_v, d_ff) for _ in range(n_layers)])

    def forward(self, x, encoder_output, tgt_mask = None, src_mask = None):
        # x: [B,T]

        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.position(x)

        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)

        return x # [B, T , d_model]


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_v, d_ff, n_layers, max_len = 5000):
        super().__init__()

        self.encoder = Encoder(d_model, vocab_size, n_heads, d_v, d_ff, n_layers, max_len)
        self.decoder = Decoder(d_model, vocab_size, n_heads, d_v, d_ff, n_layers, max_len)

        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, encoder_input, decoder_input,  tgt_mask= None,src_mask = None):
        encoder_output = self.encoder(encoder_input, src_mask)
        decoder_output = self.decoder(decoder_input,encoder_output, tgt_mask, src_mask)

        logits =  self.out_proj(decoder_output) # [B,T,d_model] @ [d_model, vocab_size] -> [B,T,vocab_size]
        return logits 


