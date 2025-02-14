import os
import random
import pandas as pd
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(42)
        

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, config.mlp_expansion_ratio * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config.mlp_expansion_ratio * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    
class FeedForwardClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, config.hidden_size, bias=config.bias)
        self.relu    = nn.ReLU()
        self.c_proj  = nn.Linear(config.hidden_size, config.num_classes, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        return x
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.config=config
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # regularization
        self.dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.p_dropout = nn.Dropout(config.dropout)
        self.ln=LayerNorm(config.n_embd, bias=config.bias)


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Self-attention; (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        y=self.p_dropout(self.proj(y))
        
        return y, att
    
class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x):
        out = self.ln_1(x)
        out, att = self.attn(out)
        x = x + out
        x = x + self.mlp(self.ln_2(x))
        return x, att
    
class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_channels = config.num_channels
        self.n_embd = config.n_embd
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(self.num_channels, self.n_embd, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, n_embd)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.n_embd)) 

        pe = torch.zeros(config.num_patches+1, config.n_embd)

        for pos in range(config.num_patches+1):
            for i in range(config.n_embd):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos/(10000 ** (i/config.n_embd)))
                else:
                    pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/config.n_embd)))

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Expand to have class token for every image in batch
        tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)

        # Adding class tokens to the beginning of each embedding
        x = torch.cat((tokens_batch,x), dim=1)

        # Add positional encoding to embeddings
        x = x + self.pe

        return x


class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0
        assert config.image_size % config.patch_size == 0, "img_size dimensions must be divisible by patch_size dimensions"

        self.pos_embedding = PositionalEncoding(config)
        self.patch_embedding = PatchEmbeddings(config)
        self.transformer = nn.ModuleDict(dict(
            wte = self.patch_embedding,
            wpe = self.pos_embedding,
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.classifier = FeedForwardClassifier(config)
        self.softmax = nn.Softmax(dim=-1)
        print("Number of parameters: %.2fk" % (self.get_num_params()/1e3,))      


    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return n_params

    def forward(self, in_data):
        all_attentions = []
        tok_emb = self.transformer.wte(in_data) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.wpe(tok_emb) # Added position embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x, att = block(x)
            all_attentions.append(att)
        x = self.transformer.ln_f(x) #(b,t,n_embd)
        x = x[:,0,:] # Taking the first token as the class token
        logits = self.classifier(x)
        logits=self.softmax(logits)
        return logits, all_attentions
        
    
