import torch
import torch.nn as nn
import math

# Transformer les inputs en embedding
class InputEmbedding(nn.models):
    def __init__(self, d_model : int, vocab_size : int):
        super().__init_()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEnconding(nn.Module):
    def __init__(self, d_model : int, seq_len : int, dropout : float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Créer une matrice de taille (seq_len, d_model)
        pe = torch_zeros(seq_len, d_model)

        # Créer un vecteur de taille (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Appliquer sin et cos pour chaque position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)