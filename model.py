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