import torch
import numpy as np


class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size, maxlen):
        """
        emb_size - размер эмбеддингов
        maxlen - длинна контекста
        """
        super(PositionalEncoding, self).__init__()

        pos_emb = torch.zeros((maxlen, emb_size))
        t = torch.arange(0, maxlen).unsqueeze(1).float()
        denom = torch.exp(-torch.arange(0, emb_size, 2).float() * (np.log(10000.) / emb_size))
        pos_emb[:, 0::2] = torch.sin(t * denom)
        pos_emb[:, 1::2] = torch.cos(t * denom)
        pos_emb = pos_emb.unsqueeze(0)
        self.register_buffer("pos_emb", pos_emb)

    def forward(self, token_embedding):
        """
        token_embedding - тензор матрицы эмбеддингов
        """
        token_embedding = token_embedding + self.pos_emb[:, : token_embedding.size(1)]
        return token_embedding
