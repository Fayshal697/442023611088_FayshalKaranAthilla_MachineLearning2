import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)  # <── nama persis seperti checkpoint

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size=256, num_heads=8, num_layers=3, dropout=0.3):
        super().__init__()
        self.src_tok_emb = nn.Embedding(src_vocab_size, embed_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size)

        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=512,   # <── harus sama seperti training
            dropout=dropout,
            batch_first=True
        )

        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)

    def forward(self, src, tgt):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        out = self.transformer(src_emb, tgt_emb)
        return self.fc_out(out)

    def encode(self, src):
        src_seq_len = src.shape[1]
        src_positions = torch.arange(0, src_seq_len, device=src.device).unsqueeze(0)
        src_emb = self.dropout(self.src_tok_emb(src) + self.positional_encoding(src_positions))
        return self.encoder(src_emb)

    def decode(self, memory, tgt):
        tgt_seq_len = tgt.shape[1]
        tgt_positions = torch.arange(0, tgt_seq_len, device=tgt.device).unsqueeze(0)
        tgt_emb = self.dropout(self.tgt_tok_emb(tgt) + self.positional_encoding(tgt_positions))
        out = self.decoder(tgt_emb, memory)
        return self.fc_out(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
