# rnn_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----- Encoder -----
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        # concat forward + backward
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden.unsqueeze(0)  # outputs: [B, T, 2H], hidden: [1, B, H]

# ----- Attention -----
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        # hidden: [B, 1, H]
        src_len = encoder_outputs.shape[1]
        hidden = hidden.permute(1, 0, 2)  # [B, 1, H]
        hidden = hidden.repeat(1, src_len, 1)  # [B, T, H]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [B, T, H]
        attention = self.v(energy).squeeze(2)  # [B, T]
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        return F.softmax(attention, dim=1)

# ----- Decoder -----
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout=0.3):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((hid_dim * 2) + emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)
        self.attention = Attention(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask=None):
        # input: [B], hidden: [1, B, H]
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))  # [B, 1, E]
        a = self.attention(hidden, encoder_outputs, mask).unsqueeze(1)  # [B, 1, T]
        weighted = torch.bmm(a, encoder_outputs)  # [B, 1, 2H]
        rnn_input = torch.cat((embedded, weighted), dim=2)  # [B, 1, E+2H]
        output, hidden = self.rnn(rnn_input, hidden)  # [B, 1, H]
        output = output.squeeze(1)
        embedded = embedded.squeeze(1)
        weighted = weighted.squeeze(1)
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))  # [B, OutDim]
        return prediction, hidden

# ----- Seq2Seq wrapper -----
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[:,0]  # <sos>

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:,t,:] = output
            top1 = output.argmax(1)
            input = trg[:,t] if torch.rand(1).item() < teacher_forcing_ratio else top1
        return outputs
