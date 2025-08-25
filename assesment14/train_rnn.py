# train_rnn.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rnn_attention import Encoder, Decoder, Seq2Seq
import sentencepiece as spm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Dataset -----
class TranslationDataset(Dataset):
    def __init__(self, path, sp_src, sp_tgt, max_len=50):
        self.pairs = []
        self.sp_src = sp_src
        self.sp_tgt = sp_tgt
        self.max_len = max_len

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                src, tgt = line.strip().split("\t")
                src_ids = [sp_src.bos_id()] + sp_src.encode(src, out_type=int)[:max_len-2] + [sp_src.eos_id()]
                tgt_ids = [sp_tgt.bos_id()] + sp_tgt.encode(tgt, out_type=int)[:max_len-2] + [sp_tgt.eos_id()]
                self.pairs.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(s) for s in src_batch]
    tgt_lens = [len(t) for t in tgt_batch]

    src_max = max(src_lens)
    tgt_max = max(tgt_lens)

    src_pad = torch.zeros(len(batch), src_max, dtype=torch.long)
    tgt_pad = torch.zeros(len(batch), tgt_max, dtype=torch.long)

    for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
        src_pad[i, :len(src)] = torch.tensor(src)
        tgt_pad[i, :len(tgt)] = torch.tensor(tgt)

    return src_pad, tgt_pad

# ----- Training -----
def train_epoch(model, iterator, optimizer, criterion, clip=1):
    model.train()
    epoch_loss = 0

    for src, tgt in iterator:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        optimizer.zero_grad()
        output = model(src, tgt)

        # ignore first token (<sos>)
        output_dim = output.shape[-1]
        output = output[:,1:,:].reshape(-1, output_dim)
        tgt = tgt[:,1:].reshape(-1)

        loss = criterion(output, tgt)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, tgt in iterator:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            output = model(src, tgt, teacher_forcing_ratio=0)  # no teacher forcing
            output_dim = output.shape[-1]
            output = output[:,1:,:].reshape(-1, output_dim)
            tgt = tgt[:,1:].reshape(-1)
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# ----- Main -----
if __name__ == "__main__":
    data_dir = "data_splits"
    train_path = os.path.join(data_dir, "train.tok.tsv")
    valid_path = os.path.join(data_dir, "valid.tok.tsv")

    sp_src = spm.SentencePieceProcessor(model_file=os.path.join(data_dir, "spm_src.model"))
    sp_tgt = spm.SentencePieceProcessor(model_file=os.path.join(data_dir, "spm_tgt.model"))

    train_ds = TranslationDataset(train_path, sp_src, sp_tgt)
    valid_ds = TranslationDataset(valid_path, sp_src, sp_tgt)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    INPUT_DIM = sp_src.vocab_size()
    OUTPUT_DIM = sp_tgt.vocab_size()
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    DROPOUT = 0.3
    N_EPOCHS = 10

    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, DROPOUT)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DROPOUT)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_valid_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    train_losses, valid_losses = [], []

    for epoch in range(N_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        valid_loss = evaluate(model, valid_loader, criterion)

        with open("logs_rnn.txt", "a") as f:
            f.write(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f}\n")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "checkpoints/rnn_attn_best.pt")

    print("✅ Training selesai, model terbaik disimpan di checkpoints/rnn_attn_best.pt")

# ---- Plot loss curve ----

import matplotlib.pyplot as plt

plt.plot(train_losses, label="Train Loss")
plt.plot(valid_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_curve.png")
plt.show()