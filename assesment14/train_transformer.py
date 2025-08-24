# train_transformer.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sentencepiece as spm
from train_rnn import TranslationDataset, collate_fn, DEVICE
from transformer_model import TransformerModel

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

    SRC_VOCAB = sp_src.vocab_size()
    TGT_VOCAB = sp_tgt.vocab_size()

    model = TransformerModel(SRC_VOCAB, TGT_VOCAB, d_model=256, nhead=8,
                             num_encoder_layers=3, num_decoder_layers=3,
                             dim_feedforward=512, dropout=0.3).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    N_EPOCHS = 10
    best_valid_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(N_EPOCHS):
        # ---- training ----
        model.train()
        train_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            optimizer.zero_grad()
            output = model(src, tgt[:,:-1])  # input tgt tanpa <eos>
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            tgt_y = tgt[:,1:].reshape(-1)   # target shift
            loss = criterion(output, tgt_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            train_loss += loss.item()

        # ---- validation ----
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for src, tgt in valid_loader:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                output = model(src, tgt[:,:-1])
                output_dim = output.shape[-1]
                output = output.reshape(-1, output_dim)
                tgt_y = tgt[:,1:].reshape(-1)
                loss = criterion(output, tgt_y)
                valid_loss += loss.item()

        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.3f} | Val Loss: {valid_loss/len(valid_loader):.3f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "checkpoints/transformer_best.pt")

    print("✅ Training selesai, model terbaik disimpan di checkpoints/transformer_best.pt")
