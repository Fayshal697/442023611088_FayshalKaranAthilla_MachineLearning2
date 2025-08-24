import os
import random

from transformers.models.bert_japanese.tokenization_bert_japanese import spm

import data

def clean_text(s):
    """Bersihkan teks sederhana: lowercase + strip spasi"""
    return s.strip().lower()

def load_dataset(raw_path):
    """Baca file raw.txt -> list pasangan (src, tgt)"""
    pairs = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            if "\t" not in line:
                continue
            src, tgt = line.strip().split("\t")[:2]  # ambil hanya dua kolom
            pairs.append((clean_text(src), clean_text(tgt)))
    return pairs

def split_dataset(pairs, valid_ratio=0.05, test_ratio=0.05, seed=42):
    """Split dataset ke train/valid/test"""
    random.seed(seed)
    random.shuffle(pairs)
    n_total = len(pairs)
    n_valid = int(n_total * valid_ratio)
    n_test = int(n_total * test_ratio)

    valid = pairs[:n_valid]
    test = pairs[n_valid:n_valid+n_test]
    train = pairs[n_valid+n_test:]
    return train, valid, test

def save_split(pairs, out_path):
    """Simpan split ke file tsv"""
    with open(out_path, "w", encoding="utf-8") as f:
        for src, tgt in pairs:
            f.write(f"{src}\t{tgt}\n")

def train_sentencepiece(input_file, model_prefix, vocab_size=8000, character_coverage=1.0):
    """Train SentencePiece model"""
    spm.SentencePieceTrainer.train(
        f"--input={input_file} --model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} --character_coverage={character_coverage} "
        f"--model_type=bpe"
    )

def apply_sentencepiece(sp_model, infile, outfile):
    sp = spm.SentencePieceProcessor(model_file=sp_model)
    with open(infile, "r", encoding="utf-8") as fin, open(outfile, "w", encoding="utf-8") as fout:
        for line in fin:
            src, tgt = line.strip().split("\t")
            src_tok = " ".join(sp.encode(src, out_type=str))
            tgt_tok = " ".join(sp.encode(tgt, out_type=str))
            fout.write(f"{src_tok}\t{tgt_tok}\n")

if __name__ == "__main__":
    raw_path = os.path.join("data", "raw.txt")   # <--- ini
    out_dir = "data_splits"  # simpan hasil split di folder baru biar rapi

    os.makedirs(out_dir, exist_ok=True)

    pairs = load_dataset(raw_path)
    print(f"Total pairs: {len(pairs)}")

    train, valid, test = split_dataset(pairs)

    save_split(train, os.path.join(out_dir, "train.tsv"))
    save_split(valid, os.path.join(out_dir, "valid.tsv"))
    save_split(test, os.path.join(out_dir, "test.tsv"))

    print(f"Train: {len(train)} | Valid: {len(valid)} | Test: {len(test)}")
    print(f"Dataset tersimpan di folder '{out_dir}'")

    # Step 2: Train SentencePiece vocab (pakai hanya data train biar realistis)
    train_src = os.path.join(out_dir, "train.src")
    train_tgt = os.path.join(out_dir, "train.tgt")

    with open(train_src, "w", encoding="utf-8") as fs, open(train_tgt, "w", encoding="utf-8") as ft:
        for src, tgt in train:
            fs.write(src + "\n")
            ft.write(tgt + "\n")

    # Latih model SentencePiece untuk src dan tgt
    train_sentencepiece(train_src, os.path.join(out_dir, "spm_src"), vocab_size=8000)
    train_sentencepiece(train_tgt, os.path.join(out_dir, "spm_tgt"), vocab_size=8000)

    # Step 3: Apply tokenization ke semua split
    for split in ["train", "valid", "test"]:
        infile = os.path.join(out_dir, f"{split}.tsv")
        outfile = os.path.join(out_dir, f"{split}.tok.tsv")
        apply_sentencepiece(os.path.join(out_dir, "spm_src.model"), infile, outfile)

    print("✅ Preprocessing selesai! File tokenized tersimpan di data_splits/*.tok.tsv")

