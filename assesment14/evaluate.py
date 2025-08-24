# evaluate.py (FIXED: sesuai rnn_attention + detokenize)
import os
import torch
import sentencepiece as spm
import sacrebleu

from rnn_attention import Encoder, Decoder, Seq2Seq
from train_rnn import DEVICE

# ---- greedy decoding untuk RNN+Attention ----
def translate_sentence(model, src_text_detok, sp_src, sp_tgt, max_len=50):
    model.eval()
    # encode src (dari teks normal)
    src_ids = [sp_src.bos_id()] + sp_src.encode(src_text_detok, out_type=int)[:max_len-2] + [sp_src.eos_id()]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    # decode step-by-step
    cur_id = sp_tgt.bos_id()
    out_ids = []
    for _ in range(max_len):
        inp = torch.tensor([cur_id], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            logits, hidden = model.decoder(inp, hidden, encoder_outputs)
        cur_id = int(logits.argmax(1).item())
        if cur_id == sp_tgt.eos_id():
            break
        out_ids.append(cur_id)

    # detokenize ke kalimat normal
    return sp_tgt.decode(out_ids)

if __name__ == "__main__":
    data_dir = "data_splits"
    test_path = os.path.join(data_dir, "test.tok.tsv")

    sp_src = spm.SentencePieceProcessor(model_file=os.path.join(data_dir, "spm_src.model"))
    sp_tgt = spm.SentencePieceProcessor(model_file=os.path.join(data_dir, "spm_tgt.model"))

    # bangun model persis seperti di train_rnn.py
    INPUT_DIM = sp_src.vocab_size()
    OUTPUT_DIM = sp_tgt.vocab_size()
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    DROPOUT = 0.3

    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, DROPOUT)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DROPOUT)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    model.load_state_dict(torch.load("checkpoints/rnn_attn_best.pt", map_location=DEVICE))
    model.eval()

    refs, hyps = [], []
    samples = []  # simpan contoh untuk ditampilkan

    with open(test_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            src_tok, tgt_tok = line.strip().split("\t")

            # detokenize src & ref dari subword pieces
            src_detok = sp_src.decode_pieces(src_tok.split())
            ref_detok = sp_tgt.decode_pieces(tgt_tok.split())

            hyp_detok = translate_sentence(model, src_detok, sp_src, sp_tgt)

            refs.append(ref_detok)
            hyps.append(hyp_detok)

            if idx < 5:
                samples.append((src_detok, ref_detok, hyp_detok))

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])

    print(f"✅ RNN SacreBLEU (detok): {bleu.score:.2f}")
    print(f"✅ RNN chrF (detok): {chrf.score:.2f}\n")

    for s, r, h in samples:
        print(f"SRC: {s}")
        print(f"REF: {r}")
        print(f"HYP: {h}\n")
