import os
import torch
import sentencepiece as spm
import sacrebleu

from transformer_model import TransformerModel
from train_transformer import DEVICE

# ---- Beam Search Decoding ----
def beam_search_decode(model, src_text, sp_src, sp_tgt, beam_size=5, max_len=50):
    model.eval()
    src_ids = [sp_src.bos_id()] + sp_src.encode(src_text, out_type=int)[:max_len-2] + [sp_src.eos_id()]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        memory = model.encode(src_tensor)

    # beam = [(sequence, score)]
    beams = [([sp_tgt.bos_id()], 0)]
    completed = []

    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq[-1] == sp_tgt.eos_id():
                completed.append((seq, score))
                continue

            tgt_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model.decode(memory, tgt_tensor)
                next_token_logits = output[:, -1, :]
                probs = torch.log_softmax(next_token_logits, dim=-1)

            topk_probs, topk_ids = probs.topk(beam_size)
            for prob, idx in zip(topk_probs[0], topk_ids[0]):
                new_seq = seq + [idx.item()]
                new_score = score + prob.item()
                new_beams.append((new_seq, new_score))

        # ambil k terbaik
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

    if not completed:
        completed = beams
    best_seq = max(completed, key=lambda x: x[1])[0]

    # remove <sos>, <eos>
    if best_seq[0] == sp_tgt.bos_id():
        best_seq = best_seq[1:]
    if best_seq[-1] == sp_tgt.eos_id():
        best_seq = best_seq[:-1]

    return sp_tgt.decode(best_seq)

# ---- MAIN ----
if __name__ == "__main__":
    data_dir = "data_splits"
    test_path = os.path.join(data_dir, "test.tok.tsv")

    sp_src = spm.SentencePieceProcessor(model_file=os.path.join(data_dir, "spm_src.model"))
    sp_tgt = spm.SentencePieceProcessor(model_file=os.path.join(data_dir, "spm_tgt.model"))

    SRC_VOCAB = sp_src.vocab_size()
    TGT_VOCAB = sp_tgt.vocab_size()

    model = TransformerModel(SRC_VOCAB, TGT_VOCAB, embed_size=256, num_heads=8, num_layers=3, dropout=0.3).to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/transformer_best.pt", map_location=DEVICE))
    model.eval()

    refs, hyps = [], []
    samples = []

    with open(test_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            src_tok, tgt_tok = line.strip().split("\t")

            # detokenize src & ref
            src_detok = sp_src.decode_pieces(src_tok.split())
            ref_detok = sp_tgt.decode_pieces(tgt_tok.split())

            hyp_detok = beam_search_decode(model, src_detok, sp_src, sp_tgt, beam_size=5)

            refs.append(ref_detok)
            hyps.append(hyp_detok)

            if idx < 5:
                samples.append((src_detok, ref_detok, hyp_detok))

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])

    print(f"✅ Transformer SacreBLEU (beam): {bleu.score:.2f}")
    print(f"✅ Transformer chrF (beam): {chrf.score:.2f}\n")

    for s, r, h in samples:
        print(f"SRC: {s}")
        print(f"REF: {r}")
        print(f"HYP: {h}\n")
