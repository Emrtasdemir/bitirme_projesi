# -*- coding: utf-8 -*-
import os, json, yaml, numpy as np
import tensorflow as tf
from src.ocr.crnn_ctc import build_crnn
from src.ocr.textcodec import load_vocab, ids_to_text
from src.ocr.ctc_utils import greedy_decode
from src.ocr.train_loop import train_step
from src.dataio.synthetic_lines import generate_batch

def _make_random_labels(batch_size, max_time, vocab_size, min_len=3, max_len=None):
    if max_len is None:
        max_len = max(5, max_time // 2)
    label_seqs = []
    for _ in range(batch_size):
        L = np.random.randint(min_len, max_len + 1)
        seq = np.random.randint(0, vocab_size, size=(L,), dtype=np.int32).tolist()
        label_seqs.append(seq)
    return label_seqs

def main(cfg_path, smoke_ctc=False, dry_train=False, steps=20, lr=1e-3):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    vocab_path = cfg["vocab_path"]
    if not os.path.isabs(vocab_path):
        vocab_path = os.path.join("bitirme", vocab_path)

    chars, id2ch, ch2id, blank_id = load_vocab(vocab_path)
    vocab_size = len(chars)

    model = build_crnn(img_h=cfg["img_h"], img_w=cfg["img_w"], vocab_size=vocab_size)
    print("Model oluşturuldu:", model.name, "| Vocab size:", vocab_size, "| Blank id:", blank_id)

    if smoke_ctc:
        x = generate_batch(batch_size=4, img_h=cfg["img_h"], img_w=cfg["img_w"])
        y = model(x, training=False).numpy()
        print("Logits şekli:", y.shape)
        seqs = greedy_decode(y, blank_index=blank_id)
        for i, s in enumerate(seqs[:2]):
            txt = ids_to_text(s, id2ch)
            print(f"[{i}] decoded (gürültü, anlamsız olabilir):", repr(txt))
        print("CTC smoke test: OK")

    if dry_train:
        print(f"Dry-run başlıyor: steps={steps}, lr={lr}")
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
        for step in range(1, steps + 1):
            x = generate_batch(batch_size=8, img_h=cfg['img_h'], img_w=cfg['img_w'])
            # Zaman boyutunu öğrenmek için bir ileri geçiş
            T = model(x, training=False).shape[1]
            label_seqs = _make_random_labels(batch_size=8, max_time=T, vocab_size=vocab_size)
            loss = train_step(model, opt, x, label_seqs, blank_index=blank_id).numpy()
            if step % 5 == 0 or step == 1:
                print(f"step {step:03d} | ctc_loss: {loss:.4f}")
        print("Dry-run: OK (eğitim iskeleti çalışıyor)")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/ocr_baseline.yaml")
    ap.add_argument("--smoke_ctc", type=int, default=0)
    ap.add_argument("--dry_train", type=int, default=0, help="1 ise sentetik etiketlerle kısa CTC eğitim denemesi yapar")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    main(args.config, smoke_ctc=bool(args.smoke_ctc), dry_train=bool(args.dry_train), steps=args.steps, lr=args.lr)
