# -*- coding: utf-8 -*-
import json

def load_vocab(vocab_path):
    # UTF-8 + BOM olasılığına karşı utf-8-sig ile okuyalım
    with open(vocab_path, "r", encoding="utf-8-sig") as f:
        d = json.load(f)
    chars = d["chars"]
    id2ch = {i: ch for i, ch in enumerate(chars)}
    ch2id = {ch: i for i, ch in enumerate(chars)}
    blank_id = len(chars)  # CTC blank sondaki id
    return chars, id2ch, ch2id, blank_id

def ids_to_text(id_list, id2ch):
    return "".join(id2ch[i] for i in id_list if i in id2ch)
