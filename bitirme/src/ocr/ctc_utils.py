# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

def ctc_loss(logits, labels, logit_lengths, label_lengths, blank_index):
    """
    Compute CTC loss.
    logits: (B, T, C)
    labels: tf.SparseTensor
    logit_lengths: (B,)
    label_lengths: (B,)
    """
    loss = tf.nn.ctc_loss(
        labels=labels,
        logits=logits,
        label_length=label_lengths,
        logit_length=logit_lengths,
        logits_time_major=False,
        blank_index=blank_index
    )
    return tf.reduce_mean(loss)

def greedy_decode(logits, blank_index):
    """
    Argmax -> collapse repeats -> remove blanks.
    logits: (B, T, C)
    return: list[list[int]]
    """
    pred_ids = np.argmax(logits, axis=-1)  # (B, T)
    results = []
    for seq in pred_ids:
        out = []
        prev = None
        for t in seq:
            if t == blank_index:
                prev = None
                continue
            if t != prev:
                out.append(int(t))
            prev = t
        results.append(out)
    return results
