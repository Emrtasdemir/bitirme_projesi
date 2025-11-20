# -*- coding: utf-8 -*-
import tensorflow as tf
from .ctc_utils import ctc_loss

@tf.function
def train_step(model, optimizer, batch_images, label_seqs, blank_index):
    """
    batch_images: (B,H,W,1) float32 [0,1]
    label_seqs: Python list of lists (değişken uzunluk)
    """
    # Python list -> RaggedTensor -> SparseTensor (grafikte güvenli)
    label_rt = tf.ragged.constant(label_seqs, dtype=tf.int32)
    label_lengths = tf.cast(label_rt.row_lengths(), tf.int32)
    labels_sp = tf.cast(label_rt.to_sparse(), tf.int32)

    with tf.GradientTape() as tape:
        logits = model(batch_images, training=True)  # (B,T,C+1)
        T = tf.shape(logits)[1]
        B = tf.shape(logits)[0]
        logit_lengths = tf.fill([B], T)
        loss = ctc_loss(
            logits=logits,
            labels=labels_sp,
            logit_lengths=logit_lengths,
            label_lengths=label_lengths,
            blank_index=blank_index
        )

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return tf.cast(loss, tf.float32)
