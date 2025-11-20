# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras as K

def build_crnn(img_h=48, img_w=192, vocab_size=80):
    inp = K.Input((img_h, img_w, 1), name="image")
    x = inp
    x = K.layers.Conv2D(64, 3, padding="same", activation="relu")(x); x = K.layers.MaxPool2D((2,2))(x)   # 48->24
    x = K.layers.Conv2D(128,3, padding="same", activation="relu")(x); x = K.layers.MaxPool2D((2,2))(x)   # 24->12
    x = K.layers.Conv2D(256,3, padding="same", activation="relu")(x); x = K.layers.MaxPool2D((2,1))(x)   # 12->6
    x = K.layers.Conv2D(256,3, padding="same", activation="relu")(x); x = K.layers.MaxPool2D((2,1))(x)   # 6->3
    x = K.layers.Conv2D(512,3, padding="same", activation="relu")(x); x = K.layers.MaxPool2D((2,1))(x)   # 3->1
    x = K.layers.Permute((2,1,3))(x)
    x = K.layers.Lambda(lambda t: tf.squeeze(t, axis=2))(x)  # [B, T, C]
    x = K.layers.Bidirectional(K.layers.LSTM(256, return_sequences=True))(x)
    x = K.layers.Bidirectional(K.layers.LSTM(256, return_sequences=True))(x)
    out = K.layers.Dense(vocab_size + 1, activation="softmax", name="logits")(x)
    return K.Model(inp, out, name="crnn_ctc")
