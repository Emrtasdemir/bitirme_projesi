# -*- coding: utf-8 -*-
import numpy as np

def generate_batch(batch_size=4, img_h=48, img_w=192):
    imgs = np.random.rand(batch_size, img_h, img_w, 1).astype(np.float32) * 0.3
    for b in range(batch_size):
        for k in range(0, img_h, 8):
            imgs[b, k:k+1, :, 0] += 0.2
        for k in range(0, img_w, 16):
            imgs[b, :, k:k+1, 0] += 0.2
        imgs[b] = np.clip(imgs[b], 0.0, 1.0)
    return imgs
