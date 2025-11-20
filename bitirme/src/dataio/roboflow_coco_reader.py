"""Roboflow COCO export okuyucu.
- COCO JSON'dan bbox + transcription alanını okur
- Satır kırpma, yeniden boyutlama ve tf.data pipeline üretir
"""
import json, os, tensorflow as tf
from PIL import Image

def load_coco_annotations(coco_json_path):
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    # Not: transcription bilgisi genelde annotation 'attributes' altında tutulur (Roboflow sürümüne göre değişebilir).
    return coco

def make_dataset(images_dir, coco_json, img_h=48, img_w=192, batch_size=32, shuffle=True):
    # TODO: gerçek COCO -> örnek listesi dönüştürme
    paths = []
    labels = []
    # Placeholder: boş dataset
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    return ds
