# Bitirme Projesi İskeleti

Bu iskelet; Roboflow ile etiketlenecek veri hazır olmadan önce eğitim/deney altyapısını kurmanız için hazırlanmıstır.
Tarih: 2025-11-05 15:38

## Klasör Yapısı
```
bitirme/
  data/                  # Roboflow exportları (train/val/test) buraya
  src/
    ocr/                 # OCR modeli (CRNN-CTC veya alternatif)
    writer_id/           # (Opsiyonel) Yazar tanıma modeli
    dataio/              # Veri okuyucular, tf.data yardımcıları
    eval/                # Metrikler, kalite kontrol
    mobile/              # TFLite/ONNX export ve mobil inference
  experiments/           # Deney konfigleri ve sonuçları
  notebooks/             # Hızlı denemeler (ipynb)
  docs/                  # Dokümanlar ve kılavuzlar
  requirements.txt
  train_ocr.py
  train_writer.py
  README.md
```

## Hızlı Başlangıç
1. `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. Roboflow exportlarını `data/` altına koyun (COCO JSON + images).
4. `python train_ocr.py --config experiments/ocr_baseline.yaml`

> Not: Tüm dosyalarda Türkçe açıklamalar bulunmaktadır.
