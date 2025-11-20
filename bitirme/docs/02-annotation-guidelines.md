# 02 - Etiketleme Kılavuzu (Roboflow)

## OCR (Satır Düzeyi Önerilir)
- Her satır için **tek bbox** ve `transcription` alanı.
- UTF-8 Türkçe karakterler *bozulmayacak*.
- Boşluk/işaretleme standardını değişmez tutun.

## Writer ID (Opsiyonel)
- Görüntüye `writer_id` (class) verin.

## İsimlendirme
- Dosya adlarında Türkçe karakter kullanmayın.
- Dataset split: train/val/test ≈ 80/10/10.

## Kalite Kontrol
- Boş `transcription` olmamalı.
- Çok uzun metinler not alın (model giriş eni gerekirse artırılacak).
