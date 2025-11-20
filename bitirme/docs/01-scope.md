# 01 - Kapsam ve Kabul Kriterleri

## Amaç
- El yazısı satır/kelime görüntülerinden **Türkçe metin** üretmek (OCR).
- (Opsiyonel) Aynı görüntüden **yazar kimliği** (writer_id) tahmini yapmak.

## Kapsam Dışı
- Tam sayfa düzen analizi (layout) bu sürümde yok.

## Karakter Seti
- Türkçe karakterler: ç, ğ, ı, İ, ö, ş, ü
- Rakamlar, temel noktalama ve boşluk.

## Metrikler (hedef)
- OCR: CER ≤ %15 (ilk kesit), WER ≤ %25.
- WriterID: Top-1 Accuracy ≥ %80 (sınıf dengesine bağlı).

## Teslimler
- Eğitim kodu, model dosyaları, TFLite export, kısa demo, rapor.
