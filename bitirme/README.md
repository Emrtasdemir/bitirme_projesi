# ✍️ Optik El Yazısı Tanıma ve Yazar Tanımlama Sistemi  
**Bitirme Projesi – 2025**

Bu proje, el yazısı görüntülerinden hem **yazar kimliği tahmini (Writer Identification)** hem de **optik karakter tanıma (OCR)** yapabilen iki modüllü bir yapay zeka sistemidir.  
Tüm kodlar, veri seti, eğitim scriptleri ve dokümantasyon bu repoda yer almaktadır.

---

# 📌 1. Proje Amacı

Bu bitirme projesinin temel amacı:

1. **El yazısı görüntülerini analiz ederek kişinin kim olduğunu tahmin etmek**  
2. **El yazısı satırlarını karakter dizisine dönüştüren bir OCR modeli geliştirmek**  
3. **Modeli masaüstünde ve mobil cihazlarda çalışabilecek şekilde tasarlamak**

---

# 📁 2. Proje Klasör Yapısı

bitirme/
├─ README.md
├─ requirements.txt
├─ data/ # Veri seti (writer identification)
│ └─ writer_id/
│ └─ veri_seti_split/
│ ├─ train/
│ ├─ val/
│ └─ test/
├─ docs/
│ ├─ 01-scope.md
│ ├─ 02-annotation-guidelines.md
│ ├─ 03-mobile-flow.md
│ └─ rapor_sablon.md
├─ notebooks/
├─ src/
│ ├─ dataio/
│ │ ├─ roboflow_coco_reader.py
│ │ └─ synthetic_lines.py
│ ├─ eval/
│ │ ├─ cer_wer.py
│ │ └─ qual_check.py
│ ├─ mobile/
│ │ ├─ export_tflite.py
│ │ └─ inference_tflite.py
│ ├─ ocr/
│ │ ├─ crnn_ctc.py
│ │ ├─ ctc_utils.py
│ │ ├─ textcodec.py
│ │ ├─ train_loop.py
│ │ └─ vocab.json
│ └─ writer_id/
│ └─ model.py
├─ train_ocr.py
└─ train_writer.py

yaml
Kodu kopyala

---

# 🧠 3. Modül 1: Yazar Tanıma (Writer Identification)

Bu modül, el yazısı görüntüsünden **hangi kişinin yazdığına** karar verir.

### ✔ Kullanılan yöntem  
CNN tabanlı bir sınıflandırma modeli

### ✔ Veri seti  
9 kişiden alınmış el yazısı sayfaları  
Aşağıdaki gibi üçe ayrılmıştır:

train/
val/
test/

bash
Kodu kopyala

### ✔ Eğitim komutu

```bash
python bitirme/train_writer.py
🔤 4. Modül 2: OCR – Optik Karakter Tanıma
El yazısı satırlarını metne dönüştürmek için CRNN + CTC tabanlı bir model geliştirilmiştir.

✔ Kullanılan mimari:
CNN → görsel özellik çıkarımı

BiLSTM → sekans öğrenme

CTC Loss → hizalama sorununu çözme

✔ Eğitim komutu:
bash
Kodu kopyala
python bitirme/train_ocr.py
📱 5. Mobil Cihazlar için TFLite Desteği
Model mobil cihazlara aktarılabilir.

✔ TFLite’e dönüştürme:
bash
Kodu kopyala
python bitirme/src/mobile/export_tflite.py
✔ Mobilde inference:
bash
Kodu kopyala
python bitirme/src/mobile/inference_tflite.py
📦 6. Kurulum ve Çalıştırma
1) Sanal ortam:
bash
Kodu kopyala
python -m venv .venv
.\.venv\Scripts\activate
2) Gereksinimler:
bash
Kodu kopyala
pip install -r bitirme/requirements.txt
3) Kurulum testi:
bash
Kodu kopyala
python verify_setup.py
📌 7. Eğitim Sonuçları (Doldurulacak)
Bu bölüme eğitimden sonra ekleyebilirsiniz:

Accuracy – Loss grafik

Confusion matrix

Örnek tahmin çıktıları

👤 8. Geliştiren
Emir Taşdemir
Bitirme Projesi – 2025
