"""
Writer ID (Yazar Tanıma) Eğitim Scripti

Bu script, el yazısı sayfalarından yazarı tanıyan bir CNN modelini eğitir.
Veri akışı:
  1. Sayfa resmi (224x224x3 RGB) → Writer ID Model → Yazar etiketi (sınıf ID)
  2. Model, classes.txt'deki sınıf listesine göre tahmin yapar

Eğitim pipeline:
  - classes.txt'den sınıf listesini okur
  - train/val/test klasörlerinden TensorFlow dataset'leri oluşturur
  - Görüntüleri normalize eder (0-255 → 0-1)
  - CNN modelini eğitir (Adam optimizer, sparse_categorical_crossentropy)
  - Test setinde değerlendirir
  - Modeli models/writer_cnn.h5 olarak kaydeder
"""
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from src.writer_id.model import build_writer_cnn

# Proje kökü = bu dosyanın olduğu 'bitirme' klasörü
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset kökü (relative path kullanarak)
DATA_ROOT = PROJECT_ROOT / "data" / "writer_id" / "veri_seti_split"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"
CLASSES_TXT = DATA_ROOT / "classes.txt"

# Model kayıt yolu
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_SAVE_PATH = MODELS_DIR / "writer_cnn.h5"

# Eğitim parametreleri
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4


def load_class_names(path: Path):
    """
    classes.txt dosyasından sınıf isimlerini okur.
    
    Args:
        path: classes.txt dosyasının yolu
    
    Returns:
        List[str]: Boş satırlar atlanmış sınıf isimleri listesi
    """
    with open(path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    return classes


def normalize_image(image, label):
    """
    Görüntüyü 0-255 aralığından 0-1 aralığına normalize eder.
    
    Args:
        image: TensorFlow tensor (uint8, 0-255)
        label: Sınıf etiketi
    
    Returns:
        (normalized_image, label): float32 görüntü ve etiket
    """
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def create_datasets():
    """
    Train/val/test TensorFlow dataset'lerini oluşturur ve normalize eder.
    
    Returns:
        (train_ds, val_ds, test_ds, class_names): Normalize edilmiş dataset'ler ve sınıf isimleri
    """
    common_kwargs = dict(
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        label_mode="int",  # Her resim için tek sınıf ID'si
    )

    # Train dataset (shuffle=True)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        shuffle=True,
        seed=42,
        **common_kwargs,
    )

    # class_names'i normalize işleminden ÖNCE al (prefetch sonrası kaybolur)
    class_names = train_ds.class_names

    # Validation dataset (shuffle=True)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        shuffle=True,
        seed=123,
        **common_kwargs,
    )

    # Test dataset (shuffle=False, sıralı kalmalı)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        shuffle=False,
        **common_kwargs,
    )

    # Normalize et: 0-255 → 0-1 (float32)
    train_ds = train_ds.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch ile performans iyileştirme
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def main():
    """
    Ana eğitim akışı:
    1. Veri yollarını kontrol et
    2. classes.txt'den sınıf listesini oku
    3. Dataset'leri yükle ve normalize et
    4. Modeli oluştur ve derle
    5. Modeli eğit
    6. Test setinde değerlendir
    7. Modeli kaydet
    """
    print("=" * 60)
    print("Writer ID (Yazar Tanıma) Eğitim Başlatılıyor")
    print("=" * 60)
    
    # 1) Dataset yollarını kontrol et
    print("\n[1] Veri yolları kontrol ediliyor...")
    print(f"  TRAIN_DIR: {TRAIN_DIR}")
    print(f"  VAL_DIR:   {VAL_DIR}")
    print(f"  TEST_DIR:  {TEST_DIR}")
    print(f"  CLASSES:   {CLASSES_TXT}")

    assert TRAIN_DIR.exists(), f"Train klasörü yok: {TRAIN_DIR}"
    assert VAL_DIR.exists(), f"Val klasörü yok: {VAL_DIR}"
    assert TEST_DIR.exists(), f"Test klasörü yok: {TEST_DIR}"
    assert CLASSES_TXT.exists(), f"classes.txt bulunamadı: {CLASSES_TXT}"
    print("  ✓ Tüm veri yolları mevcut")

    # 2) classes.txt'den sınıf isimlerini oku
    print("\n[2] Sınıf listesi okunuyor...")
    class_names_txt = load_class_names(CLASSES_TXT)
    num_classes = len(class_names_txt)
    print(f"  Sınıflar: {class_names_txt}")
    print(f"  Sınıf sayısı: {num_classes}")

    # 3) TensorFlow dataset'lerini oluştur
    print("\n[3] Dataset'ler yükleniyor ve normalize ediliyor...")
    train_ds, val_ds, test_ds, class_names_tf = create_datasets()

    # TF'in gördüğü sınıf isimleri (alfabetik sırada)
    print(f"  TF dataset sınıfları: {class_names_tf}")

    # 3.1) classes.txt ile TF class_names aynı mı, kontrol et
    assert class_names_tf == class_names_txt, (
        f"classes.txt ile klasör isimleri aynı sırada değil!\n"
        f"  classes.txt: {class_names_txt}\n"
        f"  TF dataset:  {class_names_tf}"
    )
    print("  ✓ Sınıf sıralaması doğrulandı")

    # 3.2) Bir batch alıp şekle bakalım
    for images, labels in train_ds.take(1):
        print(f"  Batch görüntü şekli: {images.shape}")
        print(f"  Görüntü veri tipi: {images.dtype}")
        print(f"  Görüntü aralığı: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
        print(f"  Batch label örnekleri: {labels.numpy()[:5]}")

    # 4) Modeli oluştur
    print("\n[4] Model oluşturuluyor...")
    model = build_writer_cnn(num_classes=num_classes, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    print("  Model yapısı:")
    model.summary()

    # 5) Modeli derle
    print("\n[5] Model derleniyor...")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    print(f"  Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"  Loss: sparse_categorical_crossentropy")
    print(f"  Metrics: accuracy")

    # 6) Modeli eğit
    print(f"\n[6] Model eğitimi başlatılıyor ({EPOCHS} epoch)...")
    print("  " + "-" * 56)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1
    )
    print("  " + "-" * 56)
    print("  ✓ Eğitim tamamlandı")

    # 7) Test setinde değerlendir
    print("\n[7] Test setinde değerlendirme yapılıyor...")
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # 8) Modeli kaydet
    print("\n[8] Model kaydediliyor...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"  ✓ Model kaydedildi: {MODEL_SAVE_PATH}")

    print("\n" + "=" * 60)
    print("Eğitim başarıyla tamamlandı!")
    print("=" * 60)


if __name__ == "__main__":
    main()
