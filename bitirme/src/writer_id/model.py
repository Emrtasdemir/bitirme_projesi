"""
Yazar Tanıma (Writer ID) için CNN Modeli

Bu modül, el yazısı sayfalarından yazarı tanıyan bir CNN modeli oluşturur.
Model, 224x224x3 boyutunda RGB görüntüler alır ve num_classes sayıda
yazar sınıfından birini tahmin eder.

Model yapısı:
- Conv2D + BatchNormalization + ReLU + MaxPooling blokları
- GlobalAveragePooling2D ile özellik çıkarımı
- Dropout ile regularizasyon
- Dense layer ile sınıflandırma
"""
from tensorflow import keras as K


def build_writer_cnn(num_classes: int, input_shape=(224, 224, 3)):
    """
    224x224x3 girdi alan ve num_classes çıktı veren CNN modeli oluşturur.
    
    Args:
        num_classes: Tahmin edilecek yazar sayısı
        input_shape: Görüntü boyutları (height, width, channels)
    
    Returns:
        Keras Model: Derlenmemiş model (compile edilmesi gerekir)
    """
    inp = K.Input(shape=input_shape, name="input_image")
    
    # İlk Conv-BN-ReLU-MaxPool bloğu
    x = K.layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inp)
    x = K.layers.BatchNormalization(name="bn1")(x)
    x = K.layers.ReLU(name="relu1")(x)
    x = K.layers.MaxPooling2D((2, 2), name="pool1")(x)
    
    # İkinci Conv-BN-ReLU-MaxPool bloğu
    x = K.layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = K.layers.BatchNormalization(name="bn2")(x)
    x = K.layers.ReLU(name="relu2")(x)
    x = K.layers.MaxPooling2D((2, 2), name="pool2")(x)
    
    # Üçüncü Conv-BN-ReLU-MaxPool bloğu
    x = K.layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = K.layers.BatchNormalization(name="bn3")(x)
    x = K.layers.ReLU(name="relu3")(x)
    x = K.layers.MaxPooling2D((2, 2), name="pool3")(x)
    
    # Dördüncü Conv-BN-ReLU-MaxPool bloğu
    x = K.layers.Conv2D(256, (3, 3), padding="same", name="conv4")(x)
    x = K.layers.BatchNormalization(name="bn4")(x)
    x = K.layers.ReLU(name="relu4")(x)
    x = K.layers.MaxPooling2D((2, 2), name="pool4")(x)
    
    # Global Average Pooling ile özellik vektörüne dönüştürme
    x = K.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    # Dropout ile regularizasyon
    x = K.layers.Dropout(0.3, name="dropout")(x)
    
    # Sınıflandırma katmanı
    out = K.layers.Dense(num_classes, activation="softmax", name="output")(x)
    
    model = K.Model(inputs=inp, outputs=out, name="writer_cnn")
    return model
