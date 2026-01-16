# MNIST Rakam Sınıflandırma - ANN ve Görüntü İşleme

Bu proje MNIST veri setini kullanarak yapay sinir ağları (ANN) ile rakam sınıflandırma yapmaktadır.

## Özellikler
- Histogram eşitleme ile kontrast iyileştirme
- Gaussian blur ile gürültü azaltma
- Canny edge detection ile kenar tespiti
- 3 katmanlı ANN modeli
- %90.10 test doğruluğu

## Kullanılan Teknolojiler
- Python 3.x
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib

## Kurulum
```bash
pip install -r requirements.txt
```

## Kullanım
```bash
python mnist_ann.py
```

## Sonuçlar
## Görüntü İşleme Aşamaları
<img src="images/not3.PNG" width="500">


## Model Mimarisi
<img src="images/not4.PNG" width="500">

## Model Performansı
<img src="images/not6.PNG" width="500">

## Model Mimarisi
- Giriş Katmanı: 128 nöron (ReLU)
- Dropout: 0.5
- Gizli Katman: 64 nöron (ReLU)
- Çıkış Katmanı: 10 nöron (Softmax)

## Performans
- Eğitim Doğruluğu: %97.88
- Test Doğruluğu: %90.10
- Test Loss: 0.6011




