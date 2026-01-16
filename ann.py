"""
MNIST veri seti: 
    rakamlama: 0-9 toplamda 10 sinif var.
    28*28 piksel boyutunda resimler.
    grayscale (siyah-beyaz) resimler.
    60000 egitim, 10000 test verisi.
    amacimiz: ann ile bu resimleri tanimlamak ya da siniflandirmak.

Image processing:
    histogram esitleme: kontrast iyilestirme.
    gaussian blur: gurultu azaltma.
    canny edge detection: kenar tespiti.

libraries:
    tensorflow: keras ile ann modeli olusturma ve egitim.
    matplotlib: gorsellestirme.
    cv2: opencv image processing.

ANN (artifical neural network) ile MNIST veri setini siniflandirma.        
"""
# import libraries
import cv2 #opencv
import numpy as np #sayisal islemler icin
import matplotlib.pyplot as plt #gorsellestirme icin

from tensorflow.keras.datasets import mnist # MNIST veri seti
from tensorflow.keras.models import Sequential #ANN modeli icin
from tensorflow.keras.layers import Dense, Dropout #ann katmanlari icin
from tensorflow.keras.optimizers import Adam # optimizer


# load MNIST dataset
(x_train, y_train),(x_test, y_test)=mnist.load_data()
print(f"x train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
"""
x_train shape: (60000, 28, 28)
y_train shape: (60000,)

"""

# image preprocessing
img = x_train[0] #ilk resmi al.
stages = {"orijinal": img}

# histogram esitleme
eq = cv2.equalizeHist(img)
stages["Histogram Esitleme"] = eq

# gaussian blur: gurultuyu azaltma
blur = cv2.GaussianBlur(eq, (5,5), 0)
stages["gaussian blur"] = blur

# canny ile kenar tespiti
edges = cv2.Canny(blur, 50, 150) #kenar tespiti
stages["canny kenarlari"] = edges

# gorsellestirme
fig, axes = plt.subplots(2, 2, figsize=(6, 6))
axes = axes.flat
for ax, (title, im) in zip(axes, stages.items()):
    ax.imshow(im, cmap = "gray")
    ax.set_title(title)
    ax.axis("off")

plt.suptitle("MNIST Image Processing Stages")   
plt.tight_layout()
plt.show()

# preprocessing fonksiyonu
def preprocess_image(img):
    """
    -histogram esitleme
    -gaussian blur 
    -canny ile kenar tespiti
    -flattering: 28*28 boyutundan 784 boyutuna cevirme.
    -normalizasyon: 0-255 arasindan 0-1 arasina cevirme.
    """
    img_eq= cv2.equalizeHist(img) #histogram esitleme.
    img_blur= cv2.GaussianBlur(img_eq, (5,5), 0) #gaussian blur
    img_edges= cv2.Canny(img_blur, 50, 150) #canny kenar tespiti
    features = img_edges.flatten()/255.0 # flattering: 28*28 boyutundan 784 boyutuna cevirme
    return features

num_train = 10000
num_test = 2000

x_train = np.array([preprocess_image(img) for img in x_train[:num_train]])
y_train_sub = y_train[:num_train]

x_test = np.array([preprocess_image(img) for img in x_test[:num_test]])
y_test_sub = y_test[:num_test]

# ann model creation
model = Sequential([
    Dense(128, activation= "relu", input_shape=(784,)), # ilk katman, 128 nöron, 28*28 = 784 boyutunda.
    Dropout(0.5), # dropout katmani overfitting i azaltmak için %50 dropout.
    Dense(64, activation = "relu"), # ikinci katman 64 noron.
    Dense(10, activation = "softmax") # cikis katmani, 10 noron (0-9 rakamlari icin)
])

# compile model
model.compile(
    optimizer = Adam(learning_rate = 0.001), # optimizer
    loss = "sparse_categorical_crossentropy", # kayıp fonksiyonu
    metrics = ["accuracy"] # metrikler
)
print(model.summary())

# ann model training
history = model.fit(
    x_train, y_train_sub,
    validation_data = (x_test, y_test_sub),
    epochs = 50,
    batch_size = 32,
    verbose = 2
)

# eveluate model performance
test_loss, test_acc = model.evaluate(x_test, y_test_sub)
print(f"Test Loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

# plot training history
plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label = "Training Loss")
plt.plot(history.history["val_loss"], label = "Validation Loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label = "Training Accuracy")
plt.plot(history.history["val_accuracy"], label = "Validation Accuracy")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend

plt.tight_layout()
plt.show()


