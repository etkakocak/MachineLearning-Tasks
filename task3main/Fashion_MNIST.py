# Exercise 1: Fashion MNIST

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Load data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Plot 16 random samples
plt.figure(figsize=(10, 10))
for i in range(16):
    idx = np.random.randint(0, 60000)
    plt.subplot(4, 4, i + 1)
    plt.imshow(train_images[idx], cmap='gray')
    plt.title(class_names[train_labels[idx]])
    plt.axis('off')
plt.show()

# Normalize the images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the neural network model
model = models.Sequential([
    layers.Input(shape=(28, 28)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
history = model.fit(train_images, train_labels, epochs=50, validation_split=0.1, callbacks=[early_stopping])

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc:.2f}")
