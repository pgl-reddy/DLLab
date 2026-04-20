import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

# Load data
(x_train, _), (x_test, _) = fashion_mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Build model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),  # Encoder
    layers.Dense(784, activation='sigmoid')                   # Decoder
])

# Compile
model.compile(optimizer='adam', loss='mse')

# Train
model.fit(x_train, x_train, epochs=5, validation_data=(x_test, x_test))

# Predict
decoded = model.predict(x_test)

# Display results
plt.figure(figsize=(10,4))
for i in range(5):
    # Original
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.axis('off')

    # Reconstructed
    plt.subplot(2,5,i+6)
    plt.imshow(decoded[i].reshape(28,28), cmap='gray')
    plt.axis('off')

plt.show()
