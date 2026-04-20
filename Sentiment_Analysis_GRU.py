import tensorflow as tf

# Load + preprocess
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=5000)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=100)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=100)

print("Training Samples: ", x_train)
print("Testing Samples: ", x_test)

# Model (simplified)
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 32),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile + train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3, validation_split=0.2)

# Test
print("Test Acc:", model.evaluate(x_test, y_test)[1])
