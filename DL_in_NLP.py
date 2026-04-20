import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
intents = {
    "greeting":{
        "patterns":["hi", "hello", "hey", "Good morning"],
        "responses":["hello", "hi there!", "hey! how can i help you?"]
        },
    "goodbye":{
        "patterns":["bye", "see you", "goodbye"],
        "responses":["goodbye", "see you later", "take care"]
        },
    "thanks":{
        "patterns":["Thanks", "Thank you!!", "That's helpfull!!"],
        "responses":["You're welcome", "Happy to help", "Anytime"]
        }
    }
sentences = []
labels = []
label_names = list(intents.keys())

for idx, tag in enumerate(label_names):
    for pattern in intents[tag]["patterns"]:
        sentences.append(pattern)
        labels.append(idx)

x = tf.constant(sentences)
y = tf.constant(labels)

vectorizer = layers.TextVectorization(
    max_tokens = 1000,
    output_sequence_length = 10
)
vectorizer.adapt(x)

model = keras.Sequential([
    vectorizer,
    layers.Embedding(1000, 16),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation = 'relu'),
    layers.Dense(len(label_names), activation = 'softmax')
])

model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

model.fit(x, y, epochs = 200, verbose = 0)
print("Training complete!")

def chatbot():
    print("chatbot")
    while True:
        user_in = input("you:")
        if user_in.lower() == "quit":
            print("Exiting!!")
            break
        input_tensor = tf.constant([user_in])
        prediction = model.predict(input_tensor, verbose = 0)
        predicted_index = np.argmax(prediction)
        tag = label_names[predicted_index]
        
        response = random.choice(intents[tag]["responses"])
        print("Bot:", response)
    
chatbot()
