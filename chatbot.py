import random
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem import WordNetLemmatizer

# Load dataset
with open("chatbot_dataset.json", "r") as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()

# Data preprocessing
words = []
classes = []
documents = []
ignore_words = ["?", "!", ".", ","]

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]))
classes = sorted(set(classes))

# Training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]
    for w in words:
        bag.append(1 if w in word_patterns else 0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build model
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation="relu"),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(len(classes), activation="softmax")
])

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train model
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save("chatbot_model.h5")

# Load trained model
def load_model():
    return tf.keras.models.load_model("chatbot_model.h5")

# Convert user input to bag-of-words
def bow(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag)

# Get response
def chat_response(user_input):
    model = load_model()
    input_bag = np.array([bow(user_input)])
    prediction = model.predict(input_bag)[0]
    max_index = np.argmax(prediction)

    if prediction[max_index] > 0.7:  # Confidence threshold
        tag = classes[max_index]
        for intent in data["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])
    return "I'm sorry, I didn't understand that."

# Chatbot loop
print("Chatbot is ready! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = chat_response(user_input)
    print(f"Bot: {response}")
