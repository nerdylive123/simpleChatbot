import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')

# Load data from JSON file
with open('request.json', 'r') as f:
    data = json.load(f)

# Preprocess data
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',', ':', ';', '-', "'s", "'ve", "'re"]
for i, request in enumerate(data['request']):
    for pattern in request['patterns']:
        # Tokenize words
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add documents in the corpus
        documents.append((w, i))
        # Add to our classes list
        if request['tag'] not in classes:
            classes.append(request['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Randomize the training data
np.random.shuffle(training)
training = np.array(training)

x, y = list(training[:, 0]), list(training[:, 1])
x_train = x[:int(len(x) * 0.8)]
x_test = x[int(len(x) * 0.8):]
y_train = y[:int(len(y) * 0.8)]
y_test = y[int(len(y) * 0.8):]

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(1000, 16, input_length=(len(x_train[0]),)),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(y_train[0]), activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer="SGD", metrics=['accuracy'])
model.summary()

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32)

# Evaluate the model
model.evaluate(x_test, y_test, verbose=1)


# # Extract patterns and responses from JSON data
# patterns = []
# responses = []
# for i, request in enumerate(data['request']):
#     for pattern in request['patterns']:
#         patterns.append(pattern.lower())
#         responses.append(i)  # Use index as class label
#
# # Tokenize patterns and responses
# tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
# tokenizer.fit_on_texts(patterns)
# patterns_seq = tokenizer.texts_to_sequences(patterns)
# patterns_pad = pad_sequences(patterns_seq, maxlen=10, padding='post')
#
# # Define the model
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(1000, 16, input_length=10),
#     tf.keras.layers.GlobalAveragePooling1D(),
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.Dense(len(responses), activation='softmax')
# ])
#
# # Compile the model
# model.compile(loss='categorical_crossentropy', optimizer="SGD", metrics=['accuracy'])
#
#
# # Train the model
# model.fit(patterns_pad, tf.keras.utils.to_categorical(responses), epochs=50)
