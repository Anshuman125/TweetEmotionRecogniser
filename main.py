# Using a Tokenizer in TensorFlow
# Padding and Truncating Sequences
# Creating and Training Recurrent Neural Networks
# Using NLP and Deep Learning to perform Text Classification

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nlp
import random

# We pass the history object that we get after model training in tensorflow to this function and it will plot the accuracy, cross validation accuracy, loss and the cross validation loss for us.
# And import the confusion matrix from sklearn to calculate our predictions against the ground truth
def show_history(h):
    epochs_trained = len(h.history['loss'])
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs_trained), h.history.get('accuracy'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_accuracy'), label='Validation')
    plt.ylim([0., 1.])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0, epochs_trained), h.history.get('loss'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_loss'), label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def show_confusion_matrix(y_true, y_pred, classes):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(8, 8))
    sp = plt.subplot(1, 1, 1)
    ctx = sp.matshow(cm)
    plt.xticks(list(range(0, 6)), labels=classes)
    plt.yticks(list(range(0, 6)), labels=classes)
    plt.colorbar(ctx)
    plt.show()


print('Using TensorFlow version', tf.__version__)

# hugging face nlp module makes data import very simple
from datasets import load_dataset
emotions_ds = load_dataset("emotion")

# Access these sets
train = emotions_ds['train']
valid = emotions_ds['validation']
test = emotions_ds['test']

# In each of the dataset the data is divided into two features "text" and "label" so now we are getting access to them by two lists using a function named get_tweets
# Here the label is present in 6 classes:
# 0. Sadness    1. Joy    2. Love    3. Anger     4. Fear    5. Surprise
def get_tweets(data):
    tweets = []
    labels = []
    for i in data:
        tweets.append(i['text'])
        labels.append(i['label'])
    return tweets, labels

tweets, labels = get_tweets(train)

# Tensorflow comes with build in tokeniser. Here tokeniser means breaking a sentence into smaller units called tokens
# for example: "This is a sentence" might be tokenised into "This", "is", "a", "sentence"
from tensorflow.keras.preprocessing.text import Tokenizer
# Tensorflow keras is an high level neural networks API allows developers to define, compile, and train deep learning models using a simplified and expressive syntax
# Why tokenising?
# Every word in the dataset is given an unique corresponding token so that each token can be used to train our Machine Learning Model. We can decide what our token would be so here we are usng only most freqently used vocabulary words and rest are tokenised as unknown

tokeniser = Tokenizer(num_words=10000, oov_token='<UNK>')
# Here we used only 10000 most frequently used words and rest are tokenised as <UNK>
tokeniser.fit_on_texts(tweets)  # Maps the words to numeric tokens
# Lets see our Tokens
print(tweets[0], '\n', tokeniser.texts_to_sequences([tweets[0]]))

# The model which I am going to create needs a fixed input shape but the tweets may not be in a fixed shape so i will be Padding and truncating the sequences
length = []
for i in tweets:
  length.append(len(i.split(' ')))
plt.hist(length, bins=len(set(length)), edgecolor='black')
plt.xlabel("Number of words")
plt.ylabel("Number of tweets")
plt.show()  # Here is the uneven distribution of lengths

# I want every tweet above the length 50 words get truncated and every tweet less than 50 words get padded
max_len = 50

from tensorflow.keras.preprocessing.sequence import pad_sequences
def get_sequences(tokeniser, tweets):
  sequences = tokeniser.texts_to_sequences(tweets)
  padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
  return padded

padded_train_sequences = get_sequences(tokeniser, tweets)

# Lets see how is it working
print(tweets[0], '\n', padded_train_sequences[0])

# Lets see how many number of examples do we have for each class of a label
plt.hist(labels, bins=11, color= 'red', edgecolor='black')
plt.xlabel("Labels")
plt.ylabel("Number of examples")
plt.show()

# I have made the labels accessible from both sides index and emotions in two different dictionaries
keys = set(labels)
values = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']
class_to_index = dict(zip(values, keys))
index_to_class = dict(zip(keys, values))

#Lets see them
print(class_to_index)
print(index_to_class)

# Making the Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=max_len),
    # Embedding here maps discrete tokens to continuous vectors of fixed size. It learns to represent words in a continuous vector space
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
    # A bidirectional layer processes the input data in both forward and backward direction and it wraps LSTM layer in it
    # The primary advantage of LSTM networks lies in their ability to effectively learn and remember information over long sequences
    # return_sequences=True means in this case, the Bidirectional LSTM layer will return the full sequence of outputs for each input sequence
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    # Here, the Bidirectional LSTM layer will return only the output corresponding to the last time step for each input sequence
    tf.keras.layers.Dense(6, activation='softmax')
    # Softmax is a generalisation of a logistic regression (which is a binary classification algorithm) to the multiclass classification contexts.
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-2),
    metrics=['accuracy']
    # ADAM stands for ADAptive Moment estimation. It can help in increasing or decreasing the learning rate accordingly
)

model.summary()

# Training the model
val_tweets, val_labels = get_tweets(valid)
val_seq = get_sequences(tokeniser, val_tweets)

train_labels = np.array(labels)
padded_train_sequences = np.array(padded_train_sequences)
val_seq = np.array(val_seq)
val_labels = np.array(val_labels)
h = model.fit(
    padded_train_sequences, train_labels,
    validation_data=(val_seq, val_labels),
    epochs=20,
    # During each epoch, the neural network processes every training example once, updating the model's weights based on the computed error or loss.
    verbose=0,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
    ]
    # It defines a callback for early stopping during the training of a neural network to prevent overfitting
    # In this case, if the validation accuracy does not improve for 2 consecutive epochs, training will be stopped early.
    # Setting verbose to different values will determine the level of detail shown in the output.
)

# Evaluating the model
show_history(h)
test_tweets, test_labels = get_tweets(test)
test_sequence = get_sequences(tokeniser, test_tweets)
test_labels = np.array(test_labels)

_=model.evaluate(test_sequence, test_labels)

# Checking
i = random.randint(0, len(test_labels)-1)
print('Sequence: ', test_tweets[i])
print('Emotion: ', index_to_class[test_labels[i]])
p = model.predict(np.expand_dims(test_sequence[i], axis=0))[0]
pred_class = index_to_class[np.argmax(p).astype('uint8')]
print("Predicted emotion: ", pred_class)

probas = model.predict(test_sequence)
preds = np.argmax(probas, axis=-1)
show_confusion_matrix(test_labels, preds, classes=['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise'])
# Confusion Matrix provides a summary of the predictions made by the model compared to the actual true values