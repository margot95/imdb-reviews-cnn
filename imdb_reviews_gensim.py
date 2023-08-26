import gensim.downloader as api
import tensorflow_datasets as tfds
from tf.keras.preprocessing.text import text_to_word_sequence
from tf.keras.preprocessing.sequence import pad_sequences
from tf.keras import layers, Sequential
from tf.keras.preprocessing.text import Tokenizer
from tf.keras.callbacks import EarlyStopping


import numpy as np


def load_data(percentage_of_sentences=None):
    train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"], batch_size=-1, as_supervised=True)

    train_sentences, y_train = tfds.as_numpy(train_data)
    test_sentences, y_test = tfds.as_numpy(test_data)

    # Take only a given percentage of the entire data
    if percentage_of_sentences is not None:
        assert(percentage_of_sentences> 0 and percentage_of_sentences<=100)

        len_train = int(percentage_of_sentences/100*len(train_sentences))
        train_sentences, y_train = train_sentences[:len_train], y_train[:len_train]

        len_test = int(percentage_of_sentences/100*len(test_sentences))
        test_sentences, y_test = test_sentences[:len_test], y_test[:len_test]

    X_train = [text_to_word_sequence(_.decode("utf-8")) for _ in train_sentences]
    X_test = [text_to_word_sequence(_.decode("utf-8")) for _ in test_sentences]

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data(percentage_of_sentences=10)

##repeating same steps as in imdb_reviews_CNN.py for tokenizing and padding
#tokenizing
tk = Tokenizer()
X_train_tk= tk.fit_on_texts(X_train)
X_test_tk= tk.fit_on_texts(X_test)
vocab_size = len(tk.word_index)
print(f'There are {vocab_size} different words in your corpus')
X_train_token = tk.texts_to_sequences(X_train)
X_test_token = tk.texts_to_sequences(X_test)
#padding
X_train_pad = pad_sequences(X_train_token, dtype='float32', padding='post', maxlen=150)
X_test_pad = pad_sequences(X_test_token, dtype='float32', padding='post', maxlen=150)


print(list(api.info()['models'].keys()))
word2vec_transfer = api.load('glove-wiki-gigaword-50')

#convert a sentence (sequence of words) into a matrix representing the words in the embedding space
def embed_sentence_with_TF(word2vec, sentence):
    embedded_sentence = []
    for word in sentence:
        if word in word2vec:
            embedded_sentence.append(word2vec[word])

    return np.array(embedded_sentence)

#converts a list of sentences into a list of matrices
def embedding(word2vec, sentences):
    embed = []

    for sentence in sentences:
        embedded_sentence = embed_sentence_with_TF(word2vec, sentence)
        embed.append(embedded_sentence)

    return embed

#Embed the training and test sentences
X_train_embed_2 = embedding(word2vec_transfer, X_train_pad)
X_test_embed_2 = embedding(word2vec_transfer, X_train_pad)

#padding
X_train_pad_2 = pad_sequences(X_train_embed_2, dtype='float32', padding='post', maxlen=150)
X_test_pad_2 = pad_sequences(X_test_embed_2, dtype='float32', padding='post', maxlen=150)
X_train_pad_2.shape

embedding_size = 50

def init_cnn_model_2():
    model = Sequential()
    model.add(layers.Conv1D(16, 3))
    model.add(layers.Flatten())
    model.add(layers.Dense(5,))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model_cnn_2 = init_cnn_model_2()


es_2 = EarlyStopping(patience=5, restore_best_weights=True)

model_cnn_2.fit(X_train_pad_2, y_train,
          epochs=20,
          batch_size=32,
          validation_split=0.3,
          callbacks=[es_2]
         )


res = model_cnn_2.evaluate(X_test_pad_2, y_test, verbose=0)

print(f'The accuracy evaluated on the test set is of {res[1]*100:.3f}%')
