import tensorflow as tf
import tensorflow_datasets as tfds
from tf.keras.preprocessing.text import text_to_word_sequence
from tf.keras.preprocessing.sequence import pad_sequences
from tf.keras.preprocessing.text import Tokenizer
from tf.keras import layers, Sequential
from tf.keras.callbacks import EarlyStopping



#loading the data

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


#Using 1D Convolution

#reminder: size of embedding space = size of the vector representing each word
embedding_size = 100

model_cnn = Sequential([
    layers.Embedding(input_dim=vocab_size+1, input_length=150, output_dim=embedding_size, mask_zero=True),
    layers.Conv1D(20, kernel_size=3),
    layers.Flatten(),
    layers.Dense(1, activation="sigmoid"),
])

model_cnn.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model_cnn.summary()


es = EarlyStopping(patience=5, restore_best_weights=True)

model_cnn.fit(X_train_pad, y_train,
          epochs=20,
          batch_size=32,
          validation_split=0.3,
          callbacks=[es]
         )


res = model_cnn.evaluate(X_test_pad, y_test, verbose=0)

print(f'The accuracy evaluated on the test set is of {res[1]*100:.3f}%')
