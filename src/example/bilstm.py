import re
import numpy as np
import tensorflow as tf
from src.pipeline import BaseDetectorPipeline

class BidirectionalLSTMPipeline(BaseDetectorPipeline):
    
    def __init__(self):
        super().__init__(
            use_tweet=True,
            account_level=False
        )

    def text_tags(self, row):
        URL_PATTERN = r"[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
        rowlist = str(row).split()
        rowlist = [word.strip().lower() for word in rowlist]
        rowlist = [word if not word.strip().startswith(
            '#') else "hashtagtag" for word in rowlist]
        rowlist = [word if not word.strip().startswith(
            '@') else "usertag" for word in rowlist]
        rowlist = [re.sub(URL_PATTERN, "urltag", word) for word in rowlist]
        return " ".join(rowlist)
    
    def read_glove_vector(self, glove_vec):
        with open(glove_vec, 'r', encoding='UTF-8') as f:
            words = set()
            word_to_vec_map = {}
            for line in f:
                w_line = line.split()
                curr_word = w_line[0]
                word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
        return word_to_vec_map

    def semantic_encoding(self, tweet_df, training):
        tweet_df = tweet_df["text"].apply(self.text_tags)
        if training:
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
            self.tokenizer.fit_on_texts(tweet_df)
            words_to_index = self.tokenizer.word_index

            word_to_vec_map = self.read_glove_vector('glove/glove.twitter.27B.50d.txt')

            self.maxLen = 100
            embed_vector_len = 50
            vocab_len = len(words_to_index) + 1

            emb_matrix = np.zeros((vocab_len, embed_vector_len))

            for word, index in words_to_index.items():
                embedding_vector = word_to_vec_map.get(word)
                if embedding_vector is not None:
                    emb_matrix[index, :] = embedding_vector

            self.embedding_layer = tf.keras.layers.Embedding(
                input_dim=vocab_len,
                output_dim=embed_vector_len,
                input_length=self.maxLen,
                weights = [emb_matrix],
                trainable=False
            )
        return tweet_df

    def bi_lstm_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential([
            self.embedding_layer,
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True, dropout=0.3)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True, dropout=0.3)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, dropout=0.3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        return model

    def classify(self, X_train, X_dev, y_train, y_dev):
        X_train_indices = self.tokenizer.texts_to_sequences(X_train["text"])
        X_train_indices = tf.keras.preprocessing.sequence.pad_sequences(X_train_indices, maxlen=self.maxLen, padding='post')

        X_dev_indices = self.tokenizer.texts_to_sequences(X_dev["text"])
        X_dev_indices = tf.keras.preprocessing.sequence.pad_sequences(X_dev_indices, maxlen=self.maxLen, padding='post')

        y_train = np.array([y_train, 1 - y_train]).T
        y_dev = np.array([y_dev, 1 - y_dev]).T

        self.model = self.bi_lstm_model()
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
            loss='categorical_crossentropy', metrics=['accuracy']
        )
        self.model.fit(
            X_train_indices,
            y_train,
            batch_size=64,
            epochs=3,
            validation_data=[X_dev_indices, y_dev],
        )
        self.model.save('ckpts')
    
    def predict(self, X_test):
        X_test_indices = self.tokenizer.texts_to_sequences(X_test["text"])
        X_test_indices = tf.keras.preprocessing.sequence.pad_sequences(X_test_indices, maxlen=self.maxLen, padding='post')
        y_pred = self.model.predict(X_test_indices)
        y_pred = y_pred[:, 0]
        return y_pred