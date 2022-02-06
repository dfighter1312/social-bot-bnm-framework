import re
import numpy as np
import tensorflow as tf
from src.pipeline import BaseDetectorPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.ensemble import AdaBoostClassifier


class DeepAccountLevelPipeline(BaseDetectorPipeline):
    
    def __init__(self, **kwargs):
        super().__init__(
            user_features=[
                'statuses_count',
                'followers_count',
                'friends_count',
                'favourites_count',
                'listed_count',
                'default_profile',
                'geo_enabled',
                'profile_use_background_image',
                'verified',
                'protected'
            ],
            **kwargs
        )

    def feature_engineering_u(self, user_df, training):
        return user_df.replace('N', np.nan).fillna(0.0)

    def classify(self, X_train, X_dev, y_train, y_dev):
        self.smote = SMOTE()
        smote_X, smote_y = self.smote.fit_resample(X_train, y_train)

        self.e = EditedNearestNeighbours()
        r_X, r_y = self.e.fit_resample(smote_X, smote_y)

        self.a = AdaBoostClassifier(random_state=0)
        self.a.fit(r_X, r_y)

    def predict(self, X_test):
        return self.a.predict(X_test)
    

class DeepTweetLevelPipeline(BaseDetectorPipeline):
    def __init__(self, **kwargs):
        super().__init__(
            use_tweet=True,
            tweet_metadata_features=[
                "favorite_count",
                "reply_count",
                "retweet_count",
                "num_hashtags",
                "num_urls"
            ],
            account_level=False,
            **kwargs
        )

    def isAllCaps(self, word):
        for c in word:
            if c.islower() or not c.isalpha():
                return False
        return True

    def hasRepeatedLetters(self, word):
        prev = ''
        prev2 = ''
        for c in word:
            if c == prev:
                if c == prev2:
                    return True
            prev2 = prev
            prev = c
        return False
        
    def text_tags(self, row):
        URL_PATTERN = "^((http[s]?|ftp):\/)?\/?([^:\/\s]+)((\/\w+)*\/)([\w\-\.]+[^#?\s]+)(.*)?(#[\w\-]+)?$"
        rowlist = str(row).split()
        rowlist = [word.strip() for word in rowlist]
        rowlist = [word if not word.strip().startswith(
            '#') else "hashtag-tag" for word in rowlist]
        rowlist = [word if not word.strip().startswith(
            '@') else "user-tag" for word in rowlist]
        rowlist = [word if not self.isAllCaps(
            word.strip()) else word.lower() + " allcaps-tag" for word in rowlist]
        rowlist = [word if not self.hasRepeatedLetters(
            word.strip()) else word + " repeated-tag" for word in rowlist]
        rowlist = [word.lower() for word in rowlist]
        rowlist = [re.sub(URL_PATTERN, "url-tag", word) for word in rowlist]
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
        tweet_df = tweet_df.apply(self.text_tags)
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

    def lstm_glove_model(self, input_shape, metadata_input_shape) -> tf.keras.Model:
        X_indices = tf.keras.Input(input_shape)
        embeddings = self.embedding_layer(X_indices)
        lstm_cell = tf.keras.layers.LSTM(32, return_sequences=True)(embeddings)
        flatten = tf.keras.layers.Flatten()(lstm_cell)
        aux_output = tf.keras.layers.Dense(1, activation='sigmoid')(flatten)
        metadata = tf.keras.Input(metadata_input_shape)
        concat = tf.keras.layers.concatenate([flatten, metadata], axis=1)
        dense_1 = tf.keras.layers.Dense(128, activation='relu')(concat)
        dense_2 = tf.keras.layers.Dense(64, activation='relu')(dense_1)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(dense_2)
        model = tf.keras.Model(inputs=[X_indices, metadata], outputs=[output, aux_output])
        return model

    def classify(self, X_train, X_dev, y_train, y_dev):
        X_train_indices = self.tokenizer.texts_to_sequences(X_train["text"])
        X_train_indices = tf.keras.preprocessing.sequence.pad_sequences(X_train_indices, maxlen=self.maxLen, padding='post')
        X_train_metadata = X_train.drop("text", axis=1)

        X_dev_indices = self.tokenizer.texts_to_sequences(X_dev["text"])
        X_dev_indices = tf.keras.preprocessing.sequence.pad_sequences(X_dev_indices, maxlen=self.maxLen, padding='post')
        X_dev_metadata = X_dev.drop("text", axis=1)

        self.model = self.lstm_glove_model(input_shape=(self.maxLen,), metadata_input_shape=(X_dev_metadata.shape[1],))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], loss_weights=[0.8, 0.2])
        self.model.fit(
            [X_train_indices, X_train_metadata],
            y_train,
            batch_size=32,
            epochs=5,
            validation_data=[[X_dev_indices, X_dev_metadata], y_dev],
            verbose=2
        )
        self.model.save('ckpts')
    
    def predict(self, X_test):
        X_test_indices = self.tokenizer.texts_to_sequences(X_test["text"])
        X_test_indices = tf.keras.preprocessing.sequence.pad_sequences(X_test_indices, maxlen=self.maxLen, padding='post')
        X_test_metadata = X_test.drop("text", axis=1)
        y_pred = self.model.predict([X_test_indices, X_test_metadata])[0]
        return y_pred