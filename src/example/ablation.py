import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional, Union, List

from matplotlib.style import use
from src.pipeline import BaseDetectorPipeline


class AblationPipeline(BaseDetectorPipeline):
    
    def __init__(
        self,
        units: Union[int, List[int]],
        dl_types: Union[str, List[str]],
        encoder: str,
        use_tweet: bool = True,
        use_tweet_metadata: bool = False,
        use_users: bool = False,
        num_layers: Optional[int] = None,
        verbose_structure: bool = True,
        max_features: int = 5000,
        epochs: int = 3,
        batch_size: int = 32,
        normalize: bool = False
    ):
        # Type check first
        if isinstance(units, list) and isinstance(dl_types, list):
            if len(units) != len(dl_types):
                raise ValueError('Number of units and DL types must be identical')
        elif isinstance(units, list):
            num_layers = len(units)
        elif isinstance(dl_types, list):
            num_layers = len(dl_types)
        elif num_layers is None:
            raise ValueError('num_layers is None but cannot find any list in units or dl_types')
        
        if not (use_tweet or use_tweet_metadata):
            raise ValueError('Tweet or tweet metadata must be used in this ablation study')
        
        self.units = units
        self.dl_types = dl_types
        self.num_layers = num_layers
        self.encoder = encoder
        self.max_features = max_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.normalize = normalize
        tweet_metadata_features = [
            "retweet_count",
            "reply_count",
            "favorite_count",
            "favorited",
            "retweeted",
            "num_hashtags",
            "num_urls",
            "num_mentions"
        ]
        super().__init__(
            user_features='all' if use_users else None,
            tweet_metadata_features=tweet_metadata_features if use_tweet_metadata else None,
            use_tweet=use_tweet,
            account_level=False
        )

    def semantic_encoding(self, tweet_df, training):
        tweet_df["text"] = tweet_df["text"].apply(self.text_tags)
        if self.encoder == 'glove':
            return self.semantic_encoding_glove(tweet_df, training)
        elif self.encoder == 'tfidf':
            return self.semantic_encoding_tfidf(tweet_df, training)
        elif self.encoder == 'word2vec':
            return self.semantic_encoding_word2vec(tweet_df, training)
        else:
            raise ValueError('tokenizer: %s was not available' % (self.encoder))

    def classify(self, X_train, X_dev, y_train, y_dev):
        if self.encoder in ['glove', 'word2vec']:
            X_train = self.tokenizer.texts_to_sequences(X_train["text"])
            X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=self.maxLen, padding='post')

            X_dev = self.tokenizer.texts_to_sequences(X_dev["text"])
            X_dev = tf.keras.preprocessing.sequence.pad_sequences(X_dev, maxlen=self.maxLen, padding='post')
        elif self.encoder in ['tfidf']:
            X_train = X_train.values
            X_dev = X_dev.values
        self.model = self.create_model()
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy', metrics=['accuracy']
        )
        self.model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=[X_dev, y_dev],
        )
        self.model.save('ckpts')

    def predict(self, X_test):
        if self.encoder in ['glove', 'word2vec']:
            X_test = self.tokenizer.texts_to_sequences(X_test["text"])
            X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=self.maxLen, padding='post')
        elif self.encoder in ['tfidf']:
            X_test = X_test.values
        y_pred = self.model.predict(X_test)
        return y_pred

    def create_model(self):
        model = tf.keras.Sequential()
        if self.encoder in ['glove', 'word2vec']:
            model.add(self.embedding_layer)
        elif self.encoder in ['tfidf']:
            pass

        if not isinstance(self.units, list):
            self.units = [self.units for i in range(self.num_layers)]
        if not isinstance(self.dl_types, list):
            self.dl_types = [self.dl_types for i in range(self.num_layers)]

        for unit, type in zip(self.units, self.dl_types):
            if type == 'cnn':
                model.add(tf.keras.layers.Conv1D(unit, kernel_size=3))
            elif type == 'lstm':
                model.add(tf.keras.layers.LSTM(unit, return_sequences=True))
            elif type == 'bilstm':
                model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(unit, return_sequences=True)))
            elif type == 'dense':
                model.add(tf.keras.layers.Dense(unit))
            elif type == 'flatten':
                model.add(tf.keras.layers.Flatten())
        
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        return model
    
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

    def semantic_encoding_glove(self, tweet_df, training):
        if training:
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
            self.tokenizer.fit_on_texts(tweet_df)
            words_to_index = self.tokenizer.word_index

            word_to_vec_map = self.read_glove_vector('glove/glove.twitter.27B.50d.txt')

            self.maxLen = 280
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

    def semantic_encoding_tfidf(self, tweet_df, training):
        if training:
            self.tfidf = TfidfVectorizer(max_features=self.max_features)
            self.tfidf.fit(tweet_df['text'])
            X = self.tfidf.transform(tweet_df['text'])
            df_trans = pd.DataFrame(
                X.toarray()
            )
            tweet_df = pd.concat([tweet_df, df_trans], axis=1).drop('text', axis=1)
            return tweet_df
        else:
            X = self.tfidf.transform(tweet_df['text'])
            df_trans = pd.DataFrame(
                X.toarray()
            )
            tweet_df = pd.concat([tweet_df, df_trans], axis=1).drop('text', axis=1)
            return tweet_df

    def semantic_encoding_word2vec(self, tweet_df, training):
        pass

    def feature_engineering_u(self, user_df, training):
        """Perform normalization"""
        if self.normalize:
            uid = user_df.pop('id')
            label = user_df.pop('label')
            user_df = user_df.fillna(0)
            if training:
                self.user_mean = user_df.mean(axis=0)
                self.user_std = user_df.std(axis=0)
                user_df = (user_df - self.user_mean) / self.user_std
            else:
                user_df = (user_df - self.user_mean) / self.user_std
            user_df['id'] = uid
            user_df['label'] = label
            return user_df.dropna(axis=1)
        else:
            return user_df.fillna(0)