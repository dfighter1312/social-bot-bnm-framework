from src.pipeline import BaseDetectorPipeline
import re
import numpy as np
import tensorflow as tf
from src.pipeline import BaseDetectorPipeline
import pandas as pd
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.util import bigrams
from nltk.lm import MLE
import time
from typing import List, Optional, Union
from src.data_read import LocalFileReader
# TODO: rewrite this using the new data_read functions!

class BaseEDAPipeline(BaseDetectorPipeline):
    '''
        Base Detector Pipeline Copy and Minorly Changed!
    '''

    def __init__(self, **kwargs):
        super().__init__(
            user_features='all',
            **kwargs
        )


    def get_data(self, dataset_name: str = None):
        """Receive the dataset"""
        if dataset_name == 'MIB':
            config = self.local_file_reader.get_mib_config()

            # TODO: check this!
            self.dfs = self.local_file_reader.read_mib_no_split(
                config,
                self.label_col,
                self.use_user,
                self.use_tweet,
                self.use_tweet_metadata
            )
            # Turn off network since there is no usage
            self.use_network = False
        elif dataset_name == 'TwiBot-20':
            # * Not worked on
            raise NotImplementedError
        else:
            raise ValueError(
                "dataset_name must be MIB or TwiBot-20. Contact the "
                "author if you need any updates. "
            )
    def preprocess_train(self):
        # Step 2: Feature Selection
        step_3_start = time.time()
        if self.verbose:
            print('Selecting features...')
        self.dfs = self.feature_selection(warn=True, **self.dfs)

        # Step 3A: Semantic Encoder (optional)
        if self.verbose:
            print('Featuring the data...')
        if self.use_tweet:
            self.dfs['tweet_df'] = self.semantic_encoding(
                self.dfs['tweet_df'],
                training=True
            )

        # Step 3B: Feature engineering (optional)
        if self.use_user:
            self.dfs['user_df'] = self.feature_engineering_u(
                self.dfs['user_df'],
                training=True
            )
            # Do a type check to ensure only numeric values remain
            self.dfs['user_df'] = self.type_check(
                self.dfs['user_df'],
                warn=True
            )
        if self.use_tweet_metadata:
            self.dfs['tweet_metadata_df'] = self.feature_engineering_ts(
                self.dfs['tweet_metadata_df'],
                training=True
            )
            # Do a type check to ensure only numeric values remain
            self.dfs['tweet_metadata_df'] = self.type_check(
                self.dfs['tweet_metadata_df'],
                warn=True
            )
        step_3_end = time.time()
        # Step 3C: Concatenate the features
        self.dfs = self.concatenate(**self.dfs)
        return step_3_start, step_3_end

    def preprocess(self, set_name = None):
        start, end = self.preprocess_train()
        if self.verbose:
            print('Preprocessing time: ', end - start)

    def run(
        self,
        dataset_name: str = 'MIB',
    ):
        raise NotImplementedError

class EDAPipeline(BaseEDAPipeline):
    
    def __init__(self, **kwargs):
        super().__init__(
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
        URL_PATTERN = r"[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
        rowlist = str(row).split()
        rowlist = [word.strip() for word in rowlist]
        rowlist = [word if not word.strip().startswith(
            '#') else "hashtagtag" for word in rowlist]
        rowlist = [word if not word.strip().startswith(
            '@') else "usertag" for word in rowlist]
        rowlist = [word if not self.isAllCaps(
            word.strip()) else word.lower() + " allcapstag" for word in rowlist]
        rowlist = [word if not self.hasRepeatedLetters(
            word.strip()) else word + " repeatedtag" for word in rowlist]
        rowlist = [word.lower() for word in rowlist]
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
        tweet_df["text"] = tweet_df["text"].apply(self.text_tags)
        if training:
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
            self.tokenizer.fit_on_texts(tweet_df["text"])
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
    
    def get_screen_name_likelihood(self, series, training):
        sequence = series.apply(lambda x: list(x.lower())).values.tolist()
        return self.get_likelihood_array(sequence, training)
        
    def get_likelihood_array(self, sequence, training, n=2):
        if training:
            train_data, padded_sent = padded_everygram_pipeline(n, sequence)
            self.mle = MLE(n)
            self.mle.fit(train_data, padded_sent)
        
        s = np.zeros((len(sequence),))

        for i, name in enumerate(sequence):
            bi = bigrams(pad_both_ends(name, n=2))
            total_score = 1
            count = 0
            for ele in bi:
                score = self.mle.score(ele[1], [ele[0]])
                total_score *= score
                count += 1
            s[i] = total_score ** (1/count)
        return s

    def feature_engineering_u(self, user_df, training):
        if 'updated' in user_df.columns:
            age = (
                pd.to_datetime(user_df.loc[:, 'updated']) - 
                pd.to_datetime(user_df.loc[:, 'created_at']).dt.tz_localize(None)
            ) / np.timedelta64(1, 'Y')
        else:
            age = (
                pd.to_datetime(pd.to_datetime('today')) - 
                pd.to_datetime(user_df.loc[:, 'created_at']).dt.tz_localize(None)
            ) / np.timedelta64(1, 'Y')
        user_df['tweet_freq'] = user_df['statuses_count'] / age
        user_df['followers_growth_rate'] = user_df['followers_count'] / age
        user_df['friends_growth_rate'] = user_df['friends_count'] / age
        user_df['favourites_growth_rate'] = user_df['favourites_count'] / age
        user_df['listed_growth_rate'] = user_df['listed_count'] / age
        user_df['followers_friends_ratio'] = user_df['followers_count'] / np.maximum(user_df['friends_count'], 1)
        user_df['screen_name_length'] = user_df['screen_name'].str.len()
        user_df['num_digits_in_screen_name'] = user_df['screen_name'].str.count('\d')
        user_df['name_length'] = user_df['name'].str.len()
        user_df['num_digits_in_name'] = user_df['name'].str.count('\d')
        user_df['description_length'] = user_df['description'].str.len()
        user_df['screen_name_likelihood'] = self.get_screen_name_likelihood(user_df.pop('screen_name'), training)
        user_df['age'] = age
        user_df['reputation'] = user_df['followers_count'] / (user_df['followers_count'] + user_df['friends_count'])
        return user_df.fillna(0.0)
