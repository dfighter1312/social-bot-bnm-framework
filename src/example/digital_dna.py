import sys
import zlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.pipeline import BaseDetectorPipeline


class DnaPipeline(BaseDetectorPipeline):
    """Reference: https://github.com/pasricha/bot-dna-compression"""
    
    def __init__(self, **kwargs):
        super().__init__(
            tweet_metadata_features=[
                'retweeted_status_id',
                'in_reply_to_status_id',
                'timestamp'
            ]
        )

    def create_dna_from_tweets(self, tweets_df):
        '''For each user id in tweets_df return a digital DNA string based on posting behaviour.'''
        
        # Add columns for counts of tweets, replies and retweets.
        tweets_df['num_retweets'] = np.where(tweets_df['retweeted_status_id'] == 0, 0, 1)
        tweets_df['num_replies'] = np.where(tweets_df['in_reply_to_status_id'] == 0, 0, 1)
        tweets_df['num_tweets'] = np.where((tweets_df['num_retweets'] == 0) & (tweets_df['num_replies'] == 0), 1, 0)

        # DNA alphabet for tweet (A), retweet (C) and reply (T).
        tweets = tweets_df['num_tweets'] == 1
        retweets = tweets_df['num_retweets'] == 1
        replies = tweets_df['num_replies'] == 1

        tweets_df.loc[:, 'DNA'] = np.where(retweets, 'C', np.where(replies, 'T', 'A'))

        # Sort tweets by timestamp..
        tweets_df = tweets_df[['user_id', 'label', 'timestamp', 'DNA']]
        tweets_df = tweets_df.sort_values(by=['timestamp'])

        # Create digital DNA string for each user account.
        dna = tweets_df.groupby(by=['user_id', 'label'])['DNA'].agg(lambda x: ''.join(x))
        
        return dna

    def compress_dna_df(self, dna):
        '''Return a dataframe with compression facts for a series of dna.'''

        # Convert DNA in string object to bytes object.
        dna_bytes = dna.apply(lambda s: s.encode('utf-8'))

        # Run compression on each DNA string in the sample.
        dna_compressed = dna_bytes.apply(lambda b: zlib.compress(b))

        # Create dataframe with compression facts.
        dna_df = pd.DataFrame({'dna': dna,
                            'original_dna_size': dna_bytes.apply(sys.getsizeof), 
                            'compressed_dna_size': dna_compressed.apply(sys.getsizeof)})
        
        dna_df['compression_ratio'] = dna_df['original_dna_size'] / dna_df['compressed_dna_size']
        
        return dna_df

    def feature_engineering_ts(self, metadata_df, training):
        dna = self.create_dna_from_tweets(metadata_df)
        dna = self.compress_dna_df(dna)
        dna.reset_index(inplace=True)
        return dna[['original_dna_size', 'compression_ratio', 'label', 'user_id']]

    def classify(self, X_train, X_dev, y_train, y_dev):
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)