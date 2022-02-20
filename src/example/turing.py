from typing import Optional
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.pipeline import BaseDetectorPipeline


class TuringPipeline(BaseDetectorPipeline):
    
    def __init__(self):
        super().__init__(
            tweet_metadata_features=[
                'num_hashtags',
                'num_mentions',
                'num_urls',
                'favorite_count',
                'retweet_count',
                'timestamp'
            ],
            use_tweet=True
        )

    def feature_engineering_ts(self, metadata_df, training):
        df_grouped = metadata_df.groupby(['user_id', 'label'])
        df_return = pd.DataFrame()

        df_return['avg_hashtags'] = df_grouped['num_hashtags'].mean()
        df_return['avg_mentions'] = df_grouped['num_mentions'].mean()
        df_return['avg_urls'] = df_grouped['num_urls'].mean()
        df_return['favorites_received'] = df_grouped['favorite_count'].sum()
        df_return['avg_retweets_received'] = df_grouped['retweet_count'].mean()

        tweet_same_time = metadata_df.groupby(['user_id', 'timestamp'])['timestamp'].agg(['count', 'nunique'])
        # Add 1 to avoid zero dimension
        df_return['avg_tweet_same_time'] = (tweet_same_time['count'] - tweet_same_time['nunique']) / (tweet_same_time['count'] + 1)
        df_return.reset_index(inplace=True)
        return df_return

    def semantic_encoding(self, tweet_df, training):
        tweet_df['text_len'] = tweet_df['text'].astype(str).apply(lambda x: len(x)) 
        df_return = pd.DataFrame()
        character_per_tweet = tweet_df.groupby(['user_id', 'label'])['text_len']
        df_return['avg_characters'] = character_per_tweet.mean()
        df_return['std_characters'] = character_per_tweet.std()
        df_return.reset_index(inplace=True)
        df_return.fillna(0.0, inplace=True)
        return df_return

    def concatenate(self, user_df: Optional[pd.DataFrame], tweet_df: Optional[pd.DataFrame], tweet_metadata_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if tweet_metadata_df is None:
            return tweet_df
        else:
            merged_df = pd.merge(tweet_metadata_df, tweet_df, how='left', on=['user_id', 'label']).fillna(0.0)
            return merged_df

    def classify(self, X_train, X_dev, y_train, y_dev):
        self.classifier = RandomForestClassifier()
        self.classifier.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = self.classifier.predict(X_test)
        return y_pred

    
    