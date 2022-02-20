import numpy as np
import pandas as pd
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from src.pipeline import BaseDetectorPipeline

class ClassificationPipeline(BaseDetectorPipeline):
    
    def __init__(self, **kwargs):
        super().__init__(
            user_features=[
                'statuses_count',
                'favourites_count',
                'listed_count',
                'updated',
                'created_at',
                'friends_count',
                'followers_count',
            ],
            tweet_metadata_features=[
                'favorite_count',
                'retweet_count',
                'num_urls',
                'source'
            ],
            **kwargs
        )

    def feature_engineering_u(self, user_df, training):
        return_df = user_df[['statuses_count', 'favourites_count', 'listed_count']]
        if 'updated' in user_df.columns:
            age = (
                pd.to_datetime(user_df['updated']) - 
                pd.to_datetime(user_df['created_at']).dt.tz_localize(None)
            ) / np.timedelta64(1, 'Y')
        else:
            age = (
                pd.to_datetime(pd.to_datetime('today')) - 
                pd.to_datetime(user_df['created_at']).dt.tz_localize(None)
            ) / np.timedelta64(1, 'Y')
        return_df.loc[:, 'age'] = age
        # Add 1 on denominator to avoid zero division
        return_df.loc[:, 'follower_to_friend_ratio'] = user_df['followers_count'] / (user_df['friends_count'] + 1)
        return_df.loc[:, 'tweet_frequency'] = user_df['statuses_count'] / age
        return_df.loc[:, 'id'] = user_df['id']
        return_df.loc[:, 'label'] = user_df['label']
        return return_df

    def feature_engineering_ts(self, metadata_df, training):
        df_grouped = metadata_df.groupby('user_id')
        df_return = pd.DataFrame()
        tweet_count = df_grouped['user_id'].count()
        favorite_received = df_grouped['favorite_count'].sum()
        df_return.loc[:, 'favorite_received'] = favorite_received
        # Add 1 on denominator to avoid zero division
        df_return.loc[:, 'favorite_received_ratio'] = favorite_received / (tweet_count + 1)

        retweet_received = df_grouped['retweet_count'].sum()
        df_return.loc[:, 'retweet_received'] = retweet_received
        df_return.loc[:, 'retweet_received_ratio'] = retweet_received / (tweet_count + 1)

        df_return.loc[:, 'url_count'] = df_grouped['num_urls'].sum()
        df_return.loc[:, 'activity_source_count'] = df_grouped['source'].nunique()
        df_return = df_return.reset_index()
        return df_return

    def concatenate(self, user_df: Optional[pd.DataFrame], tweet_df: Optional[pd.DataFrame], tweet_metadata_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if tweet_metadata_df is not None:
            df = pd.merge(
                left=user_df,
                right=tweet_metadata_df,
                how='left',
                left_on='id',
                right_on='user_id'
            ).fillna(0.0).drop('user_id', axis=1)
            return df
        else:
            return user_df

    def classify(self, X_train, X_dev, y_train, y_dev):
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
