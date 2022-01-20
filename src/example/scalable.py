from typing import Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer
from src.pipeline import BaseDetectorPipeline
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.util import trigrams
from nltk.lm import MLE


class ScalablePipeline(BaseDetectorPipeline):

    def __init__(self, **kwargs):
        super().__init__(
            dataset_name='MIB',
            user_features=[
                'statuses_count',
                'followers_count',
                'friends_count',
                'favourites_count',
                'listed_count',
                'default_profile',
                'profile_use_background_image',
                'verified',
                'updated',
                'created_at',
                'screen_name',
                'name',
                'description'
            ],
            **kwargs
        )
    
    ############# CUSTOM FUNCTIONS #######################
    def get_screen_name_likelihood(self, series):
        sequence = series.apply(lambda x: list(x.lower())).values.tolist()
        return self.get_likelihood_array(sequence)
        
    def get_likelihood_array(self, sequence, n=3):
        train_data, padded_sent = padded_everygram_pipeline(n, sequence)
        mle = MLE(n)
        mle.fit(train_data, padded_sent)
        
        s = np.zeros((len(sequence),))
        for i, name in enumerate(sequence):
            tri = trigrams(pad_both_ends(name, n=3))
            total_score = 1
            count = 0
            for ele in tri:
                score = mle.score(ele[2], [ele[0], ele[1]])
                total_score *= score
                count += 1
            s[i] = total_score ** (1/count)
        return s
    ##################################################

    def feature_engineering_u(self, user_df):
        age = (
            pd.to_datetime(user_df['updated']) - 
            pd.to_datetime(user_df['created_at']).dt.tz_localize(None)
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
        user_df['screen_name_likelihood'] = self.get_screen_name_likelihood(user_df.pop('screen_name'))
        return user_df.fillna(0.0)

    def classify(self, X_train, X_dev, y_train, y_dev):
        self.transformer = FunctionTransformer(np.log1p, validate=True)
        self.classifier = KNeighborsClassifier(9)
        X_train = self.transformer.transform(X_train)
        self.classifier.fit(X_train, y_train)
    
    def predict(self, X_test):
        X_test = self.transformer.transform(X_test)
        y_pred = self.classifier.predict(X_test)
        return y_pred