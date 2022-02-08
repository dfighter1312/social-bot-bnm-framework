from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from configparser import RawConfigParser

from sklearn.model_selection import train_test_split


class LocalFileReader:
    """Read .csv and .cfq files"""
    replace_map_dict = {
        "True": 1,
        "true": 1,
        "False": 0,
        "false": 0,
        "N": np.nan,
    }

    def __init__(self):
        self.data_parser = RawConfigParser()
        self.data_config_file = './config/data.cfg'
        self.data_parser.read(self.data_config_file)

    def get_mib_config(self) -> Dict:
        return dict(self.data_parser.items('MIB'))

    def get_twibot_config(self) -> Dict:
        return dict(self.data_parser.items('TwiBot-20'))

    def get_dataset_config(self, dataset_name) -> Dict:
        return dict(self.data_parser.items(dataset_name))

    def read_mib(
        self,
        config,
        label_column,
        use_users: bool,
        use_tweet: bool,
        use_tweet_metadata: bool
    ) -> List[Optional[pd.DataFrame]]:
        """
        Return a list of 3 optional Dataframe represents user features, tweet features and
        tweet metadata features. Each element can be a Dataframe if use_[categories] is set
        to True and None otherwise.
        """
        dfs_train = dict()
        dfs_dev = dict()
        dfs_test = dict()
        dtypes = {

        }

        # User dataframe
        paths_user = [
                'social_spambots_1_users',
                'social_spambots_2_users',
                'social_spambots_3_users',
                # 'traditional_spambots_1_users',
                # 'traditional_spambots_2_users',
                # 'traditional_spambots_3_users',
                # 'traditional_spambots_4_users',
                'fake_followers_users'
        ]
        df_bot_users = pd.concat(
            [pd.read_csv(config[path]) for path in paths_user]
        ).reset_index(drop=True)
        df_naive_users = pd.read_csv(config['genuine_users'])
        df_bot_users[label_column] = 1
        df_naive_users[label_column] = 0
        df_users = pd.concat([df_bot_users, df_naive_users], ignore_index=True)
        del(df_naive_users)
        del(df_bot_users)

        df_users_train, df_users_test, _, _ = train_test_split(
            df_users,
            df_users[label_column],
            random_state=0,
            train_size=0.8)
        df_users_dev, df_users_test, _, _ = train_test_split(
            df_users_test,
            df_users_test[label_column],
            random_state=0,
            train_size=0.5
        )
        dfs_train['user_df'] = df_users_train
        dfs_dev['user_df'] = df_users_dev
        dfs_test['user_df'] = df_users_test

        # Tweet dataframe
        if use_tweet or use_tweet_metadata:
            paths = [
                'social_spambots_1_tweets',
                'social_spambots_2_tweets',
                'social_spambots_3_tweets',
                # 'traditional_spambots_1_tweets',
                # 'traditional_spambots_2_tweets',
                # 'traditional_spambots_3_tweets',
                # 'traditional_spambots_4_tweets',
                'fake_followers_tweets'
            ]
            df_bot_tweets = pd.concat([
                pd.read_csv(
                    config[path],
                ).replace(self.replace_map_dict) for path in paths]
            ).reset_index(drop=True)
            df_naive_tweets = pd.read_csv(config['genuine_tweets'], header=None, escapechar='\\')
            df_bot_tweets[label_column] = 1
            df_naive_tweets[label_column] = 0
            df_naive_tweets.drop(12, axis=1, inplace=True)
            df_naive_tweets.columns = df_bot_tweets.columns
            df_tweets = pd.concat([df_bot_tweets, df_naive_tweets], ignore_index=True)
            
            dfs_train['tweet_metadata_df'] = pd.merge(dfs_train['user_df']['id'], df_tweets, left_on='id', right_on='user_id', suffixes=('', '_'))
            dfs_dev['tweet_metadata_df'] = pd.merge(dfs_dev['user_df']['id'], df_tweets, left_on='id', right_on='user_id', suffixes=('', '_'))
            dfs_test['tweet_metadata_df'] = pd.merge(dfs_test['user_df']['id'], df_tweets, left_on='id', right_on='user_id', suffixes=('', '_'))
            del(df_tweets)

            dfs_train['tweet_df'] = dfs_train['tweet_metadata_df'].pop('text')
            dfs_dev['tweet_df'] = dfs_dev['tweet_metadata_df'].pop('text')
            dfs_test['tweet_df'] = dfs_test['tweet_metadata_df'].pop('text')
        else:
            dfs_train['tweet_df'] = dfs_train['tweet_metadata_df'] = None
            dfs_dev['tweet_df'] = dfs_dev['tweet_metadata_df'] = None
            dfs_test['tweet_df'] = dfs_test['tweet_metadata_df'] = None

        return {
            'train': dfs_train,
            'dev': dfs_dev,
            'test': dfs_test
        }

    def read_twibot(
        self,
        config,
        label_column,
        use_users: bool,
        use_tweet: bool,
        use_tweet_metadata: bool,
        use_network: bool
    ) -> List[Optional[pd.DataFrame]]:
        """
        Return a list of 4 optional Dataframe represents user features, tweet features and
        tweet metadata features. Each element can be a Dataframe if use_[categories] is set
        to True and None otherwise.
        """
        dfs_train = dict()
        dfs_dev = dict()
        dfs_test = dict()

        df_label_train = pd.read_csv(config['train'] + config['label'])
        df_label_test = pd.read_csv(config['test'] + config['label'])
        df_label_dev = pd.read_csv(config['dev'] + config['label'])

        # User dataframe
        if use_users:
            dtypes = {
                "default_profile": 'int64',
                "geo_enabled": 'int64',
                "profile_use_background_image": 'int64',
                "verified": 'int64',
                "protected": 'int64'
            }
            df_users_train = pd.read_csv(config['train'] + config['profile_info'])
            df_users_train = df_users_train.replace({"True ": 1, "False ": 0}).astype(dtypes)
            df_users_train = df_users_train.merge(df_label_train, on='id')

            df_users_test = pd.read_csv(config['test'] + config['profile_info'])
            df_users_test = df_users_test.replace({"True ": 1, "False ": 0}).astype(dtypes)
            df_users_test = df_users_test.merge(df_label_test, on='id')

            df_users_dev = pd.read_csv(config['dev'] + config['profile_info'])
            df_users_dev = df_users_dev.replace({"True ": 1, "False ": 0}).astype(dtypes)
            df_users_dev = df_users_dev.merge(df_label_dev, on='id')

            dfs_train['user_df'] = df_users_train
            dfs_dev['user_df'] = df_users_dev
            dfs_test['user_df'] = df_users_test
        else:
            dfs_train['user_df'] = dfs_dev['user_df'] = dfs_test['user_df'] = None

        if use_tweet:
            df_tweets_train = pd.read_csv(config['train'] + config['tweet'])
            df_tweets_train = df_tweets_train.merge(df_label_train, on='id')

            df_tweets_test = pd.read_csv(config['test'] + config['tweet'])
            df_tweets_test = df_tweets_test.merge(df_label_test, on='id')

            df_tweets_dev = pd.read_csv(config['dev'] + config['tweet'])
            df_tweets_dev = df_tweets_dev.merge(df_label_dev, on='id')

            dfs_train['tweet_df'] = df_tweets_train
            dfs_dev['tweet_df'] = df_tweets_dev
            dfs_test['tweet_df'] = df_tweets_test

            dfs_train['tweet_metadata_df'] = dfs_test['tweet_metadata_df'] = dfs_dev['tweet_metadata_df'] = None
        else:
            dfs_train['tweet_df'] = dfs_test['tweet_df'] = dfs_dev['tweet_df'] = None
            dfs_train['tweet_metadata_df'] = dfs_test['tweet_metadata_df'] = dfs_dev['tweet_metadata_df'] = None
        
        if use_tweet_metadata:
            print(f"WARNING: Tweet metadata is not available in TwiBot-20 dataset")

        if use_network:
            print(f"WARNING: Current not support network features, but will do soon")

        return {
            'train': dfs_train,
            'dev': dfs_dev,
            'test': dfs_test
        }