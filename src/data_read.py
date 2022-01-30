from typing import Dict, List, Optional
import pandas as pd
from configparser import RawConfigParser

from sklearn.model_selection import train_test_split


class LocalFileReader:
    """Read .csv and .cfq files"""

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

        # User dataframe
        df_bot_users = pd.concat(
            [
                pd.read_csv(config['social_spam_1_users']),
                pd.read_csv(config['social_spam_2_users']),
                pd.read_csv(config['social_spam_3_users']),
            ]
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
            df_bot_tweets = pd.concat(
                [
                    pd.read_csv(config['social_spam_1_tweets'], nrows=200000),
                    pd.read_csv(config['social_spam_2_tweets'], nrows=200000),
                    pd.read_csv(config['social_spam_3_tweets'], nrows=200000)
                ]
            ).reset_index(drop=True)
            df_naive_tweets = pd.read_csv(config['genuine_tweets'], header=None, escapechar='\\', nrows=600000)
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

