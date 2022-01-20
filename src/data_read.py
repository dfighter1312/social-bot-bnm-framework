from typing import Dict, List, Optional
import pandas as pd
from configparser import RawConfigParser


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
        list_dfs = list()
        if use_users:
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
            list_dfs.append(df_users)
        else:
            list_dfs.append(None)

        if use_tweet or use_tweet_metadata:
            df_bot_tweets = pd.concat(
                [
                    pd.read_csv(config['social_spam_1_tweets']),
                    pd.read_csv(config['social_spam_2_tweets']),
                    pd.read_csv(config['social_spam_3_tweets']),
                ]
            ).reset_index(drop=True)
            df_naive_tweets = pd.read_csv(config['genuine_tweets'], header=None, escapechar='\\')
            df_bot_tweets[label_column] = 1
            df_naive_tweets[label_column] = 0
            df_tweets = pd.concat([df_bot_tweets, df_naive_tweets], ignore_index=True)
            df_tweets_text = df_tweets.pop('text')
            list_dfs += [
                df_tweets_text if use_tweet else None,
                df_tweets if use_tweet_metadata else None
            ]
        else:
            list_dfs += [None, None]

        return list_dfs

