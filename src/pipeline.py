import time
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from src.data_read import LocalFileReader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

np.random.seed(0)

class BaseDetectorPipeline:
    """
    Abstract pipeline for detection.
    """
    local_file_reader = LocalFileReader()
    user_df: pd.DataFrame
    tweet_df: pd.DataFrame
    tweet_metadata_df: pd.DataFrame
    network_df: pd.DataFrame

    def __init__(
        self,
        dataset_name: str = 'MIB',
        user_features: Optional[Union[List[str], str]] = None,
        tweet_metadata_features: Optional[Union[List[str], str]] = None,
        use_tweet: bool = False,
        use_network: bool = False,
        verbose: bool = False
    ):
        self.dataset_name = dataset_name

        self.user_features = user_features
        self.tweet_features = tweet_metadata_features

        self.use_user = (user_features is not None)
        self.use_tweet_metadata = (tweet_metadata_features is not None)
        self.use_tweet = use_tweet
        self.use_network = use_network

        self.verbose = verbose
        self.label_col = 'label'

    def get_data(self, dataset_name: str = None):
        """Receive the dataset"""
        if dataset_name == 'MIB':
            config = self.local_file_reader.get_mib_config()
            self.user_df, self.tweet_df, self.tweet_metadata_df = self.local_file_reader.read_mib(
                config,
                self.label_col,
                self.use_user,
                self.use_tweet,
                self.use_tweet_metadata
            )

        elif dataset_name == 'TwiBot-20':
            self.local_file_reader.get_twibot_config()
            # TODO: Update TwiBot-20 setup dataset
        else:
            raise ValueError(
                "dataset_name must be MIB or TwiBot-20. Contact the "
                "author if you need any updates. "
            )

    def dataframe_slice(self, df: pd.DataFrame, slice: List[str]):
        """Select columns from slice, ignore the one that does not exist"""
        col = pd.Index(slice).append(pd.Index([self.label_col]))
        error_columns = col.difference(df.columns)
        if len(error_columns) != 0:
            print(
                f'WARNING: {error_columns.values} does not appear in the dataset, '
                'it/they will be ignored.'
            )
        implied_slice = col.intersection(df.columns)
        return df[implied_slice]

    def feature_selection(self):
        """Selecting original features from the dataset"""
        if self.user_features == 'all':
            return
        elif isinstance(self.user_features, list):
            self.user_df = self.dataframe_slice(self.user_df, self.user_features)
        elif self.user_features is not None:
            raise ValueError('Inappropriate value for user_features')

        if self.tweet_features == 'all':
            return
        elif isinstance(self.tweet_features, list):
            self.tweet_metadata_df = self.dataframe_slice(
                self.tweet_metadata_df,
                self.tweet_features
            )
        elif self.tweet_features is not None:
            raise ValueError('Inappropriate value for tweet_features')

    def semantic_encoding(self, tweet_df):
        """
        Implementation of the semantic encoder function,
        can be implemented here or in the `classify` method.
        """
        pass

    def feature_engineering_u(self, user_df):
        """Extract feature from selected ones, optional to be implemented"""
        pass

    def feature_engineering_ts(self, metadata_df):
        """Extract feature from selected ones, optional to be implemented"""
        pass

    def type_check(self, df: pd.DataFrame):
        all_cols = df.columns
        df = df.select_dtypes('number')
        remaining_cols = df.columns
        removed_cols = all_cols.difference(remaining_cols)
        if len(removed_cols) != 0:
            print(f"WARNING: {removed_cols.values} will be removed since they are not numeric columns")
        return df

    def concatenate(
        self,
        user_df: Optional[pd.DataFrame],
        tweet_df: Optional[pd.DataFrame],
        tweet_metadata_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Concatenate all the dataframe"""
        # If 1 dataframe is retrieved, concatenate function
        # can be skipped, otherwise it need to be implemented
        if (self.use_user + self.use_tweet + self.use_network) != 1:
            raise NotImplementedError('concatenate function must be '
                'implemented since there is more than one dataframe exists.')
        elif self.use_user:
            return user_df
        elif self.use_tweet:
            return tweet_df
        elif self.use_tweet_metadata:
            return tweet_metadata_df

    def train_test_split(self, df):
        """Split into train, test and dev set with proportion (0.8, 0.1, 0.1)"""
        train_idx, test_idx = train_test_split(range(len(df)), test_size=0.2)
        dev_idx, test_idx = train_test_split(test_idx, test_size=0.5)
        labels = df.pop(self.label_col)
        return (df.iloc[train_idx], df.iloc[dev_idx], df.iloc[test_idx], 
                labels.iloc[train_idx], labels.iloc[dev_idx], labels.iloc[test_idx])

    def evaluate(self, y_pred, y_test, time_taken):
        """Show the result on test set"""
        accuracy = round(accuracy_score(y_test, y_pred), 4)
        precision = round(precision_score(y_test, y_pred), 4)
        recall = round(recall_score(y_test, y_pred), 4)
        time_taken = list(map(lambda x: round(x, 4), time_taken))
        output_text = (
            f"==== Evaluation result ====\n"
            f"Dataset name: {self.dataset_name}\nAccuracy: {accuracy}\n"
            f"Precision: {precision}\nRecall: {recall}\nFeature Engineering time: {time_taken[0]}s\n"
            f"Training time {time_taken[1]}s\nInference time: {time_taken[2]}s"
        )
        print(output_text)

    def classify(self, X_train, X_dev, y_train, y_dev):
        """Implementation of the classification algorithm, dev sets are optional to be used"""
        raise NotImplementedError

    def predict(self, X_test):
        """Predict the result"""
        raise NotImplementedError

    def run(self):
        """Process the pipeline"""
        # Step 1: Get the input dataset
        if self.verbose:
            print('Getting data...')
        self.get_data(self.dataset_name)

        # Step 2: Feature Selection
        if self.verbose:
            print('Selecting features...')
        self.feature_selection()

        # Step 3A: Semantic Encoder (optional)
        if self.verbose:
            print('Featuring the data...')
        step_3_start = time.time()
        if self.use_tweet:
            self.tweet_df = self.semantic_encoding(self.tweet_df)

        # Step 3B: Feature engineering (optional)
        if self.use_user:
            self.user_df = self.feature_engineering_u(self.user_df)
            # Do an type check to ensure only numeric values remain
            self.user_df = self.type_check(self.user_df)
        if self.use_tweet_metadata:
            self.tweet_metadata_df = self.feature_engineering_ts(self.tweet_metadata_df)
            # Do an type check to ensure only numeric values remain
            self.tweet_metadata_df = self.type_check(self.tweet_metadata_df)
        step_3_end = time.time()

        # Step 3C: Concatenate the features
        df = self.concatenate(self.user_df, self.tweet_df, self.tweet_metadata_df)
        # Delete other dataframe to save memory
        del(self.user_df)
        del(self.tweet_df)
        del(self.tweet_metadata_df)

        # Step 3D: train_test_split
        X_train, X_dev, X_test, y_train, y_dev, y_test = self.train_test_split(df)
        del(df)

        # Step 4: Classification
        if self.verbose:
            print('Classifying...')
        step_4_start = time.time()
        self.classify(X_train, X_dev, y_train, y_dev)
        step_4_end = time.time()

        # Step 5A: Predict on test set
        if self.verbose:
            print('Predicting the result...')
        step_5_start = time.time()
        y_pred = self.predict(X_test)
        step_5_end = time.time()

        # Step 5B: Evaluate the result
        self.evaluate(y_pred, y_test, [
            step_3_end - step_3_start,
            step_4_end - step_4_start,
            step_5_end - step_5_start
        ])
