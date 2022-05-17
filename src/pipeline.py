import time
import numpy as np
import tensorflow as tf
import pandas as pd
from typing import List, Optional, Union
from src.data_read import LocalFileReader
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score

np.random.seed(0)
tf.random.set_seed(0)

class BaseDetectorPipeline:
    """
    Abstract pipeline for detection.
    """
    local_file_reader = LocalFileReader()

    def __init__(
        self,
        user_features: Optional[Union[List[str], str]] = None,
        tweet_metadata_features: Optional[Union[List[str], str]] = None,
        use_tweet: bool = False,
        use_network: bool = False,
        verbose: bool = True,
        account_level: bool = True,
    ):
        self.id_col = 'id'
        self.user_id_col = 'user_id'
        self.label_col = 'label'

        self.user_features = self.reconfig_feature_list(user_features, self.id_col)
        self.tweet_features = self.reconfig_feature_list(tweet_metadata_features, self.user_id_col)

        self.use_user = (len(self.user_features) != 1)
        self.use_tweet_metadata = (len(self.tweet_features) != 1)
        self.use_tweet = use_tweet
        self.use_network = use_network

        self.verbose = verbose
        self.account_level = account_level

    def reconfig_feature_list(self, feature_list, id_col):
        if feature_list == 'all':
            return feature_list
        elif feature_list is None:
            return [id_col]
        elif isinstance(feature_list, list):
            return feature_list + [id_col]
        else:
            raise TypeError('user_features or tweet_metadata_features is not valid')

    def get_data(self, dataset_name: str = None, nrows: Optional[int] = None):
        """Receive the dataset"""
        if dataset_name == 'MIB':
            config = self.local_file_reader.get_mib_config()
            self.dfs = self.local_file_reader.read_mib(
                config,
                self.label_col,
                self.use_user,
                self.use_tweet,
                self.use_tweet_metadata,
                nrows
            )
            # Turn off network since there is no usage
            self.use_network = False
        elif dataset_name == 'MIB-2':
            config = self.local_file_reader.get_mib_2_config()
            self.dfs = self.local_file_reader.read_mib_2(
                config,
                self.label_col,
                self.use_user,
                self.use_tweet,
                self.use_tweet_metadata,
                nrows
            )
            # Turn off network since there is no usage
            self.use_network = False
        elif dataset_name == 'TwiBot-20':
            config = self.local_file_reader.get_twibot_config()
            self.dfs = self.local_file_reader.read_twibot(
                config,
                self.label_col,
                self.use_user,
                self.use_tweet,
                self.use_tweet_metadata,
                self.use_network,
                nrows
            )
            # Turn off tweet metadata since there is no usage
            self.use_tweet_metadata = False
        else:
            raise ValueError(
                "dataset_name must be MIB or TwiBot-20. Contact the "
                "author if you need any updates. "
            )

    def dataframe_slice(self, df: pd.DataFrame, slice: List[str], warn: bool):
        """Select columns from slice, ignore the one that does not exist"""
        col = pd.Index(slice).append(pd.Index([self.label_col]))
        error_columns = col.difference(df.columns)
        if len(error_columns) != 0 and warn:
            print(
                f'WARNING: {error_columns.values} does not appear in the dataset, '
                'it/they will be ignored. If you tend to use the column in feature '
                'engineering, use an if `column_name` in user_df.columns statement.'
            )
        implied_slice = col.intersection(df.columns)
        return df[implied_slice]

    def feature_selection(
        self,
        warn: bool,
        user_df: Optional[pd.DataFrame],
        tweet_df: Optional[pd.DataFrame],
        tweet_metadata_df: Optional[pd.DataFrame],
    ):
        """Selecting original features from the dataset"""
        if self.use_user:
            if self.user_features == 'all':
                pass
            elif isinstance(self.user_features, list):
                user_df = self.dataframe_slice(user_df, self.user_features, warn)
            elif self.user_features is not None:
                raise ValueError('Inappropriate value for user_features')
        if self.use_tweet_metadata:
            if self.tweet_features == 'all':
                pass
            elif isinstance(self.tweet_features, list):
                tweet_metadata_df = self.dataframe_slice(
                    tweet_metadata_df,
                    self.tweet_features,
                    warn
                )
            elif self.tweet_features is not None:
                raise ValueError('Inappropriate value for tweet_features')
        return {
            'user_df': user_df,
            'tweet_df': tweet_df,
            'tweet_metadata_df': tweet_metadata_df
        }

    def semantic_encoding(self, tweet_df, training):
        """
        Implementation of the semantic encoder function,
        can be implemented here or in the `classify` method.
        """
        return tweet_df

    def feature_engineering_u(self, user_df, training):
        """Extract feature from selected ones, optional to be implemented"""
        return user_df.fillna(0.0)

    def feature_engineering_ts(self, metadata_df, training):
        """Extract feature from selected ones, optional to be implemented"""
        return metadata_df.fillna(0.0)

    def type_check(self, df: pd.DataFrame, warn: bool):
        all_cols = df.columns
        df = df.select_dtypes('number')
        remaining_cols = df.columns
        removed_cols = all_cols.difference(remaining_cols)
        if warn and len(removed_cols) != 0:
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
        total = self.use_user + self.use_tweet + self.use_tweet_metadata + self.use_network
        if total == 1:
            if self.use_user:
                return user_df
            if self.use_tweet:
                return tweet_df
            if self.use_tweet_metadata:
                return tweet_metadata_df
            if self.use_network:
                raise NotImplementedError
        elif total == 2:
            if self.use_tweet and self.use_tweet_metadata:
                return pd.concat([tweet_df, tweet_metadata_df], axis=1).drop(
                    [self.id_col, self.id_col + '_'],
                    axis=1,
                    errors='ignore')
            elif self.use_tweet and self.use_user:
                return pd.merge(
                    user_df,
                    tweet_df,
                    left_on='id',
                    right_on='user_id' if 'user_id' in tweet_df.columns else 'id',
                    suffixes=('', '_')
                ).drop(self.label_col + '_', axis=1, errors='ignore').drop(self.user_id_col, axis=1, errors='ignore')
            else:
                raise NotImplementedError
        elif total == 3:
            if self.use_tweet and self.use_tweet_metadata and self.use_user:
                try:
                    tweet_df.pop(self.user_id_col)
                except:
                    pass
                merged_df = pd.concat([tweet_df, tweet_metadata_df], axis=1)
                return pd.merge(
                    user_df,
                    merged_df,
                    left_on='id',
                    right_on='user_id' if 'user_id' in merged_df.columns else 'id',
                    suffixes=('', '_')
                ).drop(self.label_col + '_', axis=1, errors='ignore').drop(self.user_id_col, axis=1, errors='ignore')
        else:
            raise NotImplementedError

    def grouping(self, id, y_pred, y_test):
        df_acc = pd.DataFrame()
        df_acc['id'] = id
        df_acc['pred'] = y_pred
        df_acc['true'] = y_test
        df_acc = df_acc.groupby('id').mean()
        return np.round(df_acc['pred'].values), df_acc['true'].values

    def evaluate(self, y_pred, y_test, time_taken, dataset_name):
        """Show the result on test set"""
        accuracy = round(accuracy_score(y_test, y_pred), 4)
        precision = round(precision_score(y_test, y_pred), 4)
        recall = round(recall_score(y_test, y_pred), 4)
        mcc = round(matthews_corrcoef(y_test, y_pred), 4)
        time_taken = list(map(lambda x: round(x, 4), time_taken))
        n_samples = len(y_pred)
        output_text = (
            f"\n==== Evaluation result ====\n"
            f"Dataset name: {dataset_name} ({n_samples} samples)\nAccuracy: {accuracy}\n"
            f"Precision: {precision}\nRecall: {recall}\nMCC: {mcc}\n"
            f"Training time {time_taken[0]}s\nInference time: {time_taken[1]}s\n"
        )
        print(output_text)

    def classify(self, X_train, X_dev, y_train, y_dev):
        """Implementation of the classification algorithm, dev sets are optional to be used"""
        raise NotImplementedError

    def predict(self, X_test):
        """Predict the result"""
        raise NotImplementedError

    def preprocess_train(self):
        # Step 2: Feature Selection
        if self.verbose:
            print('Selecting features...')
        self.dfs['train'] = self.feature_selection(warn=True, **self.dfs['train'])

        # Step 3A: Semantic Encoder (optional)
        if self.verbose:
            print('Featuring the data...')
        step_3_start = time.time()
        if self.use_tweet:
            # if isinstance(self.dfs['train']['tweet_df'], pd.DataFrame):
            #     self.dfs['train']['tweet_df']['text'] = self.semantic_encoding(
            #         self.dfs['train']['tweet_df']['text'],
            #         training=True
            #     )
            # else:
            self.dfs['train']['tweet_df'] = self.semantic_encoding(
                self.dfs['train']['tweet_df'],
                training=True
            )

        # Step 3B: Feature engineering (optional)
        if self.use_user:
            self.dfs['train']['user_df'] = self.feature_engineering_u(
                self.dfs['train']['user_df'],
                training=True
            )
            # Do a type check to ensure only numeric values remain
            self.dfs['train']['user_df'] = self.type_check(
                self.dfs['train']['user_df'],
                warn=True
            )
        if self.use_tweet_metadata:
            self.dfs['train']['tweet_metadata_df'] = self.feature_engineering_ts(
                self.dfs['train']['tweet_metadata_df'],
                training=True
            )
            # Do a type check to ensure only numeric values remain
            self.dfs['train']['tweet_metadata_df'] = self.type_check(
                self.dfs['train']['tweet_metadata_df'],
                warn=True
            )
        step_3_end = time.time()
        # Step 3C: Concatenate the features
        self.dfs['train'] = self.concatenate(**self.dfs['train'])
        return step_3_start, step_3_end

    def preprocess(self, set_name):
        if set_name == 'train':
            return self.preprocess_train()
        # Step 2: Feature Selection
        self.dfs[set_name] = self.feature_selection(
            warn=False,
            **self.dfs[set_name]
        )

        # Step 3A: Semantic Encoder (optional)
        if self.use_tweet:
            # if isinstance(self.dfs[set_name]['tweet_df'], pd.DataFrame):
            #     self.dfs[set_name]['tweet_df']['text'] = self.semantic_encoding(
            #         self.dfs[set_name]['tweet_df']['text'],
            #         training=True
            #     )
            # else:
            self.dfs[set_name]['tweet_df'] = self.semantic_encoding(
                self.dfs[set_name]['tweet_df'],
                training=False
            )

        # Step 3B: Feature engineering (optional)
        if self.use_user:
            self.dfs[set_name]['user_df'] = self.feature_engineering_u(
                self.dfs[set_name]['user_df'],
                training=False
            ).fillna(0.0)
            # Do a type check to ensure only numeric values remain
            self.dfs[set_name]['user_df'] = self.type_check(
                self.dfs[set_name]['user_df'],
                warn=False
            ).fillna(0.0)
        if self.use_tweet_metadata:
            self.dfs[set_name]['tweet_metadata_df'] = self.feature_engineering_ts(
                self.dfs[set_name]['tweet_metadata_df'],
                training=False
            ).fillna(0.0)
            # Do a type check to ensure only numeric values remain
            self.dfs[set_name]['tweet_metadata_df'] = self.type_check(
                self.dfs[set_name]['tweet_metadata_df'],
                warn=False
            )

        # Step 3C: Concatenate the features
        self.dfs[set_name] = self.concatenate(**self.dfs[set_name])


    def run(
        self,
        dataset_name: str = 'MIB',
        nrows: Optional[int] = None
    ):
        """Process the pipeline"""
        # Step 1: Get the input dataset
        if self.verbose:
            print(f'Getting {dataset_name} dataset...')
        self.get_data(dataset_name, nrows)

        # Step 2+3: Preprocessing
        step_3_start, step_3_end = self.preprocess('train')
        self.preprocess('dev')

        # Step 4: Classification
        if self.verbose:
            print('Classifying...')
        step_4_start = time.time()
        y_train = self.dfs['train'].pop(self.label_col)
        y_train = y_train if isinstance(y_train, pd.Series) else y_train.iloc[:, 0]
        y_dev = self.dfs['dev'].pop(self.label_col)
        y_dev = y_dev if isinstance(y_dev, pd.Series) else y_dev.iloc[:, 0]
        try:
            self.dfs['train'].pop(self.id_col)
            self.dfs['dev'].pop(self.id_col)
        except:
            self.dfs['train'].pop(self.user_id_col)
            self.dfs['dev'].pop(self.user_id_col)
        self.classify(self.dfs['train'], self.dfs['dev'], y_train, y_dev)
        step_4_end = time.time()

        # Step 5A: Predict on test set
        if self.verbose:
            print('Predicting the result...')
        step_5_start = time.time()
        self.preprocess('test')
        y_test = self.dfs['test'].pop(self.label_col)
        y_test = y_test if isinstance(y_test, pd.Series) else y_test.iloc[:, 0]
        try:
            id_test = self.dfs['test'].pop(self.id_col)
        except:
            id_test = self.dfs['test'].pop(self.user_id_col)
        id_test = id_test if isinstance(id_test, pd.Series) else id_test.iloc[:, 0]
        y_pred = self.predict(self.dfs['test'])

        # Step 5B: Grouping if the prediction is on every tweet, not on every account
        if self.account_level == False:
            y_pred, y_test = self.grouping(id_test, y_pred, y_test)
        step_5_end = time.time()

        # Step 5C: Evaluate the result
        self.evaluate(
            y_pred,
            y_test,
            [
                (step_3_end - step_3_start) + (step_4_end - step_4_start),
                step_5_end - step_5_start
            ],
            dataset_name
        )
