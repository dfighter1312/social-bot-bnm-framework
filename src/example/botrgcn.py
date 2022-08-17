import torch
import os
import numpy as np
import pandas as pd

from transformers import pipeline

from torch_geometric.nn import RGCNConv

from src.pipeline import BaseDetectorPipeline


class BotRGCNPipeline(BaseDetectorPipeline):
    
    def __init__(self, **kwargs):
        super().__init__(
            user_features=[
                'description',
                'statuses_count',
                'followers_count',
                'friends_count',
                'favourites_count',
                'screen_name',
                'protected',
                'geo_enabled',
                'verified',
                'contributors_enabled',
                'is_translator',
                'is_translation_enabled',
                'profile_background_tile',
                'profile_user_background_image',
                'has_extended_profile',
                'default_profile',
                'default_profile_image',
                'created_at'
            ],
            use_tweet=True,
            use_network=True,
            fe_return_pandas=False
        )
        self.suffix = 'test'

    def feature_engineering_u(self, user_df: pd.DataFrame, training):
        # Description preprocessing
        descr_path = f'preprocess_ckpts/BotRGCN/description_{"train" if training else "test"}_{self.suffix}.pt'
        if not os.path.exists(descr_path) or not training:
            description = user_df['description'].fillna('').values.tolist()
            feature_extraction = pipeline(
                'feature-extraction',
                model='distilroberta-base',
                tokenizer='distilroberta-base',
                device=0
            )
            description_embedded = feature_extraction(description)
            description_embedded = [torch.Tensor(d[0]) for d in description_embedded]
            description_embedded = torch.nn.utils.rnn.pad_sequence(description_embedded, batch_first=True)
            description_embedded = torch.mean(description_embedded, dim=1)
            torch.save(description_embedded, descr_path)
        else:
            description_embedded = torch.load(descr_path)

        # Numerical data preprocessing
        user_path = f'preprocess_ckpts/BotRGCN/user_{"train" if training else "test"}_{self.suffix}.pt'
        if not os.path.exists(user_path) or not training:
            user_df['age'] = (
                    pd.to_datetime(pd.to_datetime('today')) - 
                    pd.to_datetime(user_df.loc[:, 'created_at']).dt.tz_localize(None)
            ) / np.timedelta64(1, 'D')
            
            # Different to original author, we will perform z-score norm on 
            # both numerical and categorical features
            user_processed_df = user_df.drop(['id', 'created_at', 'label'], axis=1)
            user_processed_df = (user_processed_df - user_processed_df.mean(axis=0)) / user_processed_df.std(axis=0)
            user_processed_df.fillna(0, inplace=True)
            user_processed = user_processed_df.values
            user_processed = torch.Tensor(user_processed)
            torch.save(user_processed, user_path)
        else:
            user_processed = torch.load(user_path)
        
        self.user_dim = list(user_processed.size())[1]
        user_label = torch.Tensor(user_df['label'].values)
        
        return description_embedded, user_processed, user_label

    def semantic_encoding(self, tweet_df: pd.DataFrame, training):
        tweet_path = f'preprocess_ckpts/BotRGCN/tweet_{"train" if training else "test"}_{self.suffix}.pt'
        self.user_order = tweet_df['user_id'].unique()

        if not os.path.exists(tweet_path) or not training:
            feature_extract = pipeline(
                'feature-extraction',
                model='roberta-base',
                tokenizer='roberta-base',
                padding=True,
                truncation=True,
                max_length=500,
                add_special_tokens=True,
                device=0
            )
            # The order of user_id in tweet_df is identical to user_df
            tweets_embedded = []
            user_order = tweet_df['user_id'].unique()
            # tweet_df = TweetDataset(tweet_df, user_order)
            for i, user in enumerate(user_order):
                print(f"Tweet processing for user {i+1}/{len(user_order)}")
                user_tweets = tweet_df[tweet_df['user_id'] == user]['text'].fillna('').values.tolist()
                user_tweets_embedded = feature_extract(user_tweets)
                user_tweets_embedded = [torch.Tensor(t[0]) for t in user_tweets_embedded]
                user_tweets_embedded = torch.nn.utils.rnn.pad_sequence(user_tweets_embedded)
                user_tweets_embedded = torch.mean(user_tweets_embedded, dim=1)
                user_tweets_embedded = torch.mean(user_tweets_embedded, dim=0)
                tweets_embedded.append(user_tweets_embedded)
            tweets_embedded = torch.stack(tweets_embedded)
            torch.save(tweets_embedded, tweet_path)
        else:
            tweets_embedded = torch.load(tweet_path)
            user_order = tweet_df['user_id'].unique()
        return tweets_embedded, self.user_order

    def process_graph(self, following_df, follower_df, user_df, training):

        edge_index_path = f'preprocess_ckpts/BotRGCN/edge_index_{"train" if training else "test"}_{self.suffix}.pt'
        edge_type_path = f'preprocess_ckpts/BotRGCN/edge_type_{"train" if training else "test"}_{self.suffix}.pt'
        self.id2index_dict = {id: index for index, id in enumerate(user_df['id'])}

        if (not os.path.exists(edge_index_path) and not os.path.exists(edge_type_path)) or not training:
            # edge_type = np.concatenate([np.zeros(len(following_df)), np.ones(len(follower_df))])

            edge_index = []
            edge_type = []

            for source, target in following_df.values:
                source_id = self.id2index_dict[source]
                target_id = self.id2index_dict.get(target, None)
                if target_id is not None:
                    edge_index.append([source_id, target_id])
                    edge_type.append(0)
            for source, target in follower_df.values:
                source_id = self.id2index_dict[source]
                target_id = self.id2index_dict.get(target, None)
                if target_id is not None:
                    edge_index.append([source_id, target_id])
                    edge_type.append(1)
            edge_index=torch.tensor(edge_index,dtype=torch.long).t().contiguous()
            edge_type=torch.tensor(edge_type,dtype=torch.long)
            torch.save(edge_index, edge_index_path)
            torch.save(edge_type, edge_type_path)
        else:
            edge_index = torch.load(edge_index_path)
            edge_type = torch.load(edge_type_path)
        
        return edge_index, edge_type

    def reorder_tweets(self, tweets_embedded, n_users, n_dim, user_order):
        alt_tweets_embedded = torch.zeros((n_users, n_dim))
        for i, user_id in enumerate(user_order):
            idx = self.id2index_dict[user_id]
            alt_tweets_embedded[idx] = tweets_embedded[i]
        return alt_tweets_embedded

    def concatenate(self, user_df, tweet_df, tweet_metadata_df = None, following_df = None, follower_df = None) -> pd.DataFrame:
        """
        Args:
            user_df: (description_embedded, user_processed)
            tweet_df: (tweets_embedded, user_order)
            following_df: edge_index
            follower_df: edge_type
        """
        description_embedded, user_processed, user_labels = user_df
        tweets_embedded, user_order = tweet_df
        edge_index = following_df
        edge_type = follower_df

        # Processing user per order
        n_users = list(user_processed.size())[0]
        n_dim = list(tweets_embedded.size())[1]
        if list(tweets_embedded.size())[0] > 4000:
            tweets_embedded = self.reorder_tweets(tweets_embedded, n_users, n_dim, user_order)

        print("Shapes: ", description_embedded.size(), tweets_embedded.size(), user_processed.size(), edge_index.size(), edge_type.size())
        return description_embedded, tweets_embedded, user_processed, edge_index, edge_type, user_labels

    def classify(self, X_train, X_dev):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = 'cpu'
        learning_rate = 1e-3
        weight_decay = 5e-3

        # Initialize model
        self.model = BotRGCN(properties_size=self.user_dim).to(device)
        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.model.apply(init_weights)
        
        # Retrieve data
        description_embedded, tweets_embedded, user_processed, edge_index, edge_type, user_labels = X_train
        description_embedded = description_embedded.float().to(device)
        tweets_embedded = tweets_embedded.float().to(device)
        user_processed = user_processed.float().to(device)
        edge_index = edge_index.to(device)
        edge_type = edge_type.to(device)
        user_labels = user_labels.long().to(device)

        # Training
        epochs = 100
        for epoch in range(epochs):
            self.model.train()
            output = self.model(
                description_embedded,
                tweets_embedded,
                user_processed,
                edge_index,
                edge_type
            )
            loss_train = loss(output, user_labels)
            acc_train = accuracy(output, user_labels)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            print(
                'Epoch: {:04d}'.format(epoch+1),
                '- loss_train: {:.4f}'.format(loss_train.item()),
                '- acc_train: {:.4f}'.format(acc_train.item())
            )
        description_embedded = description_embedded.to('cpu')
        tweets_embedded = tweets_embedded.to('cpu')
        user_processed = user_processed.to('cpu')
        edge_index = edge_index.to('cpu')
        edge_type = edge_type.to('cpu')
        user_labels = user_labels.to('cpu')
        return self

    def predict(self, X_test):
        description_embedded, tweets_embedded, user_processed, edge_index, edge_type, user_labels = X_test
        result = self.model(
            description_embedded,
            tweets_embedded,
            user_processed,
            edge_index,
            edge_type
        )
        return result.detach().numpy()[:, 1] > 0.5, user_labels.numpy()

class TweetDataset(torch.utils.data.Dataset):
    
    def __init__(self, tweet_df, user_order):
        self.tweet_user_lst = []
        for user in user_order:
            user_tweets = tweet_df[tweet_df['user_id'] == user]['text'].fillna('').values.tolist()
            self.tweet_user_lst.append(user_tweets)
        
    def __len__(self):
        return len(self.tweet_user_lst)

    def __getitem__(self, idx):
        return self.tweet_user_lst[idx]
            

class BotRGCN(torch.nn.Module):
    
    def __init__(
        self,
        description_size = 768,
        tweet_size = 768,
        properties_size = 14,
        embedding_dimension = 128,
        dropout = 0.3
    ):
        super(BotRGCN, self).__init__()
        # self.dropout = dropout
        self.linear_relu_des = torch.nn.Sequential(
            torch.nn.Linear(description_size, int(embedding_dimension / 4)),
            torch.nn.LeakyReLU()
        )
        self.linear_relu_tweet = torch.nn.Sequential(
            torch.nn.Linear(tweet_size, int(embedding_dimension / 4)),
            torch.nn.LeakyReLU()
        )
        self.linear_relu_properties = torch.nn.Sequential(
            torch.nn.Linear(properties_size, int(embedding_dimension / 4)),
            torch.nn.LeakyReLU()
        )
        self.linear_relu_input = torch.nn.Sequential(
            torch.nn.Linear(int(embedding_dimension / 4) * 3, embedding_dimension),
            torch.nn.LeakyReLU()
        )
        self.rgcn = RGCNConv(
            embedding_dimension,
            embedding_dimension,
            num_relations=2
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.linear_output_1 = torch.nn.Sequential(
            torch.nn.Linear(embedding_dimension, embedding_dimension),
            torch.nn.LeakyReLU()
        )
        self.linear_output_2 = torch.nn.Sequential(
            torch.nn.Linear(embedding_dimension, 2),
            torch.nn.Softmax()
        )


    def forward(self, des, tweet, prop, edge_index, edge_type):
        d = self.linear_relu_des(des)
        t = self.linear_relu_tweet(tweet)
        p = self.linear_relu_properties(prop)
        x = torch.cat((d, t, p), dim=1)

        x = self.linear_relu_input(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.dropout(x)
        x = self.rgcn(x, edge_index, edge_type)
        x = self.linear_output_1(x)
        x = self.linear_output_2(x)

        return x

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)