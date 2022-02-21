import random
import pandas as pd
import torch


class RatingDataset(torch.utils.data.Dataset):
    """
    
    Parameters
    ----------
    user_list : list
        List of users.
    item_list : list
        List of items.
    rating_list : list
        List of ratings.

    Returns
    -------
    None.

    """
    def __init__(self, user_list, item_list, rating_list):
        super().__init__()
        self.user_list = user_list
        self.item_list = item_list
        self.rating_list = rating_list

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, idx):
        user = self.user_list[idx]
        item = self.item_list[idx]
        rating = self.rating_list[idx]
        
        return (torch.tensor(user, dtype=torch.long),
                torch.tensor(item, dtype=torch.long),
                torch.tensor(rating, dtype=torch.float))


class NcfData():
    """
    
    Parameters
    ----------
    df : Pandas DataFrame
        Datasframe that contains: user_id, item_id, rating, timestamp.
    num_neg : int
        Number of negative instances to pair with a positive instance.
    num_neg_test : int
        Number of negative instances for test set.
    batch_size : int
        Batch size.
    seed : int
        Seed for reproducibility.

    Returns
    -------
    None.

    """
    def __init__(self, df, num_neg, num_neg_test, batch_size, seed):
        self.df = df
        self.num_neg = num_neg
        self.num_neg_test = num_neg_test
        self.batch_size = batch_size
        self.preprocess_df = self._preprocess(self.df)
        self.item_pool = set(self.df['item_id'].unique())
        self.train_set, \
        self.test_set = self._leave_one_out_split(self.preprocess_df)
        self.negatives = self._negative_sampling(self.preprocess_df)
        random.seed(seed)
        
    def _preprocess(self, df):
        """Reindex the user and item ids and binarize the rating"""
        user_list = list(df['user_id'].drop_duplicates())
        new_user_ids = {old: new for new, old in enumerate(user_list)}

        item_list = list(df['item_id'].drop_duplicates())
        new_item_ids = {old: new for new, old in enumerate(item_list)}

        df['user_id'] = df['user_id'].apply(lambda x: new_user_ids[x])
        df['item_id'] = df['item_id'].apply(lambda x: new_item_ids[x])
        df['rating'] = df['rating'].apply(lambda x: float(x > 0))
        return df

    def _leave_one_out_split(self, df):
        df['rank_latest'] = df.groupby(['user_id'])['timestamp'] \
                              .rank(method='first', ascending=False)
        test_set = df.loc[df['rank_latest'] == 1]
        train_set = df.loc[df['rank_latest'] > 1]
        
        assert train_set['user_id'].nunique()==test_set['user_id'].nunique(), \
               'Not Match Train User with Test User'
               
        return (train_set[['user_id', 'item_id', 'rating']],
                test_set[['user_id', 'item_id', 'rating']])
    
    def _negative_sampling(self, df):
        interact_status = (df.groupby('user_id')['item_id']
                           .apply(set)
                           .reset_index()
                           .rename(columns={'item_id': 'interacted_items'}))
        
        interact_status['negative_items'] = interact_status['interacted_items'] \
                                           .apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'] \
                          .apply(lambda x: random.sample(x, self.num_neg_test))
                          
        return interact_status[['user_id',
                                'negative_items',
                                'negative_samples']]

    def get_train_instance(self):
        """

        Returns
        -------
        torch.utils.data.DataLoader
            Dataloader for train set.

        """
        users, items, ratings = [], [], []
        train_set = pd.merge(self.train_set,
                             self.negatives[['user_id', 'negative_items']],
                             on='user_id')
        train_set['negatives'] = train_set['negative_items'] \
                               .apply(lambda x: random.sample(x, self.num_neg))
                               
        for row in train_set.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.item_id))
            ratings.append(float(row.rating))
            
            for i in range(self.num_neg):
                users.append(int(row.user_id))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))  # negative samples get 0 rating
                
        dataset = RatingDataset(user_list=users,
                                item_list=items,
                                rating_list=ratings)
        
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=4)

    def get_test_instance(self):
        """

        Returns
        -------
        torch.utils.data.DataLoader
            Dataloader for test set.

        """
        users, items, ratings = [], [], []
        test_set = pd.merge(self.test_set,
                            self.negatives[['user_id', 'negative_samples']],
                            on='user_id')
        
        for row in test_set.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.item_id))
            ratings.append(float(row.rating))
            for i in getattr(row, 'negative_samples'):
                users.append(int(row.user_id))
                items.append(int(i))
                ratings.append(float(0))
                
        dataset = RatingDataset(user_list=users,
                                item_list=items,
                                rating_list=ratings)
        
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.num_neg_test+1,
                                           shuffle=False, num_workers=4)
