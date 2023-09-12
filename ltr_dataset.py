import torch
from torch.utils.data import Dataset
from sklearn.exceptions import NotFittedError

from label_simulation import UserModel


class LearningToRankDataset(Dataset):
    def __init__(self, df, label_column, list_id_column, transform=None, user_model=None, seed=None):

        # It is costly to sort before any filtering happens, but we need the groups to be together for later efficiency.
        # All later steps are expected to maintain query group order.
        df.sort_values(by=list_id_column, inplace=True)

        if user_model is not None:
            df = UserModel(label_column, list_id_column, **user_model).apply(df, seed=seed)
            df['target'] = df['implicit_target']

        feat_columns = df.columns.difference([label_column, list_id_column, 'explicit_target', 'implicit_target'])
        self.feat = df[feat_columns].values
        if transform is not None:
            try:
                self.feat = transform.transform(self.feat)
            except NotFittedError:
                self.feat = transform.fit_transform(self.feat)

        self.feat = torch.from_numpy(self.feat).float()
        self.target = torch.from_numpy(df[label_column].values).float()
        self.length = torch.from_numpy(df[list_id_column].value_counts(sort=False).values)
        self.cum_length = torch.cumsum(self.length, dim=0)
        if 'explicit_target' in df.columns:
            self.explicit_target = torch.from_numpy(df['explicit_target'].values).int()
        else:
            self.explicit_target = None

    def __getitem__(self, item):
        # All item features, targets and list ids are stored in a single flat array. Each list is stored back-to-back.
        # When getting a batch element (i.e. a list), we therefore need to slice the correct range in the flat array.
        # The start and end indices of each list can be inferred from the cum_length array.

        if item == 0:
            start_idx = 0
        else:
            start_idx = self.cum_length[item-1]
        end_idx = self.cum_length[item].item()

        item_dict = {
            'feat': self.feat[start_idx:end_idx],
            'target': self.target[start_idx:end_idx],
            'length': self.length[item].reshape(1)
        }
        if self.explicit_target is not None:
            item_dict['explicit_target'] = self.explicit_target[start_idx:end_idx]
        return item_dict

    def __len__(self):
        return self.length.shape[0]

    @staticmethod
    def collate_fn(batches):
        batch_example = batches[0]
        batch = {key: torch.cat([batch_vals[key] for batch_vals in batches]) for key in batch_example.keys()}
        return batch

    @property
    def input_dim(self):
        return self.feat.shape[1]

    @property
    def max_target(self):
        # Used in the ordinal loss function of the RankFormer
        return self.target.max().cpu().int().item()
