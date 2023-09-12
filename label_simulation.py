import numpy as np
import pandas as pd


class UserModel:
    SEEN = 0
    CLICK = 1
    PURCHASE = 4   # Manually increase business importance of purchases. Equivalent to squaring all implicit labels.

    def __init__(self, label_column, list_id_column,
                 seen_max=16,
                 seen_bootstrap=10,
                 click_noise=.1,
                 purchase_intent_kappa=.1,
                 purchase_noise=0.):
        self.label_column = label_column
        self.list_id_column = list_id_column
        self.seen_max = seen_max
        self.seen_boostrap = seen_bootstrap
        self.click_noise = click_noise
        self.purchase_intent_kappa = purchase_intent_kappa
        self.purchase_noise = purchase_noise

        self.overall_max_label = None
        self.rng = None

    def apply(self, df, seed=None):
        self.rng = np.random.default_rng(seed)

        print("Simulating implicit user feedback...")
        df['explicit_target'] = df[self.label_column].astype(int)
        df['implicit_target'] = np.zeros_like(df['explicit_target'].values)

        # Sample some (subsampled) lists
        df = self.sample_selection(df)
        self.overall_max_label = df['explicit_target'].max()

        # Sample user intent per list
        df['intent'] = df.groupby(self.list_id_column)['explicit_target'].transform(self.sample_intent)

        # Sample clicks and purchases given the simulated intent
        df = self.sample_click(df)
        df = self.sample_purchase(df)

        df = df.drop(columns=['intent'])
        return df

    def sample_selection(self, df):
        max_list_id = df[self.list_id_column].max()
        sampled_dfs = []

        for i in range(self.seen_boostrap):
            # Sample a fraction of each list
            sampled_df = df.sample(frac=1, random_state=self.rng).groupby(self.list_id_column).head(n=self.seen_max)

            # Give each list a unique id
            sampled_df[self.list_id_column] += (max_list_id + 1) * i
            sampled_dfs.append(sampled_df)

        sampled_df = pd.concat(sampled_dfs).reset_index(drop=True)
        sampled_df.sort_values(by=self.list_id_column, inplace=True)
        return sampled_df

    def rho(self, label):
        # Arbitrary function to assign a value in [0, 1] to integer, ordinal explicit labels
        relevance_factor = (np.power(2, label) - 1) / (np.power(2, self.overall_max_label) - 1)
        return relevance_factor

    def intent_probabilities(self, max_rho):
        return {
            UserModel.SEEN: 1 - max_rho,
            UserModel.CLICK: max_rho * (1 - self.purchase_intent_kappa),
            UserModel.PURCHASE: max_rho * self.purchase_intent_kappa
        }

    def sample_intent(self, label):
        max_rho = self.rho(label.max())
        intent_probs = self.intent_probabilities(max_rho)
        intent = self.rng.choice(list(intent_probs.keys()), size=1, p=list(intent_probs.values()))
        return intent * np.ones_like(label)

    def sample_idx(self, df, noise=0.):
        label = df['explicit_target']
        relevance_factor = self.rho(label)

        sample_probs = noise + (1 - noise) * relevance_factor
        sampled_idx = self.rng.uniform(0., 1., len(df)) <= sample_probs
        return sampled_idx

    def sample_click(self, df):
        click_idx = self.sample_idx(df, self.click_noise)
        click_idx[df['intent'] < UserModel.CLICK] = False
        df.loc[click_idx, 'implicit_target'] = UserModel.CLICK
        return df

    def sample_purchase(self, df):
        purchase_idx = self.sample_idx(df, self.purchase_noise)
        purchase_idx[df['intent'] < UserModel.PURCHASE] = False
        df.loc[purchase_idx, 'implicit_target'] = UserModel.PURCHASE
        return df
