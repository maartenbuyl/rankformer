import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ltr_dataset import LearningToRankDataset
from metrics import NDCG, TopNDCG, Average
from model import RankFormer, MLP

SEED = 0
ROOT_DIR = os.path.join(os.path.dirname(__file__))
DEBUG = False  # Set to True to run on a small subset of the data
DEVICE = 'cpu'
NUM_WORKERS = 0


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading data...")
    data_dir = os.path.join(ROOT_DIR, 'MSLR-WEB30K', f"Fold{SEED % 5 + 1}")
    transform = QuantileTransformer(output_distribution='normal')
    train_data = load_web30k_data(data_dir, transform, 'train')
    test_data = load_web30k_data(data_dir, transform, 'test')
    train_loader = DataLoader(train_data, batch_size=2048, shuffle=True, collate_fn=LearningToRankDataset.collate_fn,
                              num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_data, batch_size=2048, shuffle=False, collate_fn=LearningToRankDataset.collate_fn,
                             num_workers=NUM_WORKERS)

    print("Setting up model...")
    # model = MLP(input_dim=train_data.input_dim, hidden_layers=[512, 256, 128], dropout=.25)
    model = RankFormer(input_dim=train_data.input_dim, max_target=train_data.max_target,
                       tf_dim_feedforward=512, tf_nhead=1, tf_num_layers=3, head_hidden_layers=[128],
                       list_pred_strength=1.).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1)
    learning_decay_start = 20
    learning_decay = -.5
    decay_fn = lambda t: (t - learning_decay_start + 1) ** learning_decay if t >= learning_decay_start else 1.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, decay_fn)

    print("Training...")
    model.train()
    train_metrics = {
        'loss': Average(),
        'ndcg': NDCG(k=10),
        'ndcg_top': TopNDCG(max_target=train_data.max_target, k=10),
        'ndcg_explicit': NDCG(k=10)
    }
    loop = tqdm(range(200))
    for _epoch in loop:
        for batch in train_loader:
            optimizer.zero_grad()
            feat, length, target = batch['feat'].to(DEVICE), batch['length'].to(DEVICE), batch['target'].to(DEVICE)
            score = model(feat, length)
            loss = model.compute_loss(score, target, length)
            loss.backward()
            optimizer.step()

            if isinstance(score, tuple):
                # The RankFormer returns a tuple of (rank_score, list_score), but we only care about rank_score here
                score = score[0]
            update_metrics(batch, score.detach(), loss, train_metrics)
        scheduler.step()

        agg = compute_metrics(train_metrics)
        loop.set_postfix({f"train_{key}": val for key, val in agg.items()})

    print("Testing...")
    model.eval()
    test_metrics = {
        'loss': Average(),
        'ndcg': NDCG(k=10),
        'ndcg_top': TopNDCG(max_target=train_data.max_target, k=10),
        'ndcg_explicit': NDCG(k=10)
    }
    with torch.no_grad():
        for batch in test_loader:
            feat, length, target = batch['feat'].to(DEVICE), batch['length'].to(DEVICE), batch['target'].to(DEVICE)
            score = model(feat, length)
            loss = model.compute_loss(score, target, length)

            if isinstance(score, tuple):
                # The RankFormer returns a tuple of (rank_score, list_score), but we only care about rank_score here
                score = score[0]
            update_metrics(batch, score, loss, test_metrics)
    agg = compute_metrics(test_metrics)
    print({f"test_{key}": val for key, val in agg.items()})


def load_web30k_data(data_dir, transform, stage):
    path = os.path.join(data_dir, f"{stage}.txt")
    nrows = 1000 if DEBUG else None
    df = pd.read_csv(path, sep=' ', header=None, nrows=nrows).dropna(axis=1)
    df.loc[:, 1:] = df.loc[:, 1:].apply(lambda row: [el.split(':')[1] for el in row])
    df.columns = ['target'] + ['qid'] + [f'feat_{i}' for i in range(1, 137)]
    df = df.astype(float)
    df[['target', 'qid']] = df[['target', 'qid']].astype(int)

    user_model = {
        'seen_max': 16,
        'seen_bootstrap': 10,
        'click_noise': .1,
        'purchase_intent_kappa': .1,
        'purchase_noise': 0.
    }
    # user_model = None
    data = LearningToRankDataset(df, label_column='target', list_id_column='qid', transform=transform,
                                 user_model=user_model, seed=SEED)
    return data


def update_metrics(batch, score, loss, metrics):
    target = batch['target']
    length = batch['length']

    metrics['loss'].update(loss.item() * length.shape[0], weight=length.shape[0])
    metrics['ndcg'].update(score, target, length)
    metrics['ndcg_top'].update(score, target, length)
    if 'explicit_target' in batch:
        explicit_target = batch['explicit_target'].to(DEVICE)
        metrics['ndcg_explicit'].update(score, explicit_target, length)


def compute_metrics(metrics):
    return {key: metric.compute() for key, metric in metrics.items()}


if __name__ == '__main__':
    main()
