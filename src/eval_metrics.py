"""
Top-k evaluation metrics: Precision@k, Recall@k, NDCG@k and catalogue coverage
"""

import numpy as np
import pandas as pd

from config import EVAL_K, USER_RATING_THRESHOLD


def precision_at_k(recommended, relevant, k=EVAL_K):
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended, relevant, k=EVAL_K):
    if len(relevant) == 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


# Binary relevance: a hit at rank i (1-indexed) contributes 1 / log2(i + 1)
def ndcg_at_k(recommended, relevant, k=EVAL_K):
    dcg = sum(1.0 / np.log2(rank + 2)
              for rank, item in enumerate(recommended[:k]) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


# The fraction of the catalogue that ever gets recommended to anyone
def catalogue_coverage(recommendations, n_items, k=EVAL_K):
    recommended_items = set()
    for items in recommendations.values():
        recommended_items.update(items[:k])
    return len(recommended_items) / n_items


# recommendations: {user_id: ranked list of movie_ids}
# Users without any relevant test items are skipped, since recall and NDCG are undefined for them
def evaluate(recommendations, test_data: pd.DataFrame, n_items, k=EVAL_K, threshold=USER_RATING_THRESHOLD):
    positives = test_data[test_data.rating >= threshold]
    relevant = positives.groupby("user_id")["movie_id"].apply(set).to_dict()
    users = [user for user in recommendations if user in relevant]

    precisions = [precision_at_k(recommendations[u], relevant[u], k) for u in users]
    recalls = [recall_at_k(recommendations[u], relevant[u], k) for u in users]
    ndcgs = [ndcg_at_k(recommendations[u], relevant[u], k) for u in users]

    return {
        f"precision@{k}": float(np.mean(precisions)) if users else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if users else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if users else 0.0,
        "coverage": catalogue_coverage(recommendations, n_items, k),
        "n_users": len(users),
    }

