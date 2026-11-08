"""
Baseline recommenders for part1
"""

import numpy as np
import pandas as pd

from config import USER_RATING_THRESHOLD


class ItemKNN:
    def __init__(self, k=50, similarity="cosine", gamma=50, threshold=0.3):
        # k: how many of the user's rated movies, most similar to the target movie, are used
        # similarity: "cosine", "adjusted_cosine" or "jaccard"
        # gamma: significance weighting, similarities from fewer than gamma co-raters are shrunk
        # threshold: neighbours with a (shrunk) similarity below this are not used
        self.k = k
        self.Similarity = similarity
        self.gamma = gamma
        self.threshold = threshold

    # all_users and all_items define the rows and columns of the rating matrix,
    # so users and items without training ratings still get a (zero) row or column
    def fit(self, train: pd.DataFrame, all_users, all_items):
        self.user_ids = np.asarray(all_users)
        self.item_ids = np.asarray(all_items)
        self.user_index = {user: i for i, user in enumerate(self.user_ids)}
        item_index = {item: i for i, item in enumerate(self.item_ids)}

        # Rating matrix: one row per user, one column per item, 0 = not rated
        self.Ratings = np.zeros((len(self.user_ids), len(self.item_ids)), dtype=np.float32)
        rows = train.user_id.map(self.user_index).to_numpy()
        cols = train.movie_id.map(item_index).to_numpy()
        self.Ratings[rows, cols] = train.rating.to_numpy()

        # Average rating of each item over the users who rated it (0 if nobody did)
        rated = self.Ratings > 0
        self.item_means = self.Ratings.sum(axis=0) / np.maximum(rated.sum(axis=0), 1)

        self.Sim = self.compute_similarity()
        return self

    # Sim[i, j] = similarity between item i and item j, computed from the users who rated them
    def compute_similarity(self):
        # rated[u, i] = 1 if user u rated item i, else 0
        rated = (self.Ratings > 0).astype(np.float32)
        X = self.Ratings.copy()

        if self.Similarity == "adjusted_cosine":
            # Subtract each user's mean rating from that user's ratings
            user_means = X.sum(axis=1) / np.maximum(rated.sum(axis=1), 1)
            X = (X - user_means[:, None]) * rated

        if self.Similarity == "jaccard":
            # Jaccard: (users who rated both i and j) / (users who rated i or j).
            # Only uses which items were rated, not the rating values.
            intersection = rated.T @ rated                  # intersection[i, j]: users who rated both
            raters = rated.sum(axis=0)                      # raters[i]: users who rated i
            union = raters[:, None] + raters[None, :] - intersection
            Sim = intersection / np.maximum(union, 1)
        else:
            # Cosine between every pair of item columns, restricted to co-raters:
            # sim(i, j) = sum_u x_ui * x_uj / (sqrt(sum_u x_ui^2) * sqrt(sum_u x_uj^2)),
            # where every sum runs only over users u who rated both i and j.
            # Unrated entries are 0, so they drop out of the dot product automatically.
            dot_products = X.T @ X

            # squared_norms[i, j] = sum of x_ui^2 over the users who rated both i and j
            squared_norms = (X * X).T @ rated
            norms_product = np.sqrt(squared_norms * squared_norms.T)
            Sim = dot_products / (norms_product + 1e-9)

        # Significance weighting: multiply by min(n, gamma) / gamma, where n = users who rated both items
        co_raters = rated.T @ rated
        Sim = Sim * np.minimum(co_raters, self.gamma) / self.gamma

        # An item should not be its own similar item
        np.fill_diagonal(Sim, 0)
        return Sim

    # Returns {user_id: list of n recommended movie_ids, best first}
    def recommend(self, users, n=10):
        recommendations = {}
        for user in users:
            user_ratings = self.Ratings[self.user_index[user]]
            rated_items = np.where(user_ratings > 0)[0]

            # Step 2 (select): sims[a, i] = similarity between the user's a-th rated movie and movie i.
            # For every movie i, keep only the k rated movies most similar to it and zero the rest.
            sims = self.Sim[rated_items, :]
            if len(rated_items) > self.k:
                # argpartition puts the k largest values of each column first (unsorted), the rest after
                order = np.argpartition(-sims, self.k, axis=0)
                np.put_along_axis(sims, order[self.k:, :], 0, axis=0)

            # Threshold: neighbours that are not similar enough are not used either
            sims[sims < self.threshold] = 0

            # Step 3 (aggregate): pred(u, i) = avg(i) + sum_j sim(i, j) * (r(u, j) - avg(j)) / sum_j |sim(i, j)|
            # where j runs over the k neighbours of i that the user rated
            deviations = user_ratings[rated_items] - self.item_means[rated_items]
            numerator = deviations @ sims
            denominator = np.abs(sims).sum(axis=0)
            scores = self.item_means + numerator / np.maximum(denominator, 1e-9)

            # No similar enough movie among the user's ratings means no evidence, so never recommend it
            scores[denominator == 0] = -np.inf

            # Step 4 (rank): skip movies the user already rated and return the n best.
            # Movies without evidence are dropped, so the list can be shorter than n (or empty)
            scores[rated_items] = -np.inf
            best = np.argsort(-scores)[:n]
            best = best[np.isfinite(scores[best])]
            recommendations[user] = self.item_ids[best].tolist()
        return recommendations

class MostPopular:
    def __init__(self, only_positive=False):
        # only_positive: count only ratings >= USER_RATING_THRESHOLD instead of all ratings
        self.only_positive = only_positive

    def fit(self, train: pd.DataFrame):
        counted = train
        if self.only_positive:
            counted = train[train.rating >= USER_RATING_THRESHOLD]

        # Movies ordered from most to least ratings; ties keep the lower movie_id first
        counts = counted.groupby("movie_id").size()
        self.ranking = counts.sort_values(ascending=False, kind="stable").index.tolist()

        # Movies each user already rated in training, so they are not recommended again
        self.seen = train.groupby("user_id").movie_id.apply(set).to_dict()
        return self

    # Returns {user_id: list of n recommended movie_ids, best first}
    # Everyone gets the same ranking, only with their already rated movies skipped
    def recommend(self, users, n=10):
        recommendations = {}
        for user in users:
            seen = self.seen.get(user, set())
            recommendations[user] = [movie for movie in self.ranking if movie not in seen][:n]
        return recommendations
