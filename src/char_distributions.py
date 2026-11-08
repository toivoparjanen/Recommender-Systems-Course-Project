"""
Draws characterization distributions for part1
"""

import pandas as pd
import matplotlib.pyplot as plt

def draw_interaction_plots(data: pd.DataFrame):

    user_interactions = data.groupby("user_id").size()
    item_interactions = data.groupby("movie_id").size()

    user_sorted = user_interactions.sort_values(ascending=False)
    item_sorted = item_interactions.sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(user_sorted.values)
    axes[0].set_title("Distribution of interactions per user")
    axes[0].set_xlabel("Users ranked by interactions")
    axes[0].set_ylabel("Number of interactions")

    axes[1].plot(item_sorted.values)
    axes[1].set_title("Distribution of interactions per movie")
    axes[1].set_xlabel("Movies ranked by interactions")
    axes[1].set_ylabel("Number of interactions")

    plt.tight_layout()
    plt.savefig("results/figures/interactions_plot.png")


def draw_rating_distribution(data: pd.DataFrame):

    ratings = data.rating

    plt.hist(ratings, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
             edgecolor="black")

    plt.xticks([1, 2, 3, 4, 5])
    plt.xlabel("Star rating")
    plt.ylabel("Number of movies")
    plt.title("Movie Ratings Distribution")

    plt.savefig("results/figures/ratings_plot.png")

