"""
Main file for running the pipeline
"""

#imports
from config import FIXED_SEED
from src.char_distributions import draw_interaction_plots
from src.char_distributions import draw_rating_distribution
from src.split_data import temporal_split
from src.run_baselines import run_baselines

import pandas as pd
import numpy as np

# loading the data

mnames = ['movie_id', "title", "genres"]
movies_df = pd.read_table("data/movies.dat", sep="::", encoding='latin-1', header=None,names=mnames, engine='python')

rnames = ["user_id", "movie_id", "rating", "timestamp"]
ratings_df = pd.read_table("data/ratings.dat", sep="::", header=None,names=rnames, engine='python')

unames = ['user_id', 'gender', 'age', 'occupation', "zip"]
users_df = pd.read_table('data/users.dat', sep="::", header=None,names=unames, engine='python')

# Basic info
print(f"Number of users: {len(users_df.user_id.unique())}")
print(f"Number of movies: {len(movies_df.movie_id.unique())}")
print(f"Number of ratings: {ratings_df.shape[0]}")

# Characterization plots
draw_interaction_plots(ratings_df)
draw_rating_distribution(ratings_df)

# Split the data into test/validation/test splits
train_data, test_data, validate_data = temporal_split(ratings_df)

print(f"Size of training data: {train_data.shape[0]}")
print(f"Size of validation data: {validate_data.shape[0]}")
print(f"Size of test data: {test_data.shape[0]}")

# Check that the sum of splits equals the original rating data size
print(f"Total size of splits: {train_data.shape[0] + validate_data.shape[0] + test_data.shape[0]}, size of ratings: {ratings_df.shape[0]}" )

# Baselines: tune ItemKNN on validation, then evaluate every baseline on the test set
baseline_results, knn_tuning = run_baselines(train_data, validate_data, test_data, users_df.user_id, movies_df.movie_id)
knn_tuning.to_csv("results/knn_tuning.csv", index=False)
baseline_results.to_csv("results/results.csv", index=False)
