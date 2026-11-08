"""
Main file for running the pipeline
"""

#imports
from config import FIXED_SEED
from src.char_distributions import draw_interaction_plots
from src.char_distributions import draw_rating_distribution

import pandas as pd


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





