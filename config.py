# Configurations and global variables

FIXED_SEED = 256

# Length of each recommendation list used for evaluation
EVAL_K = 10

# At or above this threshold user is considered to like the movie (and recommendation is considered successful)
USER_RATING_THRESHOLD = 4

# ItemKNN tuning grid, searched on the validation set and selected by NDCG@EVAL_K
KNN_SIMILARITIES = ["cosine", "adjusted_cosine", "jaccard"]
KNN_K_VALUES = [5, 10, 25, 50]

# Fixed (not tuned) ItemKNN settings:
# significance weighting gamma ("around 25-50") and the similarity threshold for neighbours
KNN_GAMMA = 50
KNN_THRESHOLD = 0.3