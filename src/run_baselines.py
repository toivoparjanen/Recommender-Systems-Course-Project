"""
Tunes and evaluates the part1 baselines: ItemKNN and MostPopular
"""

import pandas as pd

from config import FIXED_SEED, EVAL_K, USER_RATING_THRESHOLD
from config import KNN_SIMILARITIES, KNN_K_VALUES, KNN_GAMMA, KNN_THRESHOLD
from src.baselines import ItemKNN, MostPopular
from src.eval_metrics import evaluate


# Tries every similarity and k on the validation set and returns the tuning table
# together with the pair with the best NDCG
def tune_item_knn(train, validate, all_users, all_items):
    rows = []
    for similarity in KNN_SIMILARITIES:
        for k in KNN_K_VALUES:
            knn = ItemKNN(k, similarity, KNN_GAMMA, KNN_THRESHOLD).fit(train, all_users, all_items)
            recommendations = knn.recommend(validate.user_id.unique(), n=EVAL_K)
            metrics = evaluate(recommendations, validate, len(all_items))
            rows.append({"similarity": similarity, "k": k, **metrics})
            print(f"ItemKNN validation: {similarity}, k={k}, NDCG@{EVAL_K}={metrics[f'ndcg@{EVAL_K}']:.4f}")

    tuning = pd.DataFrame(rows)
    best = tuning.loc[tuning[f"ndcg@{EVAL_K}"].idxmax()]
    return tuning, best.similarity, int(best.k)


# Evaluates a fitted model on the test set and returns one row of the results template
def test_result(model, test, n_items, model_name, settings, similarity="", k=""):
    recommendations = model.recommend(test.user_id.unique(), n=EVAL_K)
    metrics = evaluate(recommendations, test, n_items)
    print(f"{model_name} ({settings}) test: {metrics}")
    return {
        "Model": model_name,
        "Variant / settings": settings,
        "Similarity": similarity,
        "k": k,
        "Precision@10": metrics[f"precision@{EVAL_K}"],
        "Recall@10": metrics[f"recall@{EVAL_K}"],
        "NDCG@10": metrics[f"ndcg@{EVAL_K}"],
        "Catalogue coverage": metrics["coverage"],
        "Seed": FIXED_SEED,
        "Notes": f"trained on train+validation, evaluated on {metrics['n_users']} test users",
    }


# Tunes ItemKNN on validation, then retrains every baseline on train + validation
# and evaluates each once on the test set. Returns the test results and the tuning table
def run_baselines(train, validate, test, all_users, all_items):
    tuning, similarity, k = tune_item_knn(train, validate, all_users, all_items)
    print(f"Best ItemKNN on validation: {similarity}, k={k}")

    train_and_validate = pd.concat([train, validate])
    n_items = len(all_items)

    knn = ItemKNN(k, similarity, KNN_GAMMA, KNN_THRESHOLD).fit(train_and_validate, all_users, all_items)
    popular_all = MostPopular(only_positive=False).fit(train_and_validate)
    popular_positive = MostPopular(only_positive=True).fit(train_and_validate)

    results = [
        test_result(knn, test, n_items, "ItemKNN",
                    f"k and similarity tuned on validation; threshold={KNN_THRESHOLD}, gamma={KNN_GAMMA}",
                    similarity, k),
        test_result(popular_all, test, n_items, "MostPopular",
                    "all-time popularity, all ratings counted"),
        test_result(popular_positive, test, n_items, "MostPopular",
                    f"all-time popularity, only ratings >= {USER_RATING_THRESHOLD} counted"),
    ]
    return pd.DataFrame(results), tuning
