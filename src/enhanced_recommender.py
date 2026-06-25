"""
Movie recommender with extra features (book Exercise 1).

Compares four feature sets, all using MultinomialNB and the same 80/20 split:

  1. ratings only            -- the baseline from chapter 2
  2. ratings + genres        -- adds 18 genre-preference features
  3. ratings + demographics  -- adds 30 one-hot demographic features
  4. ratings + both

Each variant is scored by accuracy and AUC so we can see whether the extra
features actually help.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from data_prep import (build_rating_matrix, load_rating_data, RECOMMEND_THRESHOLD,
                       TARGET_MOVIE_ID)
import features
import visualize

RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_feature_sets():
    """Return (feature_sets dict, Y) where each value is an (n_samples x d) X."""
    df, n_users, n_movies = load_rating_data()
    data, movie_id_mapping = build_rating_matrix(df, n_users, n_movies)
    target_col = movie_id_mapping[TARGET_MOVIE_ID]

    # Base movie-rating features (target column removed).
    X_movies_all = np.delete(data, target_col, axis=1).astype(np.float64)
    genre_all = features.build_genre_features(data, movie_id_mapping, target_col)
    demo_all, _ = features.build_demographic_features(n_users)

    # Keep only users who actually rated the target movie.
    Y_raw = data[:, target_col]
    mask = Y_raw > 0
    X_movies = X_movies_all[mask]
    genre = genre_all[mask]
    demo = demo_all[mask]

    Y = Y_raw[mask].copy()
    Y[Y <= RECOMMEND_THRESHOLD] = 0
    Y[Y > RECOMMEND_THRESHOLD] = 1

    feature_sets = {
        'ratings only': X_movies,
        'ratings + genres': np.hstack([X_movies, genre]),
        'ratings + demographics': np.hstack([X_movies, demo]),
        'ratings + both': np.hstack([X_movies, genre, demo]),
    }
    return feature_sets, Y


def evaluate_variant(X, Y, alpha=1.0, fit_prior=True):
    """Train MultinomialNB on an 80/20 split and return (accuracy, auc)."""
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
    clf.fit(X_train, Y_train)
    accuracy = clf.score(X_test, Y_test)
    pos_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(Y_test, pos_prob)
    return accuracy, auc


def main():
    feature_sets, Y = build_feature_sets()

    print(f'{"feature set":<24} {"dims":>6} {"accuracy":>10} {"AUC":>8}')
    results = {}
    for name, X in feature_sets.items():
        accuracy, auc = evaluate_variant(X, Y)
        results[name] = {'accuracy': accuracy, 'auc': auc}
        print(f'{name:<24} {X.shape[1]:>6} {accuracy * 100:>9.1f}% {auc:>8.4f}')

    visualize.plot_feature_comparison(results)


if __name__ == '__main__':
    main()
