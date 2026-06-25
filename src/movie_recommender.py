"""
Train the Naive Bayes movie recommender on MovieLens 1M.

Replicates the "Training a Naive Bayes model" section of Chapter 2 of
*Python Machine Learning By Example, 4th Edition* (Yuxi H. Liu, Packt, 2024).

We use MultinomialNB (not BernoulliNB) because the rating features are integers
0-5 rather than binary. The script also saves two data-understanding plots.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from data_prep import (build_dataset, build_rating_matrix, load_rating_data)
import visualize

RANDOM_STATE = 42
TEST_SIZE = 0.2


def train_recommender(X, Y, alpha=1.0, fit_prior=True, test_size=TEST_SIZE,
                      random_state=RANDOM_STATE):
    """Split, train MultinomialNB, and return (clf, X_test, Y_test)."""
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state)
    print(f'Training samples: {len(Y_train)}, testing samples: {len(Y_test)}')

    clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
    clf.fit(X_train, Y_train)
    return clf, X_test, Y_test


def main():
    # Load full matrix so we can also plot the rating distribution.
    df, n_users, n_movies = load_rating_data()
    data, movie_id_mapping = build_rating_matrix(df, n_users, n_movies)
    X, Y = build_dataset(data, movie_id_mapping)

    n_pos = (Y == 1).sum()
    n_neg = (Y == 0).sum()
    print(f'{n_pos} positive samples and {n_neg} negative samples.')

    # Data-understanding visualizations.
    visualize.plot_rating_distribution(data)
    visualize.plot_class_balance(Y)

    # Train and score.
    clf, X_test, Y_test = train_recommender(X, Y)

    prediction_prob = clf.predict_proba(X_test)
    print('First 10 predicted probabilities [P(0), P(1)]:')
    print(np.round(prediction_prob[:10], 4))

    prediction = clf.predict(X_test)
    print('First 10 predictions:', prediction[:10])

    accuracy = clf.score(X_test, Y_test)
    print(f'The accuracy is: {accuracy * 100:.1f}%')


if __name__ == '__main__':
    main()
