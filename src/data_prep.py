"""
Load and prepare the MovieLens 1M dataset for the Naive Bayes movie recommender.

Replicates the "Preparing the data" section of Chapter 2 of
*Python Machine Learning By Example, 4th Edition* (Yuxi H. Liu, Packt, 2024).

We build a (n_users x n_movies) rating matrix from ratings.dat, then frame the
problem as binary classification for a single target movie:
    - features  X : a user's ratings of every *other* movie
    - label     Y : 1 if the user rated the target movie > 3 (a "like"), else 0
Only users who actually rated the target movie are kept, so predictions can be
validated against ground truth.
"""
import os

import numpy as np
import pandas as pd

# Anchor to the project root so the script works from any working directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'ml-1m', 'ratings.dat')
# Most-rated movie in MovieLens 1M (American Beauty, 1999) -> used as the target.
TARGET_MOVIE_ID = 2858
# Ratings strictly greater than this threshold count as a "like".
RECOMMEND_THRESHOLD = 3


def load_rating_data(data_path=DATA_PATH):
    """Read ratings.dat into a DataFrame and report basic stats."""
    df = pd.read_csv(data_path, header=None, sep='::', engine='python',
                     encoding='latin-1')
    df.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    n_users = df['user_id'].nunique()
    n_movies = df['movie_id'].nunique()
    print(f'Number of ratings: {len(df)}')
    print(f'Number of users: {n_users}')
    print(f'Number of movies: {n_movies}')
    return df, n_users, n_movies


def build_rating_matrix(df, n_users, n_movies):
    """Build an (n_users x n_movies) integer rating matrix.

    Unrated entries stay 0. Also returns the movie_id -> column index mapping.
    """
    data = np.zeros([n_users, n_movies], dtype=np.intc)
    movie_id_mapping = {}
    for user_id, movie_id, rating in zip(df['user_id'], df['movie_id'],
                                         df['rating']):
        user_id = int(user_id) - 1
        if movie_id not in movie_id_mapping:
            movie_id_mapping[movie_id] = len(movie_id_mapping)
        data[user_id, movie_id_mapping[movie_id]] = rating
    return data, movie_id_mapping


def describe_rating_distribution(data):
    """Print the distribution of rating values across the matrix."""
    values, counts = np.unique(data, return_counts=True)
    for value, count in zip(values, counts):
        print(f'Number of rating {value}: {count}')


def build_dataset(data, movie_id_mapping, target_movie_id=TARGET_MOVIE_ID,
                  recommend_threshold=RECOMMEND_THRESHOLD):
    """Turn the rating matrix into (X, Y) for binary classification.

    X: ratings of every movie except the target, for users who rated the target.
    Y: 1 if the user's target rating > recommend_threshold, else 0.
    """
    target_col = movie_id_mapping[target_movie_id]
    X_raw = np.delete(data, target_col, axis=1)
    Y_raw = data[:, target_col]

    # keep only users who actually rated the target movie
    X = X_raw[Y_raw > 0]
    Y = Y_raw[Y_raw > 0]

    Y[Y <= recommend_threshold] = 0
    Y[Y > recommend_threshold] = 1
    return X, Y


def prepare(data_path=DATA_PATH, target_movie_id=TARGET_MOVIE_ID,
            recommend_threshold=RECOMMEND_THRESHOLD, verbose=True):
    """End-to-end helper used by the training/evaluation scripts."""
    df, n_users, n_movies = load_rating_data(data_path)
    data, movie_id_mapping = build_rating_matrix(df, n_users, n_movies)
    X, Y = build_dataset(data, movie_id_mapping, target_movie_id,
                         recommend_threshold)
    if verbose:
        print('Shape of X:', X.shape)
        print('Shape of Y:', Y.shape)
        n_pos = (Y == 1).sum()
        n_neg = (Y == 0).sum()
        print(f'{n_pos} positive samples and {n_neg} negative samples.')
    return X, Y


def main():
    df, n_users, n_movies = load_rating_data()
    data, movie_id_mapping = build_rating_matrix(df, n_users, n_movies)

    print('\nRating distribution:')
    describe_rating_distribution(data)

    print('\nMost-rated movies (top 5):')
    print(df['movie_id'].value_counts().head())

    X, Y = build_dataset(data, movie_id_mapping)
    print('\nShape of X:', X.shape)
    print('Shape of Y:', Y.shape)
    n_pos = (Y == 1).sum()
    n_neg = (Y == 0).sum()
    print(f'{n_pos} positive samples and {n_neg} negative samples.')


if __name__ == '__main__':
    main()
