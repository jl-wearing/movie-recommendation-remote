"""
Extra feature engineering for the movie recommender (book Exercise 1).

The base recommender uses only a user's ratings of other movies. Here we add:

  * Genre-preference features  -- from movies.dat. For each user and each of the
    18 MovieLens genres, we sum the user's ratings of movies in that genre. The
    target movie is excluded so its rating can't leak into the features.
  * Demographic features       -- from users.dat. Gender, age bucket and
    occupation are one-hot encoded (gender 2 + age 7 + occupation 21 = 30 cols).

All features are non-negative, so they stay compatible with MultinomialNB.
Rows are indexed by ``user_id - 1`` to line up with the rating matrix.
"""
import os

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOVIES_PATH = os.path.join(PROJECT_ROOT, 'data', 'ml-1m', 'movies.dat')
USERS_PATH = os.path.join(PROJECT_ROOT, 'data', 'ml-1m', 'users.dat')

# Canonical genre order from the MovieLens 1M README.
GENRES = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
          'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
GENRE_INDEX = {g: i for i, g in enumerate(GENRES)}

# Demographic category codes from the README.
AGE_CODES = [1, 18, 25, 35, 45, 50, 56]
OCCUPATION_CODES = list(range(21))


def load_movie_genres(path=MOVIES_PATH):
    """Parse movies.dat into {movie_id: [genre indices]}."""
    movie_genres = {}
    with open(path, encoding='latin-1') as f:
        for line in f:
            movie_id, _title, genre_str = line.rstrip('\n').split('::')
            movie_genres[int(movie_id)] = [GENRE_INDEX[g]
                                           for g in genre_str.split('|')
                                           if g in GENRE_INDEX]
    return movie_genres


def build_genre_features(data, movie_id_mapping, target_col, path=MOVIES_PATH):
    """Build an (n_users x 18) genre-preference matrix.

    Entry [u, g] = sum of user u's ratings of movies in genre g, excluding the
    target movie. Computed as ``data @ G`` where G maps movie columns to genres.
    """
    movie_genres = load_movie_genres(path)
    n_movies = data.shape[1]
    G = np.zeros((n_movies, len(GENRES)), dtype=np.float64)
    for movie_id, col in movie_id_mapping.items():
        for g in movie_genres.get(movie_id, []):
            G[col, g] = 1.0
    # Drop the target movie so its rating can't leak into the genre sums.
    G[target_col, :] = 0.0
    return data @ G


def build_demographic_features(n_users, path=USERS_PATH):
    """Build an (n_users x 30) one-hot demographic matrix and column names."""
    n_cols = 2 + len(AGE_CODES) + len(OCCUPATION_CODES)
    feats = np.zeros((n_users, n_cols), dtype=np.float64)

    age_offset = 2
    occ_offset = age_offset + len(AGE_CODES)
    age_index = {code: i for i, code in enumerate(AGE_CODES)}

    with open(path, encoding='latin-1') as f:
        for line in f:
            user_id, gender, age, occupation, _zip = line.rstrip('\n').split('::')
            row = int(user_id) - 1
            feats[row, 0 if gender == 'M' else 1] = 1.0
            feats[row, age_offset + age_index[int(age)]] = 1.0
            feats[row, occ_offset + int(occupation)] = 1.0

    names = (['gender=M', 'gender=F']
             + [f'age={c}' for c in AGE_CODES]
             + [f'occ={c}' for c in OCCUPATION_CODES])
    return feats, names


def main():
    from data_prep import (build_rating_matrix, load_rating_data,
                           TARGET_MOVIE_ID)
    df, n_users, n_movies = load_rating_data()
    data, movie_id_mapping = build_rating_matrix(df, n_users, n_movies)
    target_col = movie_id_mapping[TARGET_MOVIE_ID]

    genre_feats = build_genre_features(data, movie_id_mapping, target_col)
    demo_feats, demo_names = build_demographic_features(n_users)

    print('Genre feature matrix:', genre_feats.shape)
    print('Demographic feature matrix:', demo_feats.shape)
    print('Demographic columns:', demo_names)
    print('\nExample (user 0) genre sums:')
    for g, v in zip(GENRES, genre_feats[0]):
        print(f'  {g:<12} {v:.0f}')


if __name__ == '__main__':
    main()
