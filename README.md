# Movie Recommendation Engine with Naïve Bayes

A from-the-ground-up replication of **Chapter 2 — "Building a Movie Recommendation
Engine with Naïve Bayes"** from *Python Machine Learning By Example, 4th Edition*
(Yuxi (Hayden) Liu, Packt Publishing, 2024).

The project frames movie recommendation as a **binary classification** problem:
*given how a user rated other movies, predict whether they will like a target
movie.* It builds Naïve Bayes twice — once from scratch, once with scikit-learn —
then trains and evaluates it on the real [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
dataset.

## Project layout

```
movie-recommendation-engine/
├── src/
│   ├── naive_bayes_from_scratch.py   # Toy example: NB implemented by hand
│   ├── naive_bayes_sklearn_toy.py    # Toy example: NB via scikit-learn BernoulliNB
│   └── data_prep.py                  # Load + prepare MovieLens 1M into (X, Y)
├── data/                             # MovieLens 1M (gitignored, see Setup)
├── requirements.txt
└── README.md
```

## Setup

A virtual environment (`ml-env/`) is used for all dependencies.

```bash
# activate the venv, then:
pip install -r requirements.txt
```

The MovieLens 1M dataset is not committed (it lives in the gitignored `data/`).
Download and extract it once:

```bash
curl -sSL -o data/ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
python -c "import zipfile; zipfile.ZipFile('data/ml-1m.zip').extractall('data')"
```

This produces `data/ml-1m/ratings.dat` (and `movies.dat`, `users.dat`).

## Progress

- [x] **Naïve Bayes from scratch** (toy 4-user dataset)
- [x] **Naïve Bayes with scikit-learn** (`BernoulliNB`)
- [x] **MovieLens 1M data preparation** (rating matrix + binary labels)
- [ ] Movie recommender on MovieLens 1M (`MultinomialNB`)
- [ ] Classification metrics (confusion matrix, precision/recall/F1, ROC/AUC)
- [ ] Hyperparameter tuning with k-fold cross-validation

## Findings

### 1. Naïve Bayes from scratch

`src/naive_bayes_from_scratch.py` implements the four building blocks of the
algorithm — `get_label_indices`, `get_prior`, `get_likelihood` (with Laplace
smoothing) and `get_posterior` — and runs them on the book's toy dataset:

| ID | m1 | m2 | m3 | Likes target |
|----|----|----|----|--------------|
| 1  | 0  | 1  | 1  | Y |
| 2  | 0  | 0  | 1  | N |
| 3  | 0  | 0  | 0  | Y |
| 4  | 1  | 1  | 0  | Y |
| 5  | 1  | 1  | 0  | **?** |

Running it reproduces the book's numbers exactly:

```
Prior:      {'Y': 0.75, 'N': 0.25}
Likelihood: {'Y': [0.4, 0.6, 0.4], 'N': [0.333, 0.333, 0.667]}
Posterior:  [{'Y': 0.9210, 'N': 0.0790}]
```

**Takeaway:** there is a **92.1%** probability the new user likes the target
movie. Laplace smoothing (`smoothing=1`) is essential — without it, the unseen
feature value `m1=1` in the `N` class forces `P(N|x)=0` and the model would
blindly predict `Y` every time.

### 2. Naïve Bayes with scikit-learn

`src/naive_bayes_sklearn_toy.py` runs `BernoulliNB(alpha=1.0, fit_prior=True)` on
the same toy data. `BernoulliNB` is the correct estimator because the features
are binary. It agrees with the hand-written version to the last digit:

```
Predicted probabilities (N, Y): [[0.07896399 0.92103601]]
Prediction: ['Y']
```

**Takeaway:** validating a from-scratch model against a trusted library
implementation is a good sanity check — identical outputs confirm the manual
prior/likelihood/posterior math is correct. `alpha` here is scikit-learn's name
for the Laplace smoothing factor.

### 3. Preparing the MovieLens 1M data

`src/data_prep.py` turns the raw ratings into a classification dataset:

- Reads **1,000,209 ratings** from **6,040 users** across **3,706 movies**.
- Builds a dense `6040 × 3706` rating matrix (unrated cells = 0). The matrix is
  ~99% zeros — the data is extremely **sparse**.
- Picks the most-rated movie as the **target** (movie ID `2858`, *American
  Beauty*, with 3,428 ratings) so predictions can be validated.
- Frames it as binary classification: a user's ratings of the other 3,705 movies
  are the features `X`; the label `Y` is `1` if they rated the target **> 3**.

Resulting dataset:

```
Shape of X: (3428, 3705)
Shape of Y: (3428,)
2853 positive samples and 575 negative samples.
```

**Takeaway:** the dataset is **imbalanced** (~83% positive / ~17% negative). That
is the recurring theme of the rest of the project: plain accuracy will look
flattering, so we need precision/recall/F1 and AUC to judge the model honestly.
