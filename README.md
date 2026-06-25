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
│   └── naive_bayes_from_scratch.py   # Toy example: NB implemented by hand
├── requirements.txt
└── README.md
```

## Setup

A virtual environment (`ml-env/`) is used for all dependencies.

```bash
# activate the venv, then:
pip install -r requirements.txt
```

## Progress

- [x] **Naïve Bayes from scratch** (toy 4-user dataset)
- [ ] Naïve Bayes with scikit-learn (`BernoulliNB`)
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
