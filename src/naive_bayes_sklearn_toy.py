"""
Naive Bayes on the toy movie dataset using scikit-learn's BernoulliNB.

Replicates the "Implementing Naive Bayes with scikit-learn" section of Chapter 2
of *Python Machine Learning By Example, 4th Edition* (Yuxi H. Liu, Packt, 2024).

BernoulliNB is the right scikit-learn estimator here because the toy features are
binary (each user either liked a movie or not). It should agree with the
from-scratch posterior computed in ``naive_bayes_from_scratch.py``.
"""
import numpy as np
from sklearn.naive_bayes import BernoulliNB


def main():
    X_train = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
        [1, 1, 0]])
    Y_train = ['Y', 'N', 'Y', 'Y']
    X_test = np.array([[1, 1, 0]])

    # alpha=1.0 -> Laplace smoothing; fit_prior=True -> learn P(class) from data
    clf = BernoulliNB(alpha=1.0, fit_prior=True)
    clf.fit(X_train, Y_train)

    # predict_proba returns probabilities in clf.classes_ order: ['N', 'Y']
    pred_prob = clf.predict_proba(X_test)
    print('[scikit-learn] Classes:', clf.classes_)
    print('[scikit-learn] Predicted probabilities:\n', pred_prob)

    pred = clf.predict(X_test)
    print('[scikit-learn] Prediction:', pred)


if __name__ == '__main__':
    main()
