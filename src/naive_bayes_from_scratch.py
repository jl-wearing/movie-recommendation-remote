"""
Naive Bayes implemented from scratch on a toy movie-recommendation dataset.

Replicates the "Implementing Naive Bayes from scratch" section of Chapter 2,
"Building a Movie Recommendation Engine with Naive Bayes", from
*Python Machine Learning By Example, 4th Edition* (Yuxi H. Liu, Packt, 2024).

The toy problem: given 4 users and whether they liked 3 movies (m1, m2, m3),
plus whether they liked a target movie (Y/N), predict whether a new user with
ratings (1, 1, 0) will like the target movie.

    ID  m1  m2  m3  likes target
    1   0   1   1   Y
    2   0   0   1   N
    3   0   0   0   Y
    4   1   1   0   Y
    5   1   1   0   ?   <- test case
"""
from collections import defaultdict

import numpy as np


def get_label_indices(labels):
    """Group sample indices by their label.

    @param labels: list of labels
    @return: dict, {class1: [indices], class2: [indices]}
    """
    label_indices = defaultdict(list)
    for index, label in enumerate(labels):
        label_indices[label].append(index)
    return label_indices


def get_prior(label_indices):
    """Compute prior P(class) from the grouped indices.

    @param label_indices: grouped sample indices by class
    @return: dict, {class: prior probability}
    """
    prior = {label: len(indices) for label, indices in label_indices.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= total_count
    return prior


def get_likelihood(features, label_indices, smoothing=0):
    """Compute likelihood P(feature|class) with optional Laplace smoothing.

    @param features: numpy matrix of binary features
    @param label_indices: grouped sample indices by class
    @param smoothing: int, additive (Laplace) smoothing parameter
    @return: dict, {class: P(feature=1|class) vector}
    """
    likelihood = {}
    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        total_count = len(indices)
        likelihood[label] = likelihood[label] / (total_count + 2 * smoothing)
    return likelihood


def get_posterior(X, prior, likelihood):
    """Compute posterior P(class|sample), proportional to prior * likelihood.

    @param X: testing samples
    @param prior: dict, {class: prior}
    @param likelihood: dict, {class: P(feature=1|class) vector}
    @return: list of dicts, one posterior dict per test sample
    """
    posteriors = []
    for x in X:
        # posterior is proportional to prior * likelihood
        posterior = prior.copy()
        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                posterior[label] *= likelihood_label[index] if bool_value \
                    else (1 - likelihood_label[index])
        # normalize so that all values sum up to 1
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors


def main():
    X_train = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
        [1, 1, 0]])
    Y_train = ['Y', 'N', 'Y', 'Y']
    X_test = np.array([[1, 1, 0]])

    label_indices = get_label_indices(Y_train)
    print('label_indices:\n', dict(label_indices))

    prior = get_prior(label_indices)
    print('Prior:', prior)

    smoothing = 1
    likelihood = get_likelihood(X_train, label_indices, smoothing)
    print('Likelihood:\n', likelihood)

    posterior = get_posterior(X_test, prior, likelihood)
    print('Posterior:\n', posterior)


if __name__ == '__main__':
    main()
