"""
Tune the Naive Bayes recommender with k-fold cross-validation.

Replicates the "Tuning models with cross-validation" section of Chapter 2 of
*Python Machine Learning By Example, 4th Edition* (Yuxi H. Liu, Packt, 2024).

A single fixed train/test split can be misleading, so we use StratifiedKFold to
estimate generalization performance, grid-searching over the smoothing factor
(alpha) and whether to learn the prior (fit_prior). Models are compared on AUC
(not accuracy) because the dataset is imbalanced.

Note: the book writes ``StratifiedKFold(n_splits=k, random_state=42)``. Modern
scikit-learn requires ``shuffle=True`` whenever ``random_state`` is given, so we
use that here. The methodology is identical; absolute AUCs may differ slightly
from the book's because the folds are shuffled.
"""
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import MultinomialNB

from data_prep import prepare
import visualize

K = 5
SMOOTHING_FACTOR_OPTION = [1, 2, 3, 4, 5, 6]
FIT_PRIOR_OPTION = [True, False]
RANDOM_STATE = 42


def cross_validate(X, Y, k=K):
    """Grid-search alpha x fit_prior, returning summed AUC per combination."""
    k_fold = StratifiedKFold(n_splits=k, shuffle=True,
                             random_state=RANDOM_STATE)
    auc_record = {}
    for train_indices, test_indices in k_fold.split(X, Y):
        X_train_k, X_test_k = X[train_indices], X[test_indices]
        Y_train_k, Y_test_k = Y[train_indices], Y[test_indices]
        for alpha in SMOOTHING_FACTOR_OPTION:
            if alpha not in auc_record:
                auc_record[alpha] = {}
            for fit_prior in FIT_PRIOR_OPTION:
                clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
                clf.fit(X_train_k, Y_train_k)
                pos_prob = clf.predict_proba(X_test_k)[:, 1]
                auc = roc_auc_score(Y_test_k, pos_prob)
                auc_record[alpha][fit_prior] = \
                    auc + auc_record[alpha].get(fit_prior, 0.0)
    return auc_record


def best_params(auc_record, k=K):
    """Return (alpha, fit_prior, mean_auc) for the highest mean AUC."""
    best = max(
        ((alpha, fit_prior, total / k)
         for alpha, by_prior in auc_record.items()
         for fit_prior, total in by_prior.items()),
        key=lambda t: t[2])
    return best


def main():
    X, Y = prepare(verbose=False)

    auc_record = cross_validate(X, Y)

    print('smoothing  fit_prior  auc')
    for alpha, by_prior in auc_record.items():
        for fit_prior, total in by_prior.items():
            print(f'    {alpha}        {fit_prior}    {total / K:.5f}')

    alpha, fit_prior, mean_auc = best_params(auc_record)
    print(f'\nBest params: alpha={alpha}, fit_prior={fit_prior} '
          f'(mean CV AUC = {mean_auc:.5f})')

    # Retrain on the standard 80/20 split with the best params and report AUC.
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=RANDOM_STATE)
    clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
    clf.fit(X_train, Y_train)
    pos_prob = clf.predict_proba(X_test)[:, 1]
    print(f'AUC with the best model: {roc_auc_score(Y_test, pos_prob):.4f}')

    visualize.plot_cv_auc_heatmap(auc_record, K)


if __name__ == '__main__':
    main()
