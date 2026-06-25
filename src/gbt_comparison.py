"""
Re-run the feature comparison with gradient-boosted trees.

Section 7 showed that genre/demographic features did not help MultinomialNB,
because its count-based likelihood is sensitive to feature scale. Tree-based
models split on thresholds and are invariant to feature scale, so they are a
fairer test of whether those features carry useful signal.

This script evaluates the same four feature sets with both models and charts
the AUC side by side.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from enhanced_recommender import build_feature_sets, evaluate_variant
import visualize

RANDOM_STATE = 42
TEST_SIZE = 0.2


def evaluate_gbt(X, Y):
    """Train a histogram gradient-boosting classifier; return (accuracy, auc)."""
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    clf.fit(X_train, Y_train)
    accuracy = clf.score(X_test, Y_test)
    pos_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(Y_test, pos_prob)
    return accuracy, auc


def main():
    feature_sets, Y = build_feature_sets()

    nb_results = {}
    gbt_results = {}
    header = f'{"feature set":<24} {"NB acc":>8} {"NB AUC":>8} {"GBT acc":>9} {"GBT AUC":>9}'
    print(header)
    for name, X in feature_sets.items():
        nb_acc, nb_auc = evaluate_variant(X, Y)
        gbt_acc, gbt_auc = evaluate_gbt(X, Y)
        nb_results[name] = {'accuracy': nb_acc, 'auc': nb_auc}
        gbt_results[name] = {'accuracy': gbt_acc, 'auc': gbt_auc}
        print(f'{name:<24} {nb_acc * 100:>7.1f}% {nb_auc:>8.4f} '
              f'{gbt_acc * 100:>8.1f}% {gbt_auc:>9.4f}')

    visualize.plot_feature_comparison(gbt_results,
                                      filename='gbt_feature_comparison.png')
    visualize.plot_model_auc_comparison(nb_results, gbt_results)


if __name__ == '__main__':
    main()
