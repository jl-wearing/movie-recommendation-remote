"""
Evaluate the Naive Bayes movie recommender beyond plain accuracy.

Replicates the "Evaluating classification performance" section of Chapter 2 of
*Python Machine Learning By Example, 4th Edition* (Yuxi H. Liu, Packt, 2024).

Because the dataset is imbalanced, accuracy alone is misleading. Here we compute
the confusion matrix, precision, recall, F1, and the ROC curve / AUC, and save
the confusion-matrix heatmap and ROC curve as images.
"""
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

from data_prep import prepare
from movie_recommender import train_recommender
import visualize


def compute_roc_points(pos_prob, Y_test, thresholds=None):
    """Manually sweep probability thresholds to build the ROC curve.

    Mirrors the book's hand-rolled ROC computation: for each threshold, count
    true and false positives, then convert to rates.
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.1, 0.05)

    true_pos = [0] * len(thresholds)
    false_pos = [0] * len(thresholds)
    for pred, y in zip(pos_prob, Y_test):
        for i, threshold in enumerate(thresholds):
            if pred >= threshold:
                if y == 1:
                    true_pos[i] += 1
                else:
                    false_pos[i] += 1
            else:
                break

    n_pos_test = (Y_test == 1).sum()
    n_neg_test = (Y_test == 0).sum()
    true_pos_rate = [tp / n_pos_test for tp in true_pos]
    false_pos_rate = [fp / n_neg_test for fp in false_pos]
    return false_pos_rate, true_pos_rate


def main():
    X, Y = prepare(verbose=False)
    clf, X_test, Y_test = train_recommender(X, Y)

    prediction = clf.predict(X_test)
    prediction_prob = clf.predict_proba(X_test)
    pos_prob = prediction_prob[:, 1]

    # Confusion matrix.
    cm = confusion_matrix(Y_test, prediction, labels=[0, 1])
    print('Confusion matrix [[TN, FP], [FN, TP]]:')
    print(cm)

    # Precision / recall / F1 for the positive (like) class.
    print(f'\nPrecision (like): {precision_score(Y_test, prediction, pos_label=1):.4f}')
    print(f'Recall    (like): {recall_score(Y_test, prediction, pos_label=1):.4f}')
    print(f'F1        (like): {f1_score(Y_test, prediction, pos_label=1):.4f}')
    print(f'F1     (dislike): {f1_score(Y_test, prediction, pos_label=0):.4f}')

    print('\nClassification report:')
    print(classification_report(Y_test, prediction))

    # ROC / AUC.
    auc = roc_auc_score(Y_test, pos_prob)
    print(f'AUC: {auc:.4f}')

    # Visualizations.
    visualize.plot_confusion_matrix(cm)
    false_pos_rate, true_pos_rate = compute_roc_points(pos_prob, Y_test)
    visualize.plot_roc_curve(false_pos_rate, true_pos_rate, auc)


if __name__ == '__main__':
    main()
