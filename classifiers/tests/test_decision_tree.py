from classifiers.decision_tree import DecisionTree
from classification_tool.classification_tool import ClassificationTool
from loading_tool.loading_tool import LoadingTool
from evaluation_tool.evaluation_tool import EvaluationTool

import os
import pandas as pd
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class TestDecisionTree():
    print(' ')
    tree = DecisionTree(
        max_features='sqrt', min_samples_split=2, random_state=0)
    sktree = DecisionTreeClassifier(
        criterion='entropy', min_samples_split=2, max_features='sqrt',
        random_state=0
    )
    data = pd.read_csv(os.path.join(ROOT_DIR, 'datasets', 'letter'))
    X = data[data.columns[1:]]
    y = data[data.columns[0]]
    y = y.apply(lambda x: ord(x))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.98)
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    tree.fit(X_train, y_train)
    sktree.fit(X_train, y_train)

    tree_output_file = os.path.join(ROOT_DIR, 'outputs/tree.test')
    sktree_output_file = os.path.join(ROOT_DIR, 'outputs/sktree.test')
    if os.path.isfile(tree_output_file):
        os.remove(tree_output_file)
    if os.path.isfile(sktree_output_file):
        os.remove(sktree_output_file)

    tree_clas_tool = ClassificationTool(tree)
    sktree_clas_tool = ClassificationTool(sktree)

    tree_clas_tool.save_predictions(
        (X_test, y_test), tree_output_file, False, legit=None
    )
    sktree_clas_tool.save_predictions(
        (X_test, y_test), sktree_output_file, False, legit=None
    )

    loading_tool = LoadingTool()
    eval_tool = EvaluationTool()
    stats = defaultdict(lambda: defaultdict(int))
    trues = pd.Series()
    preds = pd.Series()
    for chunk in loading_tool.load_classifications(tree_output_file, ';'):
        chunk_stats = eval_tool.compute_stats(chunk)
        trues = trues.append(chunk[0])
        preds = preds.append(chunk[1])
        for label in chunk_stats:
            stats[label]['FP'] += chunk_stats[label]['FP']
            stats[label]['FN'] += chunk_stats[label]['FN']
            stats[label]['TP'] += chunk_stats[label]['TP']

    prec = eval_tool.get_avg_precision(stats=stats)

    skstats = defaultdict(lambda: defaultdict(int))
    sktrues = pd.Series()
    skpreds = pd.Series()
    for chunk in loading_tool.load_classifications(sktree_output_file, ';'):
        chunk_stats = eval_tool.compute_stats(chunk)
        sktrues = trues.append(chunk[0])
        skpreds = preds.append(chunk[1])
        for label in chunk_stats:
            skstats[label]['FP'] += chunk_stats[label]['FP']
            skstats[label]['FN'] += chunk_stats[label]['FN']
            skstats[label]['TP'] += chunk_stats[label]['TP']

    skprec = eval_tool.get_avg_precision(stats=skstats)

    print('Scikit prec: ' + str(skprec))
    print('My prec: ' + str(prec))
