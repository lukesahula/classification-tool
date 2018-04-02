from classifiers.random_forest import RandomForest
from classification_tool.classification_tool import ClassificationTool
from loading_tool.loading_tool import LoadingTool
from evaluation_tool.evaluation_tool import EvaluationTool

import os
import pandas as pd
import math
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from joblib import Parallel

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class TestRandomForest():
    def test_random_forest(self):
        forest = RandomForest(
            max_features='sqrt', min_samples_split=2, random_state=19,
            n_estimators=10, n_jobs=-1
        )
        skforest = RandomForestClassifier(
            criterion='entropy', min_samples_split=2, max_features='sqrt',
            random_state=19, n_estimators=10, n_jobs=-1
        )
        data = pd.read_csv(
            os.path.join(ROOT_DIR, 'datasets', 'letter'), header=None
        )
        X = data[data.columns[1:]]
        y = data[data.columns[0]]
        data = None
        X.rename(columns=lambda x: x-1, inplace=True)
        y = y.apply(lambda x: ord(x))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.98, random_state=19
        )
        forest.fit(X_train, y_train)
        skforest.fit(X_train, y_train)

        forest_output_file = os.path.join(ROOT_DIR, 'outputs/forest.test')
        skforest_output_file = os.path.join(ROOT_DIR, 'outputs/skforest.test')
        if os.path.isfile(forest_output_file):
            os.remove(forest_output_file)
        if os.path.isfile(skforest_output_file):
            os.remove(skforest_output_file)

        forest_clas_tool = ClassificationTool(forest)
        skforest_clas_tool = ClassificationTool(skforest)

        with Parallel(n_jobs=-1) as p:
            forest_clas_tool.save_predictions(
                (X_test, y_test), forest_output_file, p, False, legit=None
            )
        skforest_clas_tool.save_predictions(
            (X_test, y_test), skforest_output_file, None, False, legit=None
        )

        loading_tool = LoadingTool()
        eval_tool = EvaluationTool()
        stats = defaultdict(lambda: defaultdict(int))
        trues = pd.Series()
        preds = pd.Series()
        for chunk in loading_tool.load_classifications(
            forest_output_file, ';'
        ):
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
        for chunk in loading_tool.load_classifications(
            skforest_output_file, ';'
        ):
            chunk_stats = eval_tool.compute_stats(chunk)
            sktrues = sktrues.append(chunk[0])
            skpreds = skpreds.append(chunk[1])
            for label in chunk_stats:
                skstats[label]['FP'] += chunk_stats[label]['FP']
                skstats[label]['FN'] += chunk_stats[label]['FN']
                skstats[label]['TP'] += chunk_stats[label]['TP']

        skprec = eval_tool.get_avg_precision(stats=skstats)

        assert math.isclose(prec, skprec, abs_tol=0.011)
