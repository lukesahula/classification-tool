import os
from collections import defaultdict

import pandas as pd
import pytest

import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from evaluation_tool.evaluation_tool import EvaluationTool

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))


class TestEvaluationTool(object):

    def test_compute_stats(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_keys')
        eval_tool = EvaluationTool(file_path, ';')
        result = eval_tool.compute_stats()

        expected = {
            0: {
                'TP' : 1,
                'TN' : 8,
                'FP' : 2,
                'FN' : 4
            },
            1: {
                'TP': 5,
                'TN': 5,
                'FP': 3,
                'FN': 2
            },
            2: {
                'TP': 1,
                'TN': 9,
                'FP': 3,
                'FN': 2
            }
        }


        assert result == expected

    def test_compute_precision(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_keys')
        eval_tool = EvaluationTool(file_path, ';')
        stats = eval_tool.compute_stats()

        prec = [eval_tool.compute_precision(x, stats) for x in eval_tool.labels]
        prec_sklearn = list(precision_score(
            y_true=eval_tool.trues,
            y_pred=eval_tool.preds,
            labels=eval_tool.labels,
            average=None
        ))

        assert prec == prec_sklearn

    def test_compute_recall(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_keys')
        eval_tool = EvaluationTool(file_path, ';')
        stats = eval_tool.compute_stats()

        rec = [eval_tool.compute_recall(x, stats) for x in eval_tool.labels]
        rec_sklearn = list(recall_score(
            y_true=eval_tool.trues,
            y_pred=eval_tool.preds,
            labels=eval_tool.labels,
            average=None
        ))

        assert rec == rec_sklearn

    def test_get_avg_precision(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_strings')
        eval_tool = EvaluationTool(file_path, ';')
        stats = eval_tool.compute_stats()
        labels = list(stats.keys())

        prec = eval_tool.get_avg_precision(stats=stats)
        prec_avg_sklearn = precision_score(
            y_true=eval_tool.trues,
            y_pred=eval_tool.preds,
            labels=labels,
            average='macro'
        )

        assert np.allclose(prec, prec_avg_sklearn)


    def test_get_avg_recall(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_strings')
        eval_tool = EvaluationTool(file_path, ';')
        stats = eval_tool.compute_stats()

        rec = eval_tool.get_avg_recall(stats=stats)
        rec_avg_sklearn = recall_score(
            y_true=eval_tool.trues,
            y_pred=eval_tool.preds,
            labels=eval_tool.labels,
            average='macro'
        )
        assert np.allclose(rec, rec_avg_sklearn)

    def test_compute_precision_unbalanced(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_unbalanced')
        eval_tool = EvaluationTool(file_path, ';')
        stats = eval_tool.compute_stats()

        prec = [eval_tool.compute_precision(x, stats) for x in eval_tool.labels]

        assert np.isnan(prec[4])

    def test_compute_recall_unbalanced(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_unbalanced')
        eval_tool = EvaluationTool(file_path, ';')
        stats = eval_tool.compute_stats()

        rec = [eval_tool.compute_recall(x, stats) for x in eval_tool.labels]

        assert np.isnan(rec[3])

    def test_get_avg_prec_legit(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_keys')
        eval_tool = EvaluationTool(file_path, ';', legit=0)
        stats = eval_tool.compute_stats()

        prec = eval_tool.get_avg_precision(stats, legit=False)
        eval_tool.labels.remove(0)
        prec_avg_sklearn = precision_score(
            y_true=eval_tool.trues,
            y_pred=eval_tool.preds,
            labels=eval_tool.labels,
            average='macro'
        )

        assert np.allclose(prec, prec_avg_sklearn)

    def test_get_avg_rec_legit(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_keys')
        eval_tool = EvaluationTool(file_path, ';', legit=0)
        stats = eval_tool.compute_stats()

        rec = eval_tool.get_avg_recall(stats=stats, legit=False)
        eval_tool.labels.remove(0)
        rec_avg_sklearn = recall_score(
            y_true=eval_tool.trues,
            y_pred=eval_tool.preds,
            labels=eval_tool.labels,
            average='macro'
        )

        assert np.allclose(rec, rec_avg_sklearn)


    def test_get_avg_prec_nans_false(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_unbalanced')
        eval_tool = EvaluationTool(file_path, ';')
        stats = eval_tool.compute_stats()

        prec = eval_tool.get_avg_precision(stats, nan=False)

        # TODO: Think of a better assert
        assert np.allclose(prec, 0.28472)

    def test_get_avg_rec_nans_false(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_unbalanced')
        eval_tool = EvaluationTool(file_path, ';')
        stats = eval_tool.compute_stats()

        rec = eval_tool.get_avg_recall(stats=stats, nan=False)

        # TODO: Think of a better assert
        assert np.allclose(rec, 0.30357)

    def test_get_avg_prec_nans_true(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_unbalanced')
        eval_tool = EvaluationTool(file_path, ';')
        stats = eval_tool.compute_stats()

        prec = eval_tool.get_avg_precision(stats, nan=True)

        # TODO: Think of a better assert
        assert np.allclose(prec, 0.227777)

    def test_get_avg_rec_nans_true(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_unbalanced')
        eval_tool = EvaluationTool(file_path, ';')
        stats = eval_tool.compute_stats()

        rec = eval_tool.get_avg_recall(stats=stats, nan=True)


        # TODO: Think of a better assert
        assert np.allclose(rec, 0.242857)

    def test_read_keys(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_keys')
        eval_tool = EvaluationTool(file_path, ';', agg=True)

        expected = pd.DataFrame(
            np.array([
                [1, 2, 1, 2, 1, 1, 2, 0, 1, 0, 1, 2, 0, 1, 1],
                [1, 2, 1, 2, 1, 1, 2, 0, 1, 0, 1, 2, 0, 1, 1],
                [1, 2, 1, 2, 1, 1, 2, 0, 1, 0, 1, 2, 0, 1, 1]
            ]).transpose(),
            columns=['timestamp', 'user', 'flow']
        )

        assert eval_tool.metadata.equals(expected)

    def test_compute_aggregated_stats(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_agg')
        eval_tool = EvaluationTool(file_path, ';', agg=True)
        aggregated_stats = eval_tool.compute_aggregated_stats('user')

        expected_stats = {
            1: {
                'TP' : 3,
                'FP' : 1,
                'FN' : 1
            },
            2: {
                'TP': 1,
                'FP': 0,
                'FN': 0
            },
            3: {
                'TP': 0,
                'FP': 2,
                'FN': 2
            }
        }
        assert aggregated_stats == expected_stats

    def test_get_stats_counts(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_agg')
        eval_tool = EvaluationTool(file_path, ';', agg=True)
        aggregated_stats = eval_tool.compute_aggregated_stats('user')
        expected_counts = {
            'TP': 4,
            'FP': 3,
            'FN': 3
        }
        counts = eval_tool.get_stats_counts(eval_tool.labels, aggregated_stats)
        assert expected_counts == counts
