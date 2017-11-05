import os
import pytest

import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from ..evaluation_tool import EvaluationTool

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))


class TestEvaluationTool(object):

    def test_read_data(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_keys')
        eval_tool = EvaluationTool(file_path, ';')

        trues = [1, 0, 1, 0, 1, 2, 0, 0, 2, 1, 1, 2, 1, 1, 0]
        preds = [1, 2, 1, 2, 1, 1, 2, 0, 1, 0, 1, 2, 0, 1, 1]

        assert list(eval_tool.trues), list(eval_tool.preds) == (trues, preds)

    def test_compute_stats(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_keys')
        eval_tool = EvaluationTool(file_path, ';')

        stats = {
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

        assert eval_tool.stats == stats

    def test_compute_precision(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_keys')
        eval_tool = EvaluationTool(file_path, ';')

        prec = [eval_tool.compute_precision(x) for x in eval_tool.labels]
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

        rec = [eval_tool.compute_recall(x) for x in eval_tool.labels]
        rec_sklearn = list(recall_score(
            y_true=eval_tool.trues,
            y_pred=eval_tool.preds,
            labels=eval_tool.labels,
            average=None
        ))

        assert rec == rec_sklearn

    def test_read_data_strings(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/min_example_strings')
        eval_tool = EvaluationTool(file_path, ';')

        trues = ['b', 'a', 'b', 'a', 'b', 'c', 'a', 'a',
                 'c', 'b', 'b', 'c', 'b', 'b', 'a']
        preds = ['b', 'c', 'b', 'c', 'b', 'b', 'c', 'a',
                 'b', 'a', 'b', 'c', 'a', 'b', 'b']

        assert list(eval_tool.trues), list(eval_tool.preds) == (trues, preds)

    def test_compute_stats_strings(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/min_example_strings')
        eval_tool = EvaluationTool(file_path, ';')

        stats = {
            'a': {
                'TP' : 1,
                'TN' : 8,
                'FP' : 2,
                'FN' : 4
            },
            'b': {
                'TP': 5,
                'TN': 5,
                'FP': 3,
                'FN': 2
            },
            'c': {
                'TP': 1,
                'TN': 9,
                'FP': 3,
                'FN': 2
            }
        }

        assert eval_tool.stats == stats

    def test_compute_precision_strings(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_strings')
        eval_tool = EvaluationTool(file_path, ';')

        prec = [eval_tool.compute_precision(x) for x in eval_tool.labels]
        prec_sklearn = list(precision_score(
            y_true=eval_tool.trues,
            y_pred=eval_tool.preds,
            labels=eval_tool.labels,
            average=None
        ))

        assert prec == prec_sklearn

    def test_compute_recall_strings(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_strings')
        eval_tool = EvaluationTool(file_path, ';')

        rec = [eval_tool.compute_recall(x) for x in eval_tool.labels]
        rec_sklearn = list(recall_score(
            y_true=eval_tool.trues,
            y_pred=eval_tool.preds,
            labels=eval_tool.labels,
            average=None,

        ))

        assert rec == rec_sklearn

    def test_get_avg_precision(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_strings')
        eval_tool = EvaluationTool(file_path, ';')

        prec = eval_tool.get_avg_precision()
        prec_avg_sklearn = precision_score(
            y_true=eval_tool.trues,
            y_pred=eval_tool.preds,
            labels=eval_tool.labels,
            average='macro'
        )

        assert np.allclose(prec, prec_avg_sklearn)


    def test_get_avg_recall(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_strings')
        eval_tool = EvaluationTool(file_path, ';')

        rec = eval_tool.get_avg_recall()
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

        prec = [eval_tool.compute_precision(x) for x in eval_tool.labels]

        assert np.isnan(prec[4])

    def test_compute_recall_unbalanced(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_unbalanced')
        eval_tool = EvaluationTool(file_path, ';')

        rec = [eval_tool.compute_recall(x) for x in eval_tool.labels]

        assert np.isnan(rec[3])

    def test_get_avg_prec_legit(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_keys')
        eval_tool = EvaluationTool(file_path, ';', legit=0)

        prec = eval_tool.get_avg_precision(legit=False)
        labels = list(eval_tool.labels)
        labels.remove(eval_tool.legit)
        prec_avg_sklearn = precision_score(
            y_true=eval_tool.trues,
            y_pred=eval_tool.preds,
            labels=labels,
            average='macro'
        )

        assert np.allclose(prec, prec_avg_sklearn)

    def test_get_avg_rec_legit(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_keys')
        eval_tool = EvaluationTool(file_path, ';', legit=0)

        rec = eval_tool.get_avg_recall(legit=False)
        labels = list(eval_tool.labels)
        labels.remove(eval_tool.legit)
        rec_avg_sklearn = recall_score(
            y_true=eval_tool.trues,
            y_pred=eval_tool.preds,
            labels=labels,
            average='macro'
        )

        assert np.allclose(rec, rec_avg_sklearn)


    def test_get_avg_prec_nans_false(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_unbalanced')
        eval_tool = EvaluationTool(file_path, ';')

        prec = eval_tool.get_avg_precision(nan=False)

        # TODO: Think of a better assert
        assert np.allclose(prec, 0.28472)

    def test_get_avg_rec_nans_false(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_unbalanced')
        eval_tool = EvaluationTool(file_path, ';')

        rec = eval_tool.get_avg_recall(nan=False)

        # TODO: Think of a better assert
        assert np.allclose(rec, 0.30357)

    def test_get_avg_prec_nans_true(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_unbalanced')
        eval_tool = EvaluationTool(file_path, ';')

        prec_avg = eval_tool.get_avg_precision(nan=True)

        # TODO: Think of a better assert
        assert np.allclose(prec_avg, 0.227777)

    def test_get_avg_rec_nans_true(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_unbalanced')
        eval_tool = EvaluationTool(file_path, ';')

        rec_avg = eval_tool.get_avg_recall(nan=True)


        # TODO: Think of a better assert
        assert np.allclose(rec_avg, 0.242857)

    def test_read_keys(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_keys')
        eval_tool = EvaluationTool(file_path, ';', agg_key='user')

        keys = ['user', 'hostname', 'pcapID', 'user', 'hostname', 'user',
                'user', 'user', 'hostname', 'user', 'hostname', 'pcapID',
                'hostname', 'user', 'user']

        assert list(eval_tool.keys) == keys
