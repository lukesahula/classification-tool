import os
from collections import defaultdict

import pandas as pd
import pytest

import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from evaluation_tool.evaluation_tool import EvaluationTool
from loading_tool.loading_tool import LoadingTool

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))


class TestEvaluationTool(object):
    def test_compute_stats(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_keys')
        eval_tool = EvaluationTool(legit=0)
        load_tool = LoadingTool()
        result = defaultdict(lambda: defaultdict(int))
        for chunk in load_tool.load_classifications(file_path, ';', True):
            chunk_stats = eval_tool.compute_stats(chunk)
            for label in chunk_stats:
                result[label]['FP'] += chunk_stats[label]['FP']
                result[label]['FN'] += chunk_stats[label]['FN']
                result[label]['TP'] += chunk_stats[label]['TP']

        expected = defaultdict(lambda: defaultdict(int))
        expected[0]['TP'] = 1
        expected[0]['FP'] = 2
        expected[0]['FN'] = 4
        expected[1]['TP'] = 5
        expected[1]['FP'] = 3
        expected[1]['FN'] = 2
        expected[2]['TP'] = 1
        expected[2]['FP'] = 3
        expected[2]['FN'] = 2

        assert result == expected

    def test_compute_precision(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_keys')
        eval_tool = EvaluationTool(legit=0)
        load_tool = LoadingTool()
        stats = defaultdict(lambda: defaultdict(int))
        trues = pd.Series()
        preds = pd.Series()
        for chunk in load_tool.load_classifications(file_path, ';', True):
            chunk_stats = eval_tool.compute_stats(chunk)
            trues = trues.append(chunk[0])
            preds = preds.append(chunk[1])
            for label in chunk_stats:
                stats[label]['FP'] += chunk_stats[label]['FP']
                stats[label]['FN'] += chunk_stats[label]['FN']
                stats[label]['TP'] += chunk_stats[label]['TP']

        labels = [1, 2]
        prec = [eval_tool.compute_precision(x, stats) for x in labels]
        prec_sklearn = list(precision_score(
            y_true=trues,
            y_pred=preds,
            labels=labels,
            average=None
        ))

        assert prec == prec_sklearn

    def test_compute_recall(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_keys')
        eval_tool = EvaluationTool(legit=0)
        load_tool = LoadingTool()
        stats = defaultdict(lambda: defaultdict(int))
        trues = pd.Series()
        preds = pd.Series()
        for chunk in load_tool.load_classifications(file_path, ';', True):
            chunk_stats = eval_tool.compute_stats(chunk)
            trues = trues.append(chunk[0])
            preds = preds.append(chunk[1])
            for label in chunk_stats:
                stats[label]['FP'] += chunk_stats[label]['FP']
                stats[label]['FN'] += chunk_stats[label]['FN']
                stats[label]['TP'] += chunk_stats[label]['TP']

        labels = [1, 2]
        rec = [eval_tool.compute_recall(x, stats) for x in labels]
        rec_sklearn = list(recall_score(
            y_true=trues,
            y_pred=preds,
            labels=labels,
            average=None
        ))

        assert rec == rec_sklearn

    def test_get_avg_precision(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_strings')
        eval_tool = EvaluationTool()
        load_tool = LoadingTool()
        stats = defaultdict(lambda: defaultdict(int))
        trues = pd.Series()
        preds = pd.Series()
        for chunk in load_tool.load_classifications(file_path, ';'):
            chunk_stats = eval_tool.compute_stats(chunk)
            trues = trues.append(chunk[0])
            preds = preds.append(chunk[1])
            for label in chunk_stats:
                stats[label]['FP'] += chunk_stats[label]['FP']
                stats[label]['FN'] += chunk_stats[label]['FN']
                stats[label]['TP'] += chunk_stats[label]['TP']

        prec = eval_tool.get_avg_precision(stats=stats)
        prec_avg_sklearn = precision_score(
            y_true=trues,
            y_pred=preds,
            labels=eval_tool.labels,
            average='macro'
        )

        assert np.allclose(prec, prec_avg_sklearn)

    def test_get_avg_precision_from_specific_labels(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_strings')
        eval_tool = EvaluationTool()
        load_tool = LoadingTool()
        stats = defaultdict(lambda: defaultdict(int))
        trues = pd.Series()
        preds = pd.Series()
        for chunk in load_tool.load_classifications(file_path, ';'):
            chunk_stats = eval_tool.compute_stats(chunk)
            trues = trues.append(chunk[0])
            preds = preds.append(chunk[1])
            for label in chunk_stats:
                stats[label]['FP'] += chunk_stats[label]['FP']
                stats[label]['FN'] += chunk_stats[label]['FN']
                stats[label]['TP'] += chunk_stats[label]['TP']

        labels = [0, 1, 2]
        prec = eval_tool.get_avg_precision(stats=stats, par_labels=labels)
        prec_avg_sklearn = precision_score(
            y_true=trues,
            y_pred=preds,
            labels=labels,
            average='macro'
        )

        assert np.allclose(prec, prec_avg_sklearn)

    def test_get_avg_recall(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_strings')
        eval_tool = EvaluationTool()
        load_tool = LoadingTool()
        stats = defaultdict(lambda: defaultdict(int))
        trues = pd.Series()
        preds = pd.Series()
        for chunk in load_tool.load_classifications(file_path, ';'):
            chunk_stats = eval_tool.compute_stats(chunk)
            trues = trues.append(chunk[0])
            preds = preds.append(chunk[1])
            for label in chunk_stats:
                stats[label]['FP'] += chunk_stats[label]['FP']
                stats[label]['FN'] += chunk_stats[label]['FN']
                stats[label]['TP'] += chunk_stats[label]['TP']

        rec = eval_tool.get_avg_recall(stats=stats)
        rec_avg_sklearn = recall_score(
            y_true=trues,
            y_pred=preds,
            labels=eval_tool.labels,
            average='macro'
        )
        assert np.allclose(rec, rec_avg_sklearn)

    def test_get_avg_recall_from_specific_labels(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_strings')
        eval_tool = EvaluationTool()
        load_tool = LoadingTool()
        stats = defaultdict(lambda: defaultdict(int))
        trues = pd.Series()
        preds = pd.Series()
        for chunk in load_tool.load_classifications(file_path, ';'):
            chunk_stats = eval_tool.compute_stats(chunk)
            trues = trues.append(chunk[0])
            preds = preds.append(chunk[1])
            for label in chunk_stats:
                stats[label]['FP'] += chunk_stats[label]['FP']
                stats[label]['FN'] += chunk_stats[label]['FN']
                stats[label]['TP'] += chunk_stats[label]['TP']

        labels = [0, 1, 2]
        rec = eval_tool.get_avg_recall(stats=stats, par_labels=labels)
        rec_avg_sklearn = recall_score(
            y_true=trues,
            y_pred=preds,
            labels=labels,
            average='macro'
        )
        assert np.allclose(rec, rec_avg_sklearn)

    def test_compute_precision_unbalanced(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_unbalanced')
        eval_tool = EvaluationTool()
        load_tool = LoadingTool()
        stats = defaultdict(lambda: defaultdict(int))
        trues = pd.Series()
        preds = pd.Series()
        for chunk in load_tool.load_classifications(file_path, ';'):
            chunk_stats = eval_tool.compute_stats(chunk)
            trues = trues.append(chunk[0])
            preds = preds.append(chunk[1])
            for label in chunk_stats:
                stats[label]['FP'] += chunk_stats[label]['FP']
                stats[label]['FN'] += chunk_stats[label]['FN']
                stats[label]['TP'] += chunk_stats[label]['TP']

        prec = [
            eval_tool.compute_precision(x, stats) for x in eval_tool.labels
        ]

        assert np.isnan(prec[4])

    def test_compute_recall_unbalanced(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_unbalanced')
        eval_tool = EvaluationTool()
        load_tool = LoadingTool()
        stats = defaultdict(lambda: defaultdict(int))
        trues = pd.Series()
        preds = pd.Series()
        for chunk in load_tool.load_classifications(file_path, ';'):
            chunk_stats = eval_tool.compute_stats(chunk)
            trues = trues.append(chunk[0])
            preds = preds.append(chunk[1])
            for label in chunk_stats:
                stats[label]['FP'] += chunk_stats[label]['FP']
                stats[label]['FN'] += chunk_stats[label]['FN']
                stats[label]['TP'] += chunk_stats[label]['TP']

        rec = [eval_tool.compute_recall(x, stats) for x in eval_tool.labels]

        assert np.isnan(rec[3])

    def test_get_avg_prec_legit(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_keys')
        eval_tool = EvaluationTool(legit=0)
        load_tool = LoadingTool()
        stats = defaultdict(lambda: defaultdict(int))
        trues = pd.Series()
        preds = pd.Series()
        for chunk in load_tool.load_classifications(file_path, ';'):
            chunk_stats = eval_tool.compute_stats(chunk)
            trues = trues.append(chunk[0])
            preds = preds.append(chunk[1])
            for label in chunk_stats:
                stats[label]['FP'] += chunk_stats[label]['FP']
                stats[label]['FN'] += chunk_stats[label]['FN']
                stats[label]['TP'] += chunk_stats[label]['TP']

        prec = eval_tool.get_avg_precision(stats, legit=False)
        eval_tool.labels.remove(0)
        prec_avg_sklearn = precision_score(
            y_true=trues,
            y_pred=preds,
            labels=eval_tool.labels,
            average='macro'
        )

        assert np.allclose(prec, prec_avg_sklearn)

    def test_get_avg_rec_legit(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_keys')
        eval_tool = EvaluationTool(legit=0)
        load_tool = LoadingTool()
        stats = defaultdict(lambda: defaultdict(int))
        trues = pd.Series()
        preds = pd.Series()
        for chunk in load_tool.load_classifications(file_path, ';'):
            chunk_stats = eval_tool.compute_stats(chunk)
            trues = trues.append(chunk[0])
            preds = preds.append(chunk[1])
            for label in chunk_stats:
                stats[label]['FP'] += chunk_stats[label]['FP']
                stats[label]['FN'] += chunk_stats[label]['FN']
                stats[label]['TP'] += chunk_stats[label]['TP']

        rec = eval_tool.get_avg_recall(stats=stats, legit=False)
        eval_tool.labels.remove(0)
        rec_avg_sklearn = recall_score(
            y_true=trues,
            y_pred=preds,
            labels=eval_tool.labels,
            average='macro'
        )

        assert np.allclose(rec, rec_avg_sklearn)

    def test_get_avg_prec_nans_false(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_unbalanced')
        eval_tool = EvaluationTool()
        load_tool = LoadingTool()
        stats = defaultdict(lambda: defaultdict(int))
        trues = pd.Series()
        preds = pd.Series()
        for chunk in load_tool.load_classifications(file_path, ';'):
            chunk_stats = eval_tool.compute_stats(chunk)
            trues = trues.append(chunk[0])
            preds = preds.append(chunk[1])
            for label in chunk_stats:
                stats[label]['FP'] += chunk_stats[label]['FP']
                stats[label]['FN'] += chunk_stats[label]['FN']
                stats[label]['TP'] += chunk_stats[label]['TP']

        prec = eval_tool.get_avg_precision(stats, nan=False)

        # TODO: Think of a better assert
        assert np.allclose(prec, 0.28472)

    def test_get_avg_rec_nans_false(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_unbalanced')
        eval_tool = EvaluationTool()
        load_tool = LoadingTool()
        stats = defaultdict(lambda: defaultdict(int))
        trues = pd.Series()
        preds = pd.Series()
        for chunk in load_tool.load_classifications(file_path, ';'):
            chunk_stats = eval_tool.compute_stats(chunk)
            trues = trues.append(chunk[0])
            preds = preds.append(chunk[1])
            for label in chunk_stats:
                stats[label]['FP'] += chunk_stats[label]['FP']
                stats[label]['FN'] += chunk_stats[label]['FN']
                stats[label]['TP'] += chunk_stats[label]['TP']

        rec = eval_tool.get_avg_recall(stats=stats, nan=False)

        # TODO: Think of a better assert
        assert np.allclose(rec, 0.30357)

    def test_get_avg_prec_nans_true(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_unbalanced')
        eval_tool = EvaluationTool()
        load_tool = LoadingTool()
        stats = defaultdict(lambda: defaultdict(int))
        trues = pd.Series()
        preds = pd.Series()
        for chunk in load_tool.load_classifications(file_path, ';'):
            chunk_stats = eval_tool.compute_stats(chunk)
            trues = trues.append(chunk[0])
            preds = preds.append(chunk[1])
            for label in chunk_stats:
                stats[label]['FP'] += chunk_stats[label]['FP']
                stats[label]['FN'] += chunk_stats[label]['FN']
                stats[label]['TP'] += chunk_stats[label]['TP']

        prec = eval_tool.get_avg_precision(stats, nan=True)

        # TODO: Think of a better assert
        assert np.allclose(prec, 0.227777)

    def test_get_avg_rec_nans_true(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_unbalanced')
        eval_tool = EvaluationTool()
        load_tool = LoadingTool()
        stats = defaultdict(lambda: defaultdict(int))
        trues = pd.Series()
        preds = pd.Series()
        for chunk in load_tool.load_classifications(file_path, ';'):
            chunk_stats = eval_tool.compute_stats(chunk)
            trues = trues.append(chunk[0])
            preds = preds.append(chunk[1])
            for label in chunk_stats:
                stats[label]['FP'] += chunk_stats[label]['FP']
                stats[label]['FN'] += chunk_stats[label]['FN']
                stats[label]['TP'] += chunk_stats[label]['TP']

        rec = eval_tool.get_avg_recall(stats=stats, nan=True)

        # TODO: Think of a better assert
        assert np.allclose(rec, 0.242857)

    def test_read_keys(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_keys')
        load_tool = LoadingTool()
        metadata = pd.DataFrame()
        for chunk in load_tool.load_classifications(file_path, ';', True):
            metadata = metadata.append(chunk[2])

        expected = pd.DataFrame(
            np.array([
                [1, 2, 1, 2, 1, 1, 2, 0, 1, 0, 1, 2, 0, 1, 1],
                [1, 2, 1, 2, 1, 1, 2, 0, 1, 0, 1, 2, 0, 1, 1],
                [1, 2, 1, 2, 1, 1, 2, 0, 1, 0, 1, 2, 0, 1, 1]
            ]).transpose(),
            columns=['timestamp', 'host', 'user']
        )
        assert np.allclose(expected, metadata)

    def test_compute_aggregated_stats(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_agg')
        eval_tool = EvaluationTool()
        load_tool = LoadingTool()
        stats = defaultdict(lambda: defaultdict(set))
        trues = pd.Series()
        preds = pd.Series()
        metadata = pd.DataFrame()
        for chunk in load_tool.load_classifications(file_path, ';', True):
            chunk_stats = eval_tool.compute_stats_for_agg('user', chunk)
            trues = trues.append(chunk[0])
            preds = preds.append(chunk[1])
            metadata = metadata.append(chunk[2])
            for k, v in chunk_stats.items():
                stats[k]['FP'] = stats[k]['FP'] | v['FP']
                stats[k]['FN'] = stats[k]['FN'] | v['FN']
                stats[k]['TP'] = stats[k]['TP'] | v['TP']
        stats = eval_tool.aggregate_stats(stats)

        expected_stats = {
            1: {
                'TP': 3,
                'FP': 1,
                'FN': 1
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
        assert stats == expected_stats

    def test_compute_relaxed_stats(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_relax')
        eval_tool = EvaluationTool(legit=0)
        load_tool = LoadingTool()
        stats = defaultdict(lambda: defaultdict(set))
        trues = pd.Series()
        preds = pd.Series()
        metadata = pd.DataFrame()
        for chunk in load_tool.load_classifications(file_path, ';', True):
            chunk_stats = eval_tool.compute_stats_for_agg('user', chunk, True)
            trues = trues.append(chunk[0])
            preds = preds.append(chunk[1])
            metadata = metadata.append(chunk[2])
            for k, v in chunk_stats.items():
                stats[k]['FP'] = stats[k]['FP'] | v['FP']
                stats[k]['FN'] = stats[k]['FN'] | v['FN']
                stats[k]['TP'] = stats[k]['TP'] | v['TP']
        stats = eval_tool.aggregate_stats(stats)

        expected_stats = {
            0: {
                'TP': 1,
                'FP': 1,
                'FN': 1
            },
            1: {
                'TP': 7,
                'FP': 0,
                'FN': 2
            },
            2: {
                'TP': 1,
                'FP': 0,
                'FN': 0
            },
            3: {
                'TP': 2,
                'FP': 1,
                'FN': 3
            }
        }
        assert stats == expected_stats

    def test_get_stats_counts(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_agg')
        eval_tool = EvaluationTool()
        load_tool = LoadingTool()
        stats = defaultdict(lambda: defaultdict(set))
        trues = pd.Series()
        preds = pd.Series()
        metadata = pd.DataFrame()
        for chunk in load_tool.load_classifications(file_path, ';', True):
            chunk_stats = eval_tool.compute_stats_for_agg('user', chunk)
            trues = trues.append(chunk[0])
            preds = preds.append(chunk[1])
            metadata = metadata.append(chunk[2])
            for k, v in chunk_stats.items():
                stats[k]['FP'] = stats[k]['FP'] | v['FP']
                stats[k]['FN'] = stats[k]['FN'] | v['FN']
                stats[k]['TP'] = stats[k]['TP'] | v['TP']
        stats = eval_tool.aggregate_stats(stats)

        expected_counts = {
            'TP': 4,
            'FP': 3,
            'FN': 3
        }
        counts = eval_tool.get_stats_counts(eval_tool.labels, stats)
        assert expected_counts == counts

    def test_get_stats_counts_one_label(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_one_label')
        eval_tool = EvaluationTool()
        load_tool = LoadingTool()
        stats = defaultdict(lambda: defaultdict(set))
        trues = pd.Series()
        preds = pd.Series()
        metadata = pd.DataFrame()
        for chunk in load_tool.load_classifications(file_path, ';', True):
            chunk_stats = eval_tool.compute_stats_for_agg('user', chunk)
            trues = trues.append(chunk[0])
            preds = preds.append(chunk[1])
            metadata = metadata.append(chunk[2])
            for k, v in chunk_stats.items():
                stats[k]['FP'] = stats[k]['FP'] | v['FP']
                stats[k]['FN'] = stats[k]['FN'] | v['FN']
                stats[k]['TP'] = stats[k]['TP'] | v['TP']
        stats = eval_tool.aggregate_stats(stats)

        expected_counts = {
            'TP': 1,
            'FP': 0,
            'FN': 0
        }
        counts = eval_tool.get_stats_counts(1, stats)
        assert expected_counts == counts

    def test_get_labels_with_prec_above(self):
        file_path = os.path.join(ROOT_DIR, 'datasets/tests/example_keys')
        e_tool = EvaluationTool()
        l_tool = LoadingTool()
        stats = defaultdict(lambda: defaultdict(int))
        trues = pd.Series()
        preds = pd.Series()

        for chunk in l_tool.load_classifications(file_path, ';', True):
            chunk_stats = e_tool.compute_stats(chunk)
            for label in chunk_stats:
                stats[label]['FP'] += chunk_stats[label]['FP']
                stats[label]['FN'] += chunk_stats[label]['FN']
                stats[label]['TP'] += chunk_stats[label]['TP']

        prec = [e_tool.compute_precision(x, stats) for x in e_tool.labels]

        threshold = 0.3
        precs_above_threshold = e_tool.get_labels_with_prec_above_thres(
            threshold,
            e_tool.labels,
            stats
        )
        expected = [0, 1]
        assert expected == precs_above_threshold
