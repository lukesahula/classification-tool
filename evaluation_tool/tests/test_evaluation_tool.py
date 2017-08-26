
import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from ..evaluation_tool import EvaluationTool



class TestEvaluationTool(object):


    def test_read_data_minimal(self):
        file_path = 'datasets/tests/min_example'
        eval_tool = EvaluationTool(file_path, ';')

        assert len(eval_tool.true) == 15
        assert len(eval_tool.pred) == 15
        assert eval_tool.true.value_counts()[0] == 5
        assert eval_tool.true.value_counts()[1] == 7
        assert eval_tool.true.value_counts()[2] == 3
        assert eval_tool.pred.value_counts()[0] == 3
        assert eval_tool.pred.value_counts()[1] == 8
        assert eval_tool.pred.value_counts()[2] == 4

    def test_compute_stats_minimal(self):
        file_path = 'datasets/tests/min_example'
        eval_tool = EvaluationTool(file_path, ';')

        assert len(eval_tool.stats) == 3

        assert eval_tool.stats[0]['TP'] == 1
        assert eval_tool.stats[0]['TN'] == 8
        assert eval_tool.stats[0]['FP'] == 2
        assert eval_tool.stats[0]['FN'] == 4

        assert eval_tool.stats[1]['TP'] == 5
        assert eval_tool.stats[1]['TN'] == 5
        assert eval_tool.stats[1]['FP'] == 3
        assert eval_tool.stats[1]['FN'] == 2

        assert eval_tool.stats[2]['TP'] == 1
        assert eval_tool.stats[2]['TN'] == 9
        assert eval_tool.stats[2]['FP'] == 3
        assert eval_tool.stats[2]['FN'] == 2

    def test_compute_precision_minimal(self):
        file_path = 'datasets/tests/min_example'
        eval_tool = EvaluationTool(file_path, ';')

        prec = [eval_tool.compute_precision(x) for x in eval_tool.labels]
        prec_sklearn = list(precision_score(
            y_true=eval_tool.true,
            y_pred=eval_tool.pred,
            labels=eval_tool.labels,
            average=None
        ))

        assert prec == prec_sklearn

    def test_compute_recall_minimal(self):
        file_path = 'datasets/tests/min_example'
        eval_tool = EvaluationTool(file_path, ';')

        rec = [eval_tool.compute_recall(x) for x in eval_tool.labels]
        rec_sklearn = list(recall_score(
            y_true=eval_tool.true,
            y_pred=eval_tool.pred,
            labels=eval_tool.labels,
            average=None
        ))

        assert rec == rec_sklearn



    def test_read_data_minimal_string(self):
        file_path = 'datasets/tests/min_example_strings'
        eval_tool = EvaluationTool(file_path, ';')

        assert len(eval_tool.true) == 15
        assert len(eval_tool.pred) == 15
        assert eval_tool.true.value_counts()['a'] == 5
        assert eval_tool.true.value_counts()['b'] == 7
        assert eval_tool.true.value_counts()['c'] == 3
        assert eval_tool.pred.value_counts()['a'] == 3
        assert eval_tool.pred.value_counts()['b'] == 8
        assert eval_tool.pred.value_counts()['c'] == 4

    def test_compute_stats_minimal_string(self):
        file_path = 'datasets/tests/min_example_strings'
        eval_tool = EvaluationTool(file_path, ';')

        assert len(eval_tool.stats) == 3

        assert eval_tool.stats['a']['TP'] == 1
        assert eval_tool.stats['a']['TN'] == 8
        assert eval_tool.stats['a']['FP'] == 2
        assert eval_tool.stats['a']['FN'] == 4

        assert eval_tool.stats['b']['TP'] == 5
        assert eval_tool.stats['b']['TN'] == 5
        assert eval_tool.stats['b']['FP'] == 3
        assert eval_tool.stats['b']['FN'] == 2

        assert eval_tool.stats['c']['TP'] == 1
        assert eval_tool.stats['c']['TN'] == 9
        assert eval_tool.stats['c']['FP'] == 3
        assert eval_tool.stats['c']['FN'] == 2

    def test_compute_precision_minimal_string(self):
        file_path = 'datasets/tests/min_example_strings'
        eval_tool = EvaluationTool(file_path, ';')

        prec = [eval_tool.compute_precision(x) for x in eval_tool.labels]
        prec_sklearn = list(precision_score(
            y_true=eval_tool.true,
            y_pred=eval_tool.pred,
            labels=eval_tool.labels,
            average=None
        ))

        assert prec == prec_sklearn

    def test_compute_recall_minimal_string(self):
        file_path = 'datasets/tests/min_example_strings'
        eval_tool = EvaluationTool(file_path, ';')

        rec = [eval_tool.compute_recall(x) for x in eval_tool.labels]
        rec_sklearn = list(recall_score(
            y_true=eval_tool.true,
            y_pred=eval_tool.pred,
            labels=eval_tool.labels,
            average=None
        ))

        assert rec == rec_sklearn

    def test_compute_precision_string(self):
        file_path = 'datasets/tests/example_strings'
        eval_tool = EvaluationTool(file_path, ';')

        prec = [eval_tool.compute_precision(x) for x in eval_tool.labels]
        prec_sklearn = list(precision_score(
            y_true=eval_tool.true,
            y_pred=eval_tool.pred,
            labels=eval_tool.labels,
            average=None
        ))

        assert prec == prec_sklearn

    def test_compute_recall_string(self):
        file_path = 'datasets/tests/example_strings'
        eval_tool = EvaluationTool(file_path, ';')

        rec = [eval_tool.compute_recall(x) for x in eval_tool.labels]
        rec_sklearn = list(recall_score(
            y_true=eval_tool.true,
            y_pred=eval_tool.pred,
            labels=eval_tool.labels,
            average=None,

        ))

        assert rec == rec_sklearn

    def test_get_avg_precision(self):
        file_path = 'datasets/tests/example_strings'
        eval_tool = EvaluationTool(file_path, ';')

        prec = eval_tool.get_avg_precision()

        prec_avg_sklearn = precision_score(
            y_true=eval_tool.true,
            y_pred=eval_tool.pred,
            labels=eval_tool.labels,
            average='macro'
        )
        assert np.allclose(prec, prec_avg_sklearn)


    def test_get_avg_recall(self):
        file_path = 'datasets/tests/example_strings'
        eval_tool = EvaluationTool(file_path, ';')

        rec = eval_tool.get_avg_recall()

        rec_avg_sklearn = recall_score(
            y_true=eval_tool.true,
            y_pred=eval_tool.pred,
            labels=eval_tool.labels,
            average='macro'
        )
        assert np.allclose(rec, rec_avg_sklearn)

    def test_compute_precision_unbalanced(self):
        file_path = 'datasets/tests/example_unbalanced'
        eval_tool = EvaluationTool(file_path, ';')

        prec = [eval_tool.compute_precision(x) for x in eval_tool.labels]

        assert np.isnan(prec[4])

    def test_compute_recall_unbalanced(self):
        file_path = 'datasets/tests/example_unbalanced'
        eval_tool = EvaluationTool(file_path, ';')

        rec = [eval_tool.compute_recall(x) for x in eval_tool.labels]

        assert np.isnan(rec[3])

    def test_get_avg_prec_legit(self):
        file_path = 'datasets/tests/min_example'
        eval_tool = EvaluationTool(file_path, ';', legit=0)

        prec = eval_tool.get_avg_precision(legit=False)

        labels = list(eval_tool.labels)
        labels.remove(eval_tool.legit)

        prec_avg_sklearn = precision_score(
            y_true=eval_tool.true,
            y_pred=eval_tool.pred,
            labels=labels,
            average='macro'
        )
        assert np.allclose(prec, prec_avg_sklearn)

    def test_get_avg_rec_legit(self):
        file_path = 'datasets/tests/min_example'
        eval_tool = EvaluationTool(file_path, ';', legit=0)

        rec = eval_tool.get_avg_recall(legit=False)

        labels = list(eval_tool.labels)
        labels.remove(eval_tool.legit)

        rec_avg_sklearn = recall_score(
            y_true=eval_tool.true,
            y_pred=eval_tool.pred,
            labels=labels,
            average='macro'
        )
        assert np.allclose(rec, rec_avg_sklearn)


    def test_get_avg_prec_nans_false(self):
        file_path = 'datasets/tests/example_unbalanced'
        eval_tool = EvaluationTool(file_path, ';')

        prec = eval_tool.get_avg_precision(nan=False)

        # TODO: Think of a better assert
        assert np.allclose(prec, 0.28472)

    def test_get_avg_rec_nans_false(self):
        file_path = 'datasets/tests/example_unbalanced'
        eval_tool = EvaluationTool(file_path, ';')

        rec = eval_tool.get_avg_recall(nan=False)

        # TODO: Think of a better assert
        assert np.allclose(rec, 0.30357)

    def test_get_avg_prec_nans_true(self):
        file_path = 'datasets/tests/example_unbalanced'
        eval_tool = EvaluationTool(file_path, ';')

        prec_avg = eval_tool.get_avg_precision(nan=True)

        # TODO: Think of a better assert
        assert np.allclose(prec_avg, 0.227777)

    def test_get_avg_rec_nans_true(self):
        file_path = 'datasets/tests/example_unbalanced'
        eval_tool = EvaluationTool(file_path, ';')

        rec_avg = eval_tool.get_avg_recall(nan=True)


        # TODO: Think of a better assert
        assert np.allclose(rec_avg, 0.242857)
