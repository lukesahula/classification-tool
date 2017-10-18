import os
import pytest

import pandas as pd

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC

from ..classification_tool import ClassificationTool
from ...evaluation_tool.evaluation_tool import EvaluationTool

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))


class TestClassificationTool(object):

    def test_load_datasets(self):
        tr_path = os.path.join(
            ROOT_DIR, 'datasets/cisco_datasets/data/20170104'
        )
        t_path = os.path.join(
            ROOT_DIR, 'datasets/cisco_datasets/data/20170111'
        )
        clas_tool = ClassificationTool(None)
        clas_tool.training_data = clas_tool.load_dataset(tr_path, '1000')
        clas_tool.testing_data = clas_tool.load_dataset(t_path, '1000')

        assert (clas_tool.testing_data[0].shape[0]
                + clas_tool.training_data[0].shape[0]) == 4000

    def test_train_classifier(self):
        tr_path = os.path.join(
            ROOT_DIR, 'datasets/cisco_datasets/data/20170104'
        )
        t_path = os.path.join(
            ROOT_DIR, 'datasets/cisco_datasets/data/20170111'
        )
        rfc = RFC(n_estimators=100, criterion="entropy", n_jobs=-1)
        clas_tool = ClassificationTool(rfc)
        clas_tool.t_data = clas_tool.load_dataset(t_path, '1000')
        clas_tool.train_classifier(tr_path, '1000')

        # TODO: Think of a better assert
        assert (clas_tool.classifier.score(
            clas_tool.t_data[0], clas_tool.t_data[1]) == 0.7585)

    def test_save_predictions(self):
        tr_path = os.path.join(
            ROOT_DIR, 'datasets/cisco_datasets/data/20170104'
        )
        t_path = os.path.join(
            ROOT_DIR, 'datasets/cisco_datasets/data/20170111'
        )
        rfc = RFC(n_estimators=100, criterion="entropy", n_jobs=-1)
        clas_tool = ClassificationTool(rfc)

        clas_tool.train_classifier(tr_path, '1000')

        output_file = os.path.join(ROOT_DIR, 'outputs/rfc.cisco')

        clas_tool.save_predictions(t_path, '1000', output_file)

        assert os.path.isfile(output_file)

    def test_quantize_data(self):
        column = pd.Series(range(20))
        clas_tool = ClassificationTool(None)
        column = clas_tool.quantize_data(column)
        print(column)
