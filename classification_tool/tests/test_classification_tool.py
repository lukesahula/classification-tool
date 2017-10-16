import os
import pytest

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC

from ..classification_tool import ClassificationTool
from ...evaluation_tool.evaluation_tool import EvaluationTool

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))


class TestClassificationTool(object):

    def test_load_datasets(self):
        tr_path = os.path.join(ROOT_DIR, 'datasets/letter.scale.tr')
        t_path = os.path.join(ROOT_DIR, 'datasets/letter.scale.t')
        clas_tool = ClassificationTool(None, tr_path, t_path)
        clas_tool.training_data = clas_tool.load_dataset(clas_tool.tr_path)
        clas_tool.testing_data = clas_tool.load_dataset(clas_tool.t_path)

        assert (clas_tool.testing_data[0].shape[0]
                + clas_tool.training_data[0].shape[0]) == 15500

    def test_train_classifier(self):
        tr_path = os.path.join(ROOT_DIR, 'datasets/letter.scale.tr')
        t_path = os.path.join(ROOT_DIR, 'datasets/letter.scale.t')
        rfc = RFC(n_estimators=100, criterion="entropy", n_jobs=-1)
        clas_tool = ClassificationTool(rfc, tr_path, t_path)
        clas_tool.t_data = clas_tool.load_dataset(clas_tool.t_path)
        clas_tool.train_classifier()

        # TODO: Think of a better assert
        assert pytest.approx(clas_tool.classifier.score(
            clas_tool.t_data[0], clas_tool.t_data[1]), 0.95)

    def test_save_predictions(self):
        tr_path = os.path.join(ROOT_DIR, 'datasets/letter.scale.tr')
        t_path = os.path.join(ROOT_DIR, 'datasets/letter.scale.t')
        rfc = RFC(n_estimators=100, criterion="entropy", n_jobs=-1)
        clas_tool = ClassificationTool(rfc, tr_path, t_path)

        clas_tool.train_classifier()

        output_file = os.path.join(ROOT_DIR, 'outputs/rfc.letter')

        clas_tool.save_predictions(output_file)

        assert os.path.isfile(output_file)
