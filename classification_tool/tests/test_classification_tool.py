import os
import pytest

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC

from classification_tool.classification_tool import ClassificationTool
from evaluation_tool.evaluation_tool import EvaluationTool

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))


class TestClassificationTool(object):

    def test_load_datasets(self):
        training_file_path = os.path.join(ROOT_DIR, 'datasets/letter.scale.tr')
        testing_file_path = os.path.join(ROOT_DIR, 'datasets/letter.scale.t')
        clas_tool = ClassificationTool(training_file_path, testing_file_path)

        assert (clas_tool.testing_data[0].shape[0]
                + clas_tool.training_data[0].shape[0]) == 15500

    def test_train_classifier(self):
        training_file_path = os.path.join(ROOT_DIR, 'datasets/letter.scale.tr')
        testing_file_path = os.path.join(ROOT_DIR, 'datasets/letter.scale.t')
        clas_tool = ClassificationTool(training_file_path, testing_file_path)

        rfc = RFC(n_estimators=100, criterion="entropy", n_jobs=-1)
        clas_tool.train_classifier(classifier=rfc)

        # TODO: Think of a better assert
        assert pytest.approx(clas_tool.classifier.score(
            clas_tool.testing_data[0], clas_tool.testing_data[1]), 0.95)

    def test_save_predictions(self):
        training_file_path = os.path.join(ROOT_DIR, 'datasets/letter.scale.tr')
        testing_file_path = os.path.join(ROOT_DIR, 'datasets/letter.scale.t')
        clas_tool = ClassificationTool(training_file_path, testing_file_path)

        rfc = RFC(n_estimators=100, criterion="entropy", n_jobs=-1)
        clas_tool.train_classifier(classifier=rfc)

        output_file = os.path.join(ROOT_DIR, 'outputs/rfc.letter')

        clas_tool.save_predictions(output_file)

        assert os.path.isfile(output_file)

