import os
import pytest

from sklearn.ensemble import RandomForestClassifier as RFC

from classification_tool.classification_tool import ClassificationTool
from loading_tool.loading_tool import LoadingTool

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets'))

class TestClassificationTool(object):

    def test_train_classifier(self):
        tr_path = os.path.join(DATA_DIR, 'test_tr')
        rfc = RFC(n_estimators=100, criterion="entropy", n_jobs=-1)
        sampling_settings = {
            'bin_count': 16,
            'neg_samples': 7,
            'bin_samples': 20,
            'seed': 0
        }
        loading_tool = LoadingTool(sampling_settings)
        clas_tool = ClassificationTool(rfc, loading_tool)
        clas_tool.train_classifier(tr_path)

        assert list(clas_tool.classifier.classes_) == [0, 1, 2, 3]

    def test_save_predictions(self):
        tr_path = os.path.join(DATA_DIR, 'test_tr')
        t_path = os.path.join(DATA_DIR, 'test_t')
        rfc = RFC(n_estimators=100, criterion="entropy", n_jobs=-1)
        sampling_settings = {
            'bin_count': 16,
            'neg_samples': 7,
            'bin_samples': 20,
            'seed': 0
        }
        loading_tool = LoadingTool(sampling_settings)
        clas_tool = ClassificationTool(rfc, loading_tool)
        clas_tool.train_classifier(tr_path)

        output_file = os.path.join(ROOT_DIR, 'outputs/rfc.test')
        clas_tool.save_predictions(t_path, output_file)
        assert os.path.isfile(output_file)

        os.remove(output_file)
