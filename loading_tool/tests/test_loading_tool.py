import os
import pandas as pd
import pytest

from loading_tool.loading_tool import *

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets'))

class TestLoadingTool(object):

    def test_load_testing_data(self):
        path = DATA_DIR
        sampling_settings = {
            'bin_count': 16,
            'neg_samples': 2,
            'bin_samples': 2,
            'seed': 0
        }
        loading_tool = LoadingTool(sampling_settings)
        result = []
        for chunk in loading_tool.load_testing_data(path):
            result.append(chunk)

        result = (
            pd.concat([result[0][0], result[1][0]]),
            pd.concat([result[0][1], result[1][1]]),
            pd.concat([result[0][2], result[1][2]])
        )

        data = result[0]
        labels = result[1]
        metadata = result[2]

        data.reset_index(drop=True, inplace=True)
        labels.reset_index(drop=True, inplace=True)
        metadata.reset_index(drop=True, inplace=True)

        result = data, labels, metadata

        expected_labels = [0, 0, 0, 1, 2, 3]
        expected_features = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3]
        ]
        expected_metadata = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ]
        expected = (
            pd.DataFrame(expected_features),
            pd.Series(expected_labels),
            pd.DataFrame(expected_metadata)
        )
        assert (expected[0].equals(result[0])
                and expected[1].equals(result[1])
                and expected[2].equals(result[2]))


    def test_load_training_data(self):
        path = DATA_DIR
        sampling_settings = {
            'bin_count': 16,
            'neg_samples': 2,
            'bin_samples': 2,
            'seed': 0
        }
        loading_tool = LoadingTool(sampling_settings)
        result = loading_tool.load_training_data(path)
        expected_labels = [0, 0, 1, 2, 3]
        expected_features = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [3, 3, 3, 3]
        ]
        expected_metadata = [
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ]
        expected = (
            pd.DataFrame(expected_features),
            pd.Series(expected_labels),
            pd.DataFrame(expected_metadata)
        )
        assert (expected[0].equals(result[0])
                and expected[1].equals(result[1])
                and expected[2].equals(result[2]))

    def test_quantize_data_binned(self):
        path = os.path.join(DATA_DIR, 'test_quantization')
        sampling_settings = {
            'bin_count': 16,
            'neg_samples': None,
            'bin_samples': 3,
            'seed': 0
        }
        dataset = pd.read_csv(path, sep=';', header=None)
        loading_tool = LoadingTool(sampling_settings)
        loading_tool.quantize_data((dataset, None, None))
        dataset.loc[-1] = [11, 19, 14, 10]
        dataset.index = dataset.index + 1
        dataset = dataset.sort_index()
        dataset = loading_tool.quantize_data((dataset, None, None))

        expected = pd.DataFrame([
            [11.25, 19.375, 14.375, 10],
            [10.625, 10.625, 10.625, 10],
            [15.625, 15.625, 15.625, 10],
            [20.0, 20.0, 20.0, 10]
        ])
        assert expected.equals(dataset[0])

    def test_quantize_data_unbinned(self):
        path = os.path.join(DATA_DIR, 'test_quantization')
        sampling_settings = {
            'bin_count': 16,
            'neg_samples': None,
            'bin_samples': 3,
            'seed': 0
        }
        dataset = pd.read_csv(path, sep=';', header=None)
        loading_tool = LoadingTool(sampling_settings)
        dataset = loading_tool.quantize_data((dataset, None, None))

        expected = pd.DataFrame([
            [10.625, 10.625, 10.625, 10],
            [15.625, 15.625, 15.625, 10],
            [20.0, 20.0, 20.0, 10]
        ])
        assert expected.equals(dataset[0])

    def test_load_classifications(self):
        path = os.path.join(DATA_DIR, 'test_classifications')
        load_tool = LoadingTool()
        delim = ';'
        trues = pd.Series()
        preds = pd.Series()
        for chunk in load_tool.load_classifications(path, delim):
            trues = trues.append(chunk[0])
            preds = preds.append(chunk[1])
        result = (list(trues), list(preds))
        expected = (
            [10, 10, 5, 4, 4, 4, 10, 8, 10, 9],
            [10, 9, 5, 4, 9, 8, 8, 8, 10, 9],
        )
        assert result == expected

    def test_load_classifications_keys(self):
        path = os.path.join(DATA_DIR, 'test_classifications_keys')
        delim = ';'
        load_tool = LoadingTool()
        metadata = pd.DataFrame()
        for chunk in load_tool.load_classifications(path, delim, True):
            metadata = metadata.append(chunk[2])

        expected = pd.DataFrame(
            np.array([
                [1, 1, 2, 1, 2, 1, 2, 2, 2, 2],
                [1, 1, 2, 1, 2, 1, 2, 2, 2, 2],
                [1, 1, 2, 1, 2, 1, 2, 2, 2, 2],
            ]).transpose(),
            columns=['timestamp', 'user', 'flow']
        )
        assert metadata.equals(expected)

