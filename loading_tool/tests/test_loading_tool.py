import os
import pandas as pd
import pytest

from loading_tool.loading_tool import *

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets'))

class TestLoadingTool(object):

    def test_compute_nans_per_class_ratio(self):
        path = DATA_DIR
        sampling_settings = {
            'bin_count': 16,
            'neg_samples': 2,
            'bin_samples': 2,
            'seed': 0,
            'nan_value': None
        }
        loading_tool = LoadingTool(sampling_settings)
        result = loading_tool.load_training_data(path)
        series = []
        for col in result[0]:
            series.append(pd.to_numeric(result[0][col]))
        data = pd.DataFrame(series).transpose(), result[1], result[2]

        nan_ratio = loading_tool.compute_nans_per_class_ratio(data)
        expected = {
            0: 0,
            1: 0.75,
            2: 1.0,
            3: 0.25,
            4: 0.5
        }
        assert nan_ratio == expected


    def test_load_testing_data(self):
        path = DATA_DIR
        sampling_settings = {
            'bin_count': 16,
            'neg_samples': 2,
            'bin_samples': 2,
            'seed': 0,
            'nan_value': 'const'
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


        series = []
        for col in result[0]:
            series.append(pd.to_numeric(result[0][col]))

        result = data, labels, metadata
        result = pd.DataFrame(series).transpose(), result[1], result[2]

        expected_labels = [0, 0, 0, 1, 2, 3, 4]
        expected_features = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, -1000000, -1000000, -1000000],
            [-1000000, -1000000, -1000000, -1000000],
            [3, -1000000, 3, 3],
            [-1000000, 4, -1000000, 4]
        ]
        expected_features = np.array(expected_features, np.float64)
        expected_metadata = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4]
        ]
        expected = (
            pd.DataFrame(expected_features),
            pd.Series(expected_labels),
            pd.DataFrame(expected_metadata)
        )
        assert (expected[0].equals(result[0])
                and expected[1].equals(result[1])
                and expected[2].equals(result[2]))

    def test_load_training_data_const_nan_value(self):
        path = DATA_DIR
        sampling_settings = {
            'bin_count': 16,
            'neg_samples': 2,
            'bin_samples': 2,
            'seed': 0,
            'nan_value': 'const'
        }
        loading_tool = LoadingTool(sampling_settings)
        result = loading_tool.load_training_data(path)
        series = []
        for col in result[0]:
            series.append(pd.to_numeric(result[0][col]))
        result = pd.DataFrame(series).transpose(), result[1], result[2]
        expected_labels = [0, 0, 1, 2, 3, 4]
        expected_features = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, -1000000, -1000000, -1000000],
            [-1000000, -1000000, -1000000, -1000000],
            [3, -1000000, 3, 3],
            [-1000000, 4, -1000000, 4]
        ]
        expected_features = np.array(expected_features, np.float64)
        expected_metadata = [
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4]
        ]
        expected = (
            pd.DataFrame(expected_features),
            pd.Series(expected_labels),
            pd.DataFrame(expected_metadata)
        )
        assert (expected[0].equals(result[0])
                and expected[1].equals(result[1])
                and expected[2].equals(result[2]))

    def test_load_training_data_mean_nan_value(self):
        path = DATA_DIR
        sampling_settings = {
            'bin_count': 16,
            'neg_samples': 2,
            'bin_samples': 2,
            'seed': 0,
            'nan_value': 'mean'
        }
        loading_tool = LoadingTool(sampling_settings)
        result = loading_tool.load_training_data(path)
        series = []
        for col in result[0]:
            series.append(pd.to_numeric(result[0][col]))
        result = pd.DataFrame(series).transpose(), result[1], result[2]
        expected_labels = [0, 0, 1, 2, 3, 4]
        expected_features = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 4/3, 1, 1.75],
            [1, 4/3, 1, 1.75],
            [3, 4/3, 3, 3],
            [1, 4, 1, 4]
        ]
        expected_features = np.array(expected_features, np.float64)
        expected_metadata = [
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4]
        ]
        expected = (
            pd.DataFrame(expected_features),
            pd.Series(expected_labels),
            pd.DataFrame(expected_metadata)
        )
        assert (expected[0].equals(result[0])
                and expected[1].equals(result[1])
                and expected[2].equals(result[2]))

    def test_load_training_data_median_nan_value(self):
        path = DATA_DIR
        sampling_settings = {
            'bin_count': 16,
            'neg_samples': 2,
            'bin_samples': 2,
            'seed': 0,
            'nan_value': 'median'
        }
        loading_tool = LoadingTool(sampling_settings)
        result = loading_tool.load_training_data(path)
        series = []
        for col in result[0]:
            series.append(pd.to_numeric(result[0][col]))
        result = pd.DataFrame(series).transpose(), result[1], result[2]
        expected_labels = [0, 0, 1, 2, 3, 4]
        expected_features = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 1.5],
            [0.5, 0, 0, 1.5],
            [3, 0, 3, 3],
            [0.5, 4, 0, 4]
        ]
        expected_features = np.array(expected_features, np.float64)
        expected_metadata = [
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4]
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
            'seed': 0,
            'nan_value': 'const'
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
            'seed': 0,
            'nan_value': 'const'
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
            columns=['timestamp', 'host', 'user']
        )
        assert metadata.equals(expected)

