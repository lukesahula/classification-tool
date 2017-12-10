import os
import pandas as pd
import pytest

from loading_tool.loading_tool import *

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'datasets'))

class TestLoadingTool(object):

    def test_load_cisco_dataset(self):
        path = DATA_DIR
        sampling_settings = {
            'bin_count': 16,
            'neg_samples': 2,
            'bin_samples': 2,
            'seed': 0
        }
        loading_tool = LoadingTool(sampling_settings)
        result = loading_tool.load_cisco_dataset(path)
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

    def test_sample_indices(self):
        sampling_settings = {
            'bin_count': 16,
            'neg_samples': 10,
            'bin_samples': 0,
            'seed': 0
        }
        loading_tool = LoadingTool(sampling_settings)
        indices = list(range(40))
        sampled_indices = loading_tool.sample_indices(indices, 10)
        expected = [24, 26, 2, 16, 32, 31, 25, 19, 30, 11]

        assert sampled_indices == expected

    def test_compute_bins(self):
        path = os.path.join(DATA_DIR, 'test_quantization')
        sampling_settings = {
            'bin_count': 16,
            'neg_samples': None,
            'bin_samples': 3,
            'seed': 0
        }
        dataset = pd.read_csv(path, sep=';', header=None)
        loading_tool = LoadingTool(sampling_settings)
        bins = loading_tool.compute_bins(dataset)
        expected_bin = [
            10, 10.625, 11.25, 11.875, 12.5, 13.125, 13.75, 14.375, 15.0,
            15.625, 16.25, 16.875, 17.5, 18.125, 18.75, 19.375, 20.0
        ]
        expected = [expected_bin, expected_bin, expected_bin, [10]]

        assert bins == expected

    def test_quantize_data(self):
        path = os.path.join(DATA_DIR, 'test_quantization')
        sampling_settings = {
            'bin_count': 16,
            'neg_samples': None,
            'bin_samples': 3,
            'seed': 0
        }
        dataset = pd.read_csv(path, sep=';', header=None)
        loading_tool = LoadingTool(sampling_settings)
        loading_tool.bins = loading_tool.compute_bins(dataset)
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
        delim = ';'
        result = load_classifications(path, delim)
        result = (list(result[0]), list(result[1]))
        expected = (
            [10, 10, 5, 4, 4, 4, 10, 8, 10, 9],
            [10, 9, 5, 4, 9, 8, 8, 8, 10, 9],
        )
        assert result == expected

    def test_load_classifications_keys(self):
        path = os.path.join(DATA_DIR, 'test_classifications_keys')
        delim = ';'
        result = load_classifications(path, delim, metadata=True)
        result = result[2]

        expected = pd.DataFrame(
            np.array([
                [1, 1, 2, 1, 2, 1, 2, 2, 2, 2],
                [1, 1, 2, 1, 2, 1, 2, 2, 2, 2],
                [1, 1, 2, 1, 2, 1, 2, 2, 2, 2],
            ]).transpose(),
            columns=['timestamp', 'user', 'flow']
        )
        assert result.equals(expected)

