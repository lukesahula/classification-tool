import os
import pandas as pd
import numpy as np
import glob
from collections import defaultdict


class LoadingTool():
    def __init__(self, sampling_settings=None, bins=None):
        self.bins = bins
        if sampling_settings:
            self.bin_count = sampling_settings['bin_count']
            self.bin_samples = sampling_settings['bin_samples']
            self.neg_samples = sampling_settings['neg_samples']
            self.seed = np.random.RandomState(sampling_settings['seed'])
            self.nan_value = sampling_settings['nan_value']

        self.means = None
        self.medians = None

    def __compute_nan_counts_per_class(self, data):
        counts_per_row = data[0].apply(lambda x: x.count(), axis=1)
        nan_counts_per_row = []
        for c in counts_per_row:
            nan_counts_per_row.append(len(data[0].columns) - c)

        total_counts = defaultdict(int)
        nan_counts = defaultdict(int)
        for nan, v, l in zip(nan_counts_per_row, counts_per_row, data[1]):
            total_counts[l] += nan + v
            nan_counts[l] += nan
        return nan_counts, total_counts

    def __replace_nans(self, data, value):
        """
        Replaces missing values by mean, median or a constant value outside of
        the feature interval.
        :param data: Dataframe
        :param mean: True/False
        :return: Dataframe
        """
        if not value:
            return data
        if value == 'const':
            return data.replace(to_replace=np.nan, value=-1000000)
        if value == 'mean':
            if self.means is None:
                self.means = data.mean()
            for col in data:
                data[col].replace(
                    to_replace=np.nan, value=self.means[col], inplace=True
                )
            return data.replace(to_replace=np.nan, value=-1000000)
        if value == 'median':
            if self.medians is None:
                self.medians = data.median()
            for col in data:
                data[col].replace(
                    to_replace=np.nan, value=self.medians[col], inplace=True
                )
            return data.replace(to_replace=np.nan, value=-1000000)

    def load_training_data(self, path):
        """
        Reads cisco specific gzipped files at given path and samples the
        negatives.
        :param path: Path to the gzipped files
        :return: A tuple where 0th element is a dataframe of features and
        1st element is a series of labels.
        """

        # Load and sample negative records first.
        files = glob.glob(os.path.join(path, 'neg', '*.gz'))
        data = self.__sample_negatives(files)

        # Read positive records and concatenate with sampled negatives.
        files = glob.glob(os.path.join(path, 'pos', '*.gz'))
        data = pd.concat(
            [data, pd.concat(
                pd.read_csv(f, sep='\t', header=None) for f in files
            )]
        )

        metadata = data[data.columns[:3]]
        labels = data[data.columns[3]]
        data = data[data.columns[4:]]
        data.rename(columns=lambda x: x-4, inplace=True)
        metadata.reset_index(drop=True, inplace=True)
        labels.reset_index(drop=True, inplace=True)
        data.reset_index(drop=True, inplace=True)

        data = self.__replace_nans(data, self.nan_value)

        return data, labels, metadata

    def load_testing_data(self, path, compute_nans=False):
        """
        Reads cisco specific gzipped files at given path without sampling
        the negatives, yielding the data one file at a time.
        :param path: Path to the gzipped files
        :return: A tuple where 0th element is a dataframe of features and
        1st element is a series of labels.
        """

        files = glob.glob(os.path.join(path, 'neg', '*.gz'))
        files += glob.glob(os.path.join(path, 'pos', '*.gz'))
        for f in files:
            data = pd.read_csv(f, sep='\t', header=None)

            metadata = data[data.columns[:3]]
            labels = data[data.columns[3]]
            data = data[data.columns[4:]]
            data.rename(columns=lambda x: x-4, inplace=True)

            counts = None
            if compute_nans:
                counts = self.__compute_nan_counts_per_class((data, labels))
            data = self.__replace_nans(data, self.nan_value)

            yield data, labels, metadata, counts

    def __sample_negatives(self, files):
        """
        Creates a random sample of negative records from given files.
        :param files: Negative gzipped files.
        :return: Dataframe of sampled negatives.
        """
        data = pd.DataFrame()
        samples_per_part = self.neg_samples // len(files)
        for file in files:
            unsampled = pd.read_csv(file, sep='\t', header=None)
            sampled_indices = self.__sample_indices(
                list(unsampled.index.values),
                samples_per_part
            )
            sampled = unsampled.iloc[sampled_indices]
            data = pd.concat([data, sampled])
        return data

    def __sample_indices(self, indices, samples_count):
        """
        Creates a random sample of the datasets indices
        :param indices: A list of indices.
        :return: A list of sampled indices.
        """
        return self.seed.choice(a=indices, size=samples_count, replace=False)

    def __compute_bins(self, dataset):
        """
        Computes bins for data quantization.
        :param dataset: Pandas dataframe with feature vectors
        :return: A list containing bins for each feature
        """
        bins = []

        for column in dataset.columns:
            sampled = dataset[column].sample(
                self.bin_samples,
                random_state=self.seed
            )

            if sampled.nunique() == 1:
                bins.append([sampled.iloc[0]])
            else:
                bins.append(
                    list(
                        pd.qcut(
                            x=sampled,
                            q=16,
                            retbins=True,
                            duplicates='drop'
                        )[1]
                    )
                )
        return bins

    def quantize_data(self, dataset):
        """
        Performs data quantization over all feature columns.
        :param dataset: A tuple with data (features, labels, metadata)
        Features are a pd.DataFrame, labels a series, metadata a pd.DataFrame
        :return: A tuple with quantized features (features, labels, metadata)
        """
        if not self.bins:
            self.bins = self.__compute_bins(dataset[0])

        quantized_frame = pd.DataFrame()

        for column in dataset[0].columns:
            digitized_list = np.digitize(dataset[0][column], self.bins[column])
            quantized_frame[column] = [
                self.bins[column][i] if i < len(self.bins[column])
                else self.bins[column][i-1] for i in digitized_list
            ]

        return quantized_frame, dataset[1], dataset[2]

    def load_classifications(self, file_path, delimiter, read_metadata=False):
        """
        Reads true/pred data from a file and saves it to dataframes.
        Optionally also reads metadata into a pandas dataframe.
        :param file_path: Path to the file
        :param delimiter: Symbol or a string by which the data is delimited.
        :param read_metadata: Whether to read metadata as well.
        """
        columns = ['true', 'pred']
        if read_metadata:
            metadata_columns = ['timestamp', 'host', 'user']
            columns.extend(metadata_columns)

        reader = pd.read_csv(
            file_path,
            delimiter,
            header=None,
            names=columns,
            chunksize=500000
        )

        for record in reader:
            trues = record['true']
            preds = record['pred']
            data_tuple = trues, preds
            if read_metadata:
                metadata = pd.DataFrame(
                    data=np.array(
                        [
                            record['timestamp'],
                            record['host'],
                            record['user']
                        ]).transpose(),
                    columns=metadata_columns
                )
                data_tuple = trues, preds, metadata
            yield data_tuple
