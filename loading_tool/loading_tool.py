import os
import pandas as pd
import numpy as np
import glob
import random

class LoadingTool():

    def __init__(self, sampling_settings, bins=None):

        self.bins = bins
        self.bin_count = sampling_settings['bin_count']
        self.bin_samples = sampling_settings['bin_samples']
        self.neg_samples = sampling_settings['neg_samples']
        self.seed = sampling_settings['seed']

    def load_training_data(self, path):
        """
        Reads cisco specific gzipped files at given path.
        :param path: Path to the gzipped files
        :return: A tuple where 0th element is a dataframe of features and
        1st element is a series of labels.
        """

        # Load and sample negative records first.
        files = glob.glob(os.path.join(path, 'neg', '*.gz'))
        data = self.sample_negatives(files)

        # Read positive records and concatenate with sampled negatives.
        files = glob.glob(os.path.join(path, 'pos', '*.gz'))
        data = pd.concat(
            [data, pd.concat(
                pd.read_csv(f, sep='\t', header=None) for f in files
            )]
        )

        # Replace nans
        data.replace(
            to_replace=np.nan,
            value=-1000000,
            inplace=True
        )

        metadata = data[data.columns[:3]]
        labels = data[data.columns[3]]
        data = data[data.columns[4:]]
        data.rename(columns=lambda x: x-4, inplace=True)
        metadata.reset_index(drop=True, inplace=True)
        labels.reset_index(drop=True, inplace=True)
        data.reset_index(drop=True, inplace=True)

        return data, labels, metadata

    def load_testing_data(self, path):
        """
        Reads cisco specific gzipped files at given path.
        :param path: Path to the gzipped files
        :return: A tuple where 0th element is a dataframe of features and
        1st element is a series of labels.
        """

        files = glob.glob(os.path.join(path, 'neg', '*.gz'))
        files += glob.glob(os.path.join(path, 'pos', '*.gz'))
        for f in files:
            data = pd.read_csv(f, sep='\t', header=None)

            # Replace nans
            data.replace(
                to_replace=np.nan,
                value=-1000000,
                inplace=True
            )

            metadata = data[data.columns[:3]]
            labels = data[data.columns[3]]
            data = data[data.columns[4:]]
            data.rename(columns=lambda x: x-4, inplace=True)

            yield data, labels, metadata

    def sample_negatives(self, files):
        """
        Creates a random sample of negative records from given files.
        :param files: Negative gzipped files.
        :return: Dataframe of sampled negatives.
        """
        data = pd.DataFrame()
        samples_per_part = self.neg_samples // len(files)
        for file in files:
            unsampled = pd.read_csv(file, sep='\t', header=None)
            sampled_indices = self.sample_indices(
                list(unsampled.index.values),
                samples_per_part
            )
            sampled = unsampled.iloc[sampled_indices]
            data = pd.concat([data, sampled])
        return data


    def sample_indices(self, indices, samples_count):
        """
        Creates a random sample of the datasets indices
        :param indices: A list of indices.
        :return: A list of sampled indices.
        """
        random.seed(self.seed)
        return random.sample(indices, samples_count)


    def compute_bins(self, dataset):
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
            self.bins = self.compute_bins(dataset[0])

        quantized_frame = pd.DataFrame()

        for column in dataset[0].columns:
            digitized_list = np.digitize(dataset[0][column], self.bins[column])
            quantized_frame[column] = [
                self.bins[column][i] if i < len(self.bins[column])
                else self.bins[column][i-1] for i in digitized_list
            ]

        return quantized_frame, dataset[1], dataset[2]

    def load_classifications(self, file_path, delimiter):
        """
        Reads true/pred data from a file and saves it to dataframes.
        Optionally reads also metadata into a pandas dataframe.
        :param file_path: Path to the file
        :param delimiter: Symbol or a string by which the data is delimited.
        :param metadata: Whether to read metadata as well.
        """
        columns = ['true', 'pred', 'timestamp', 'user', 'flow']
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
            metadata = pd.concat([record['timestamp'], record['user'], record['flow']])
            yield trues, preds, metadata
