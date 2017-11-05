import os
import pandas as pd
import numpy as np
import glob
import random

class LoadingTool():

    def __init__(self, sampling_settings):

        self.bins = None
        self.bin_samples = sampling_settings['bin_samples']
        self.neg_samples = sampling_settings['neg_samples']
        self.seed = sampling_settings['seed']

    def load_cisco_dataset(self, path):
        """
        Reads cisco specific gzipped files at given path.
        :param path: Path to the gzipped files
        :return: A tuple where 0th element is a dataframe of features and
        1st element is a series of labels.
        """

        # Load negative records first.
        files = glob.glob(os.path.join(path, 'neg', '*.gz'))
        data = pd.concat(pd.read_csv(f, sep='\t', header=None) for f in files)

        # Sample negative records.
        sampled_indices = self.sample_indices(list(data.index.values))
        data = data.iloc[sampled_indices]

        # Read positive records and concatenate with sampled negatives.
        files = glob.glob(os.path.join(path, 'pos', '*.gz'))
        data = pd.concat([data, pd.concat(pd.read_csv(f, sep='\t', header=None) for f in files)])

        # Replace nans
        data.replace(
            to_replace=np.nan,
            value=-1000000,
            inplace=True
        )

        labels = data[data.columns[3]]
        data = data[data.columns[4:]]
        data.rename(columns=lambda x: x-4, inplace=True)
        labels.reset_index(drop=True, inplace=True)
        data.reset_index(drop=True, inplace=True)

        return (data, labels)

    def sample_indices(self, indices):
        """
        Creates a random sample of the datasets indices
        :param indices: A list of indices.
        :return: A list of sampled indices.
        """
        random.seed(self.seed)
        return random.sample(indices, self.neg_samples)


    def compute_bins(self, dataset):
        """
        Computes bins for data quantization.
        :param dataset: Pandas dataframe with feature vectors
        :return: A list containing bins for each feature
        """
        bins = []

        for column in dataset.columns:
            sampled = dataset[column].sample(self.bin_samples, random_state=0)

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
        :param dataset: Pandas dataframe with feature vectors
        :return: Pandas dataframe with quantized feature vectors
        """
        if not self.bins:
            self.bins = self.compute_bins(dataset)

        quantized_frame = pd.DataFrame()

        for column in dataset.columns:
            digitized_list = np.digitize(dataset[column], self.bins[column])
            quantized_frame[column] = [
                self.bins[column][i] if i < len(self.bins[column])
                else self.bins[column][i-1] for i in digitized_list
            ]

        return quantized_frame

def load_classifications(file_path, delimiter):
    """
    Reads true/pred data from a file and saves it to dataframes.
    :param file_path: Path to the file
    :param delimiter: Symbol or a string by which the data is delimited.
    """
    columns = ['true', 'pred', 'key']
    df = pd.read_csv(file_path, delimiter, header=None, names=columns)
    trues = df['true']
    preds = df['pred']
    keys = []
    if not df['key'].isnull().any():
        keys = df['key']

    return (trues, preds, keys)
