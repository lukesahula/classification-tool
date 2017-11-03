import os
import pandas as pd
import numpy as np
import csv

class ClassificationTool():

    def __init__(self, classifier):

        self.classifier = classifier

    def load_dataset(self, path, samples=None):
        # TODO: Extract to a separate loader.
        """
        Reads a csv file specified by path.
        :param path: Path to the csv file.
        :param samples: Number of neg samples, smelly engineering. TODO
        :return: A tuple where 0th element is a dataframe of features and
                 1st element is a series of labels. Specific for cisco data.
        """

        pos = os.path.join(path, 'pos', 'pos_test2')
        neg = os.path.join(path, 'neg_seed_01_samples_' + samples, 'sampled')

        data = pd.DataFrame()
        data = pd.concat((data, pd.read_csv(pos, sep='\t', header=None)))
        data = pd.concat((data, pd.read_csv(neg, sep='\t', header=None)))

        data.replace(
            to_replace=np.nan,
            value=-1000000,
            inplace=True
        )

        classes = data[data.columns[3]]
        records = data[data.columns[4:]]
        records.columns = list(range(len(records.columns)))

        return (records, classes)

    def compute_bins(self, dataset, bin_samples):
        """
        Computes bins for data quantization.
        :param dataset: Pandas dataframe with feature vectors
        :return: A list containing bins for each feature
        """
        bins = []

        for column in dataset.columns:
            sampled = dataset[column].sample(bin_samples, random_state=0)

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

    def quantize_data(self, dataset, bin_samples):
        """
        Performs data quantization over all feature columns.
        :param dataset: Pandas dataframe with feature vectors
        :return: Pandas dataframe with quantized feature vectors
        """
        bins = self.compute_bins(dataset, bin_samples)

        quantized_frame = pd.DataFrame()

        for column in dataset.columns:
            digitized_list = np.digitize(dataset[column], bins[column])
            quantized_frame[column] = [
                bins[column][i] if i < len(bins[column])
                else bins[column][i-1] for i in digitized_list
            ]

        return quantized_frame


    def train_classifier(self, tr_path, samples, bin_samples):
        """
        Trains the classifier.
        :param tr_path: Path to training data.
        :param samples: Number of neg samples, smelly engineering. TODO
        """
        self.tr_data = self.load_dataset(tr_path, samples)

        self.tr_data = (
            self.quantize_data(self.tr_data[0], bin_samples),
            self.tr_data[1]
        )

        self.classifier.fit(self.tr_data[0], self.tr_data[1])
        self.tr_data = None

    def save_predictions(self, t_path, samples, output_file, bin_samples):
        """
        Predicts labels on testing data and writes it to csv file.
        :param t_path: Path to testing data.
        :param samples: Number of neg samples, smelly engineering. TODO
        :param output_file: Path to output file.
        """
        def chunks(dataframe, n):
            """
            Yield successive n-sized chunks from dataframe.
            """
            for i in range(0, len(dataframe), n):
                yield dataframe[i:i + n]


        self.t_data = self.load_dataset(t_path, samples)

        self.t_data = (
            self.quantize_data(self.t_data[0], bin_samples),
            self.t_data[1]
        )

        with open(output_file, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter=';')

            preds = []
            for chunk in chunks(self.t_data[0].index, 1000):
                to_predict = self.t_data[0].ix[chunk]
                preds = preds + list(self.classifier.predict(to_predict))

            writer.writerows(zip(
                    self.t_data[1],
                    preds
                )
            )
