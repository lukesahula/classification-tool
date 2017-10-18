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

        pos = os.path.join(path, 'pos', 'pos_test')
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
        records.columns = list(range(51))

        return (records, classes)

    def compute_bins(self, column):

        sampled = column.sample(int(len(column)/2), random_state=50000)

        if sampled.nunique() == 1:
            return [sampled.iloc[0]]
        return list(
            pd.qcut(x=sampled, q=16, retbins=True, duplicates='drop')[1]
        )

    def quantize_data(self, column):
        quantized = []
        for value in column:
            for index in range(len(self.bins[column.name])):
                if value <= self.bins[column.name][index]:
                    quantized.append(self.bins[column.name][index])
                    break
                if index == len(self.bins[column.name]) - 1:
                    quantized.append(self.bins[column.name][-1])
        return quantized


    def train_classifier(self, tr_path, samples):
        """
        Trains the classifier.
        :param tr_path: Path to training data.
        :param samples: Number of neg samples, smelly engineering. TODO
        """
        self.tr_data = self.load_dataset(tr_path, samples)

        self.bins = [
            self.compute_bins(
                self.tr_data[0][column]
            ) for column in self.tr_data[0].columns
        ]

        quantized_frame = pd.DataFrame()

        for column in self.tr_data[0].columns:
            quantized_frame[column] = self.quantize_data(self.tr_data[0][column])

        self.tr_data = (
            quantized_frame,
            self.tr_data[1]
        )

        quantized_frame = None

        self.classifier.fit(self.tr_data[0], self.tr_data[1])
        self.tr_data = None

    def save_predictions(self, t_path, samples, output_file):
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

        quantized_frame = pd.DataFrame()

        for column in self.t_data[0].columns:
            quantized_frame[column] = self.quantize_data(self.t_data[0][column])

        self.t_data = (
            quantized_frame,
            self.t_data[1]
        )

        quantized_frame = None

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
