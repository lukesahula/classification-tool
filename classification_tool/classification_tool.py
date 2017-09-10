from sklearn import datasets
import os
import glob
import pandas as pd
import numpy as np
import csv
import scipy

class ClassificationTool():

    def __init__(self):

        self.training_data = None
        self.testing_data = None
        self.classifier = None

    def load_dataset(self, path, zipped=False):

        if not zipped:
            data = datasets.load_svmlight_file(path)
            return (pd.DataFrame(data[0].toarray()), data[1])
        else:
            files = glob.glob(os.path.join(path, 'neg', '*.gz'))
            files += glob.glob(os.path.join(path, 'pos', '*.gz'))

            data = pd.concat(
                pd.read_csv(f, sep='\t', header=None) for f in files
            )

            data.replace(
                to_replace=np.nan,
                value=-1000000,
                inplace=True
            )

            return (data[data.columns[4:]], data[data.columns[3]])

    def train_classifier(self, classifier):
        self.classifier = classifier
        self.classifier.fit(self.training_data[0], self.training_data[1])
        self.training_data = None

    def save_predictions(self, output_file):

        def chunks(dataframe, n):
            """
            Yield successive n-sized chunks from dataframe.
            """
            for i in range(0, len(dataframe), n):
                yield dataframe[i:i + n]

        with open(output_file, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter=';')

            test_result = pd.DataFrame()
            for chunk in chunks(self.testing_data[0].index, 1000):
                test_data = self.testing_data[0].ix[chunk]

                test_result = pd.concat(
                    [test_result, pd.DataFrame(self.classifier.predict(test_data))])


            writer.writerows(zip(
                    self.testing_data[1],
                    test_result[0].values
                )
            )
