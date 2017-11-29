import csv

class ClassificationTool():

    def __init__(self, classifier, loading_tool):

        self.classifier = classifier
        self.loading_tool = loading_tool

    def train_classifier(self, tr_path):
        """
        Trains the classifier.
        :param tr_path: Path to training data.
        """
        self.tr_data = self.loading_tool.load_cisco_dataset(tr_path)

        self.tr_data = (
            self.loading_tool.quantize_data(self.tr_data[0]),
            self.tr_data[1]
        )

        self.classifier.fit(self.tr_data[0], self.tr_data[1])
        self.tr_data = None

    def save_predictions(self, t_path, output_file):
        """
        Predicts labels on testing data and writes it to csv file.
        :param t_path: Path to testing data.
        :param output_file: Path to output file.
        """
        def chunks(dataframe, n):
            """
            Yield successive n-sized chunks from dataframe.
            """
            for i in range(0, len(dataframe), n):
                yield dataframe[i:i + n]


        self.t_data = self.loading_tool.load_cisco_dataset(t_path)

        self.t_data = (
            self.loading_tool.quantize_data(self.t_data[0]),
            self.t_data[1],
            self.t_data[2]
        )

        with open(output_file, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter=';')

            preds = []
            for chunk in chunks(self.t_data[0].index, 1000):
                to_predict = self.t_data[0].ix[chunk]
                preds = preds + list(self.classifier.predict(to_predict))

            writer.writerows(
                zip(
                    self.t_data[1],
                    preds,
                    self.t_data[2][0],
                    self.t_data[2][1],
                    self.t_data[2][2]
                )
            )

        self.t_data = None
