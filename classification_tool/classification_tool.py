from sklearn import datasets


class ClassificationTool():

    def __init__(self, training_file, testing_file):

        self.training_data = None
        self.testing_data = None
        self.classifier = None

        self.__load_datasets(training_file, testing_file)

    def __load_datasets(self, training_file, testing_file):
        self.training_data = datasets.load_svmlight_file(training_file)
        self.testing_data = datasets.load_svmlight_file(testing_file)

    def train_classifier(self, classifier):
        self.classifier = classifier
        self.classifier.fit(self.training_data[0], self.training_data[1])

    def save_predictions(self, output_file):
        with open(output_file, 'w', encoding='utf-8') as file:
            for true, pred in zip(
                    self.testing_data[1],
                    self.classifier.predict(self.testing_data[0])
            ):
                file.write(str(true) + ';' + str(pred) + '\n')
