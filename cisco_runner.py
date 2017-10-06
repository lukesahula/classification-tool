from classification_tool.classification_tool import ClassificationTool
from evaluation_tool.evaluation_tool import EvaluationTool
from sklearn.ensemble import RandomForestClassifier as RFC


class CiscoRunner():

    def execute_run(self):
        cisco_training = (
            'classification_tool/datasets/cisco_datasets/data/20170104'
        )
        cisco_testing = (
            'classification_tool/datasets/cisco_datasets/data/20170111'
        )

        rfc = RFC(
            n_estimators=10,
            max_features=10,
            min_samples_split=1000,
            criterion="entropy",
            n_jobs=-1
        )
        clas_tool = ClassificationTool()
        clas_tool.training_data = clas_tool.load_dataset(cisco_training, True)
        clas_tool.train_classifier(rfc)
        clas_tool.testing_data = clas_tool.load_dataset(cisco_testing, True)

        predictions_output = 'classification_tool/outputs/rfc.cisco'

        clas_tool.save_predictions(predictions_output)

        eval_tool = EvaluationTool(predictions_output, ';')
        print("Average precision: %f" %(eval_tool.get_avg_precision()))
        print("Average recall: %f" %(eval_tool.get_avg_recall()))

        print("\nPrecisions per label:")
        for label in eval_tool.labels:
            print("Label: %.1f, precision: %f"
                  %(label, eval_tool.compute_precision(label)))

        print("\nRecalls per label:")
        for label in eval_tool.labels:
            print("Label: %.1f, recall: %f"
                  %(label, eval_tool.compute_recall(label)))

runner = CiscoRunner()
runner.execute_run()
