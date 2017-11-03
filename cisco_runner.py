from utils.utils import tee
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

        eval_output = 'evaluation_tool/outputs/rfc.cisco'

        n_estimators = 10
        max_features = 10
        min_samples_split = 1000
        criterion = 'entropy'
        n_jobs = -1
        samples = '100000'
        bin_samples = 50000
        random_state = 0

        with open(eval_output, 'w', encoding='utf-8') as f:
            tee('Running cisco runner with the following configuration:\n', f)
            tee('Classifier: {}'.format(str(RFC)), f)
            tee('Number of trees in the forest: {}'
                .format(str(n_estimators)), f)
            tee('Max number of features for a split: {}'
                .format(str(max_features)), f)
            tee('Min number of samples for a split: {}'
                .format(str(min_samples_split)), f)
            tee('Criterion: {}'.format(criterion), f)
            tee('Number of jobs: {}'.format(str(n_jobs)), f)
            tee('Number of negative samples: {}'.format(samples), f)
            tee('Number of samples for quantization: {}'
                .format(str(bin_samples)), f)
            tee('Seed: {}\n'.format(random_state), f)


        rfc = RFC(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_split=min_samples_split,
            criterion=criterion,
            n_jobs=n_jobs,
            random_state=random_state
        )
        clas_tool = ClassificationTool(rfc)
        clas_tool.train_classifier(cisco_training, samples, bin_samples)
        predictions_output = 'classification_tool/outputs/rfc.cisco'
        clas_tool.save_predictions(
            cisco_testing,
            samples,
            predictions_output,
            bin_samples
        )

        eval_tool = EvaluationTool(predictions_output, ';')
        with open(eval_output, 'a', encoding='utf-8') as f:
            tee('Average precision: {}'
                .format(eval_tool.get_avg_precision()), f)
            tee('Average recall: {}'.format(eval_tool.get_avg_recall()), f)
            tee('\nPrecisions per label:', f)

            for label in eval_tool.labels:
                tee('Label: %.1f, precision: %f'
                    %(label, eval_tool.compute_precision(label)), f)

            tee('\nRecalls per label:', f)
            for label in eval_tool.labels:
                tee('Label: %.1f, recall: %f'
                    %(label, eval_tool.compute_recall(label)), f)

runner = CiscoRunner()
runner.execute_run()
