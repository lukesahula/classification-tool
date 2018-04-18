import os

import datetime

from utils.utils import tee
from loading_tool.loading_tool import LoadingTool
from classification_tool.classification_tool import ClassificationTool
from evaluation_tool.evaluation_tool import EvaluationTool
from classifiers.random_forest import RandomForest as RF
from classifiers.decision_tree import DecisionTree as DT
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.externals import joblib
from collections import defaultdict
from joblib import Parallel
import numpy as np
import pandas as pd


class SerializableClassifier():
    def __init__(self, classifier, bins=None):
        self.classifier = classifier
        self.bins = bins


class CiscoRunner():
    def __write_settings(self, output, runner_settings, sampling_settings):
        with open(output, 'w', encoding='utf-8') as f:
            tee('Running cisco runner with the following configuration:\n', f)
            tee('Classifier: {}'.format(str(runner_settings['classifier'])), f)
            tee('Number of trees in the forest: {}'
                .format(str(runner_settings['n_estimators'])), f)
            tee('Max number of features for a split: {}'
                .format(str(runner_settings['max_features'])), f)
            tee('Min number of samples for a split: {}'
                .format(str(runner_settings['min_samples_split'])), f)
            tee('Criterion: {}'.format(runner_settings['criterion']), f)
            tee('Number of jobs: {}'.format(str(runner_settings['n_jobs'])), f)
            tee('Number of negative samples: {}'
                .format(str(sampling_settings['neg_samples'])), f)
            tee('Number of samples for quantization: {}'
                .format(str(sampling_settings['bin_samples'])), f)
            tee('Bin count: {}'
                .format(str(sampling_settings['bin_count'])), f)
            tee('Seed: {}'.format(runner_settings['random_state']), f)
            tee('Aggregated by: {}'.format(runner_settings['agg_by']), f)
            tee('Relaxed: {}'.format(runner_settings['relaxed']), f)
            tee('NaN value: {}\n'.format(runner_settings['nan_value']), f)

    def __compute_nan_ratios(self, path, loading_tool):
        nan_counts_total = defaultdict(int)
        class_counts_total = defaultdict(int)
        for t_data in loading_tool.load_testing_data(t_path):
            nan_counts, class_counts = t_data[3]
            for k in nan_counts.keys():
                nan_counts_total[k] += nan_counts[k]
                class_counts_total[k] += class_counts[k]

        nan_ratios = dict(
            (n, nan_counts_total[n] / class_counts_total[n])
            for n in set(nan_counts_total) | set(class_counts_total)
        )
        return nan_ratios

    def __write(self, output, text):
        with open(output, 'a', encoding='utf-8') as f:
            tee(text, f)

    def __write_stats(self, output, eval_tool, stats, nan_ratios=None):
        def write_stats_above_prec_threshold(threshold, eval_tool, stats, f):
            labels = eval_tool.get_labels_with_prec_above_thres(
                thres=threshold,
                labels=eval_tool.labels,
                stats=stats
            )
            avg_recall = eval_tool.get_avg_recall(stats, par_labels=labels)
            stats_counts = eval_tool.get_stats_counts(labels, stats)
            tps = [stats[l]['TP'] for l in labels]
            median_tps = np.median(tps)
            sum_tps = stats_counts['TP']
            fps = [stats[l]['FP'] for l in labels]
            median_fps = np.median(fps)
            sum_fps = stats_counts['FP']
            tee('Number of classes with precision >= %1.2f: %.0f, '
                'Avg. recall: %3.3f, Median TPs: %.0f, Sum TPs: %.0f, '
                'Median FPs: %.0f, Sum FPs: %.0f'
                % (threshold, len(labels), avg_recall, median_tps,
                    sum_tps, median_fps, sum_fps),
                f)

        counts = eval_tool.get_stats_counts(eval_tool.labels, stats)

        with open(output, 'a', encoding='utf-8') as f:
            tee('Avg Stats:\n', f)
            tee('Average precision w/o legit & NaN: {}'
                .format(eval_tool.get_avg_precision(stats, False, False)), f)
            tee('Average precision w/o legit: {}'
                .format(eval_tool.get_avg_precision(stats, False, True)), f)
            tee('Average recall w/o legit: {}'
                .format(eval_tool.get_avg_recall(stats, False, True)), f)
            tee('\nAverage precision w/o NaN: {}'
                .format(eval_tool.get_avg_precision(stats, True, False)), f)
            tee('Average overall precision: {}'
                .format(eval_tool.get_avg_precision(stats)), f)
            tee('Average overall recall: {}'
                .format(eval_tool.get_avg_recall(stats)), f)
            tee('\nOverall stats:\n', f)
            tee('TPS: {}'.format(counts['TP']), f)
            tee('FPS: {}'.format(counts['FP']), f)
            tee('FNS: {}\n'.format(counts['FN']), f)

            write_stats_above_prec_threshold(1.00, eval_tool, stats, f)
            write_stats_above_prec_threshold(0.95, eval_tool, stats, f)
            write_stats_above_prec_threshold(0.90, eval_tool, stats, f)
            write_stats_above_prec_threshold(0.80, eval_tool, stats, f)
            write_stats_above_prec_threshold(0.50, eval_tool, stats, f)

            tee('Individual stats:\n', f)
            if nan_ratios:
                tee('label\tprecis\trecall\ttps\tfps\tfns\tnan_ratio', f)
                for label in eval_tool.labels:
                    counts = eval_tool.get_stats_counts(label, stats)
                    tee('%3.0f\t%4.3f\t%4.3f %6.0f %6.0f %6.0f\t%1.3f'
                        % (label, eval_tool.compute_precision(label, stats),
                            eval_tool.compute_recall(label, stats),
                            counts['TP'], counts['FP'], counts['FN'],
                            nan_ratios.get(label, 0)),
                        f)
            else:
                tee('label\tprecis\trecall\ttps\tfps\tfns', f)
                for label in eval_tool.labels:
                    counts = eval_tool.get_stats_counts(label, stats)
                    tee('%3.0f\t%4.3f\t%4.3f %6.0f %6.0f %6.0f'
                        % (label, eval_tool.compute_precision(label, stats),
                            eval_tool.compute_recall(label, stats),
                            counts['TP'], counts['FP'], counts['FN']),
                        f)

    def execute_run(
        self, par_classifier=None, dump=True, output_dir=None, nan_ratio=False,
        tr_path='classification_tool/datasets/cisco_datasets/data/test_tr',
        t_path='classification_tool/datasets/cisco_datasets/data/test_t',
        classifier=RFC, n_estimators=100, max_features='sqrt',
        min_samples_split=2, criterion='entropy', n_jobs=-1,
        random_state=0, nan_value=None, relaxed=False, agg_by=None, method=None
    ):

        clsfr_output = os.path.join(output_dir, 'clsfr')

        eval_output = os.path.join(output_dir, 'eval')
        predictions_output = os.path.join(output_dir, 'clas')
        os.makedirs(output_dir)

        # Configuration for the runner
        runner_settings = {
            'classifier': classifier,
            'n_estimators': n_estimators,
            'max_features': max_features,
            'min_samples_split': min_samples_split,
            'criterion': criterion,
            'n_jobs': n_jobs,
            'random_state': random_state,
            'nan_value': nan_value,
            'relaxed': relaxed,
            'agg_by': agg_by,
        }

        # Configutarion for data preprocessing
        sampling_settings = {
            'bin_count': 16,
            'neg_samples': 50000,
            'bin_samples': 1000,
            'seed': random_state,
            'nan_value': nan_value
        }

        self.__write_settings(eval_output, runner_settings, sampling_settings)

        if not par_classifier:
            if classifier == RFC:
                clsfr = RFC(
                    n_estimators=n_estimators,
                    max_features=max_features,
                    min_samples_split=min_samples_split,
                    criterion=criterion,
                    n_jobs=n_jobs,
                    random_state=random_state
                )
            elif classifier == DT:
                clsfr = DT(
                    max_features=max_features,
                    min_samples_split=min_samples_split,
                    random_state=random_state,
                    method=method
                )
            else:
                clsfr = RF(
                    max_features=max_features,
                    min_samples_split=min_samples_split,
                    n_estimators=n_estimators,
                    n_jobs=n_jobs,
                    random_state=random_state,
                    method=method
                )

            loading_tool = LoadingTool(sampling_settings)
            clas_tool = ClassificationTool(clsfr)
            self.__write(eval_output, 'Loading training data')
            tr_data = loading_tool.load_training_data(tr_path)
            self.__write(eval_output, 'Initiate quantization')
            tr_data = loading_tool.quantize_data(tr_data)

            if method == 'mia':
                tr_data = (
                    tr_data[0].replace(to_replace=np.nan, value=-1000000), tr_data[1], tr_data[2]
                )

            self.__write(eval_output, 'Initiate growing')
            clas_tool.train_classifier(tr_data)
            tr_data = None
        elif type(par_classifier) == SerializableClassifier:
            loading_tool = LoadingTool(sampling_settings, par_classifier.bins)
            clas_tool = ClassificationTool(par_classifier.classifier)
        else:
            par_classifier = joblib.load(par_classifier)
            loading_tool = LoadingTool(sampling_settings, par_classifier.bins)
            clas_tool = ClassificationTool(par_classifier.classifier)

        if nan_ratio:
            nan_ratios = self.__compute_nan_ratios(t_path, loading_tool)

        ser_classifier = SerializableClassifier(
            clas_tool.classifier,
            loading_tool.bins
        )

        if dump:
            self.__write(eval_output, 'Dumping classifier')
            joblib.dump(ser_classifier, clsfr_output, compress=3)

        self.__write(eval_output, 'Predicting')
        with Parallel(n_jobs=n_jobs) as parallel:
            for t_data in loading_tool.load_testing_data(t_path):
                t_data = loading_tool.quantize_data(t_data)
                clas_tool.save_predictions(
                    t_data,
                    predictions_output,
                    parallel,
                )
        t_data = None

        eval_tool = EvaluationTool(legit=0)

        if agg_by:
            stats = defaultdict(lambda: defaultdict(set))
        else:
            stats = defaultdict(lambda: defaultdict(int))

        self.__write(eval_output, 'Begin evaluation')
        for chunk in loading_tool.load_classifications(
                predictions_output, ';', True):
            if agg_by:
                chunk_stats = eval_tool.compute_stats_for_agg(
                    agg_by, chunk, relaxed)
                for k, v in chunk_stats.items():
                    stats[k]['FP'] = stats[k]['FP'] | v['FP']
                    stats[k]['FN'] = stats[k]['FN'] | v['FN']
                    stats[k]['TP'] = stats[k]['TP'] | v['TP']
            else:
                chunk_stats = eval_tool.compute_stats(chunk)
                for label in chunk_stats:
                    stats[label]['FP'] += chunk_stats[label]['FP']
                    stats[label]['FN'] += chunk_stats[label]['FN']
                    stats[label]['TP'] += chunk_stats[label]['TP']
        if agg_by:
            stats = eval_tool.aggregate_stats(stats)

        self.__write_stats(eval_output, eval_tool, stats)

        return ser_classifier

    def get_corellation_matrix(self, path, output_dir):
        os.makedirs(output_dir)
        eval_tool = EvaluationTool()
        sampling_settings = {
            'bin_count': 16,
            'neg_samples': 10000,
            'bin_samples': 1000,
            'seed': 0,
            'nan_value': None,
        }
        loading_tool = LoadingTool(sampling_settings)
        data = loading_tool.load_training_data(path)[0]
        corr_matrix = data.corr().fillna(0)
        corr_path = os.path.join(output_dir, 'corr')
        corr_matrix.to_csv(corr_path, sep='\t', encoding='utf-8')
        norm = np.linalg.norm(corr_matrix)
        norm_path = os.path.join(output_dir, 'norm')
        with open(norm_path, 'w') as f:
            f.write(str(norm))

    def get_missingness_correlation_matrix(self, path, output_dir):
        os.makedirs(output_dir)
        eval_tool = EvaluationTool()
        sampling_settings = {
            'bin_count': 16,
            'neg_samples': 10000,
            'bin_samples': 1000,
            'seed': 0,
            'nan_value': None,
        }
        loading_tool = LoadingTool(sampling_settings)
        data = loading_tool.load_training_data(path)[0]
        data[~pd.isnull(data)] = 1
        data[pd.isnull(data)] = 0
        corr_matrix = data.corr().fillna(0)
        corr_path = os.path.join(output_dir, 'corr_missingness')
        corr_matrix.to_csv(corr_path, sep='\t', encoding='utf-8')
        norm = np.linalg.norm(corr_matrix)
        norm_path = os.path.join(output_dir, 'norm_missingness')
        with open(norm_path, 'w') as f:
            f.write(str(norm))


runner = CiscoRunner()

out_dir = datetime.datetime.now().isoformat()
out_corr = os.path.join('corr_outputs', out_dir)
out_dir_unagg = os.path.join('runner_outputs', out_dir, 'unaggregated')
out_dir_agg_by_u = os.path.join('runner_outputs', out_dir, 'agg_by_user')
out_dir_agg_by_u_r = os.path.join('runner_outputs', out_dir, 'agg_by_user_rel')
out_dir_otfi = os.path.join('runner_outputs', out_dir, 'otfi')
out_dir_mia = os.path.join('runner_outputs', out_dir, 'mia')
clsfr_path = os.path.join('runner_outputs', 'custom', 'unaggregated', 'clsfr')


# CORR
# runner.get_correlation_matrix(
#    'classification_tool/datasets/cisco_datasets/data/test_tr', out_corr
# )
# CORR MISINGNESS
# runner.get_missingness_correlation_matrix(
#    'classification_tool/datasets/cisco_datasets/data/test_tr', out_corr
# )

# OTFI
runner.execute_run(
    classifier=RF, agg_by=None, relaxed=False, dump=False, output_dir=out_dir_otfi,
    nan_value=None, n_estimators=20, method='otfi'
)


# MIA
# runner.execute_run(
#    classifier=RF, agg_by=None, relaxed=False, dump=True, output_dir=out_dir_mia,
#    nan_value=None, n_estimators=20, method='mia'
# )

# UNAG RFC
clsfr = runner.execute_run(
    classifier=RF, agg_by=None, relaxed=False,
    dump=False, output_dir=out_dir_unagg, nan_value=-1000000,
    n_estimators=20
)

# UNAG RFC_scikit
# clsfr = runner.execute_run(
#     classifier=RFC, agg_by=None, relaxed=False,
#     dump=True, output_dir=out_dir_unagg, nan_value=-1000000,
#     n_estimators=100
# )

# runner.execute_run(
#      par_classifier=clsfr, agg_by='user', relaxed=False,
#      dump=False, output_dir=out_dir_agg_by_u, nan_value='mean'
# )
# runner.execute_run(
#      par_classifier=clsfr, agg_by='user', relaxed=True,
#      dump=True, output_dir=out_dir_agg_by_u_r, nan_value='median'
# )
