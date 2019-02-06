import argparse
import glob
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import pdb
import seaborn as sns
import sys
from pathlib import Path
from sklearn.model_selection import LeaveOneOut

sys.path.append("./mgeval")
from mgeval import core, utils

ABRV_TO_MODEL = {'nc': 'no_cond',
                 'ic': 'inter_cond',
                 'cc': 'chord_cond',
                 'bc': 'barpos_cond',
                 'cic': 'chord_inter_cond',
                 'cbc': 'chord_barpos_cond',
                 'ibc': 'inter_barpos_cond',
                 'cibc': 'chord_inter_barpos_cond'}

class MGEval:
    """
    Wrapper around Richard Yang's MGEval
    """
    def __init__(self, pred_dir, target_dir):
        self.pred_set = glob.glob(op.join(pred_dir, "4eval*_0", "4eval*_0_melody.mid"))
        self.target_set = glob.glob(op.join(target_dir, "*.mid"))

        pred_samples = len(self.pred_set)
        target_samples = len(self.target_set)

        if pred_samples > target_samples:
            self.num_samples = target_samples
        else:
            self.num_samples = pred_samples

        self.num_samples = 100
        self.metrics = core.metrics()

    def get_metric(self, metric_name, pred_metric_shape, target_metric_shape, *args, **kwargs):
        pred_metric = np.zeros((self.num_samples,) + pred_metric_shape)
        target_metric = np.zeros((self.num_samples,) + target_metric_shape)

        for sample in range(self.num_samples):
            pred_metric[sample] = getattr(self.metrics, metric_name)(core.extract_feature(self.pred_set[sample]), *args, **kwargs)
            target_metric[sample] = getattr(self.metrics, metric_name)(core.extract_feature(self.target_set[sample]), *args, **kwargs)

        return pred_metric, target_metric

    def inter_set_cross_validation(self, pred_metric, target_metric):
        loo = LeaveOneOut()
        loo.get_n_splits(np.arange(self.num_samples))

        inter_set_distance_cv = np.zeros((self.num_samples, 1, self.num_samples))

        for train_index, test_index in loo.split(np.arange(self.num_samples)):
            inter_set_distance_cv[test_index[0]][0] = utils.c_dist(pred_metric[test_index], target_metric)

        return inter_set_distance_cv

    def intra_set_cross_validation(self, pred_metric, target_metric):
        loo = LeaveOneOut()
        loo.get_n_splits(np.arange(self.num_samples))

        pred_intra_set_distance_cv = np.zeros((self.num_samples, 1, self.num_samples - 1))
        target_intra_set_distance_cv = np.zeros((self.num_samples, 1, self.num_samples - 1))

        for train_index, test_index in loo.split(np.arange(self.num_samples)):
            pred_intra_set_distance_cv[test_index[0]][0] = utils.c_dist(pred_metric[test_index],
                                                                        pred_metric[train_index])
            target_intra_set_distance_cv[test_index[0]][0] = utils.c_dist(target_metric[test_index],
                                                                          target_metric[train_index])

        return pred_intra_set_distance_cv, target_intra_set_distance_cv

    def visualize(self, metric_name, pred_intra, target_intra, inter):
        for measurement, label in zip([pred_intra, target_intra, inter], ["pred_intra", "target_intra", "inter"]):
            transposed = np.transpose(measurement, (1, 0, 2)).reshape(1, -1)
            sns.kdeplot(transposed[0], label=label)

        plt.title(metric_name)
        plt.xlabel('Euclidean distance')
        plt.savefig(metric_name + '.png')
        # plt.show()

    def intra_inter_difference(self, metric_name, pred_intra, target_intra, inter):
        transposed = []

        for measurement, label in zip([pred_intra, target_intra, inter], ["pred_intra", "target_intra", "inter"]):
            transposed_meas = np.transpose(measurement, (1, 0, 2)).reshape(1, -1)
            transposed.append(transposed_meas)

        print(metric_name + ':')
        print('------------------------')
        print(' Predictions')
        print('  KL divergence:', utils.kl_dist(transposed[0][0], transposed[2][0]))
        print('  Overlap area:', utils.overlap_area(transposed[0][0], transposed[2][0]))
        print(' Targets')
        print('  KL divergence:', utils.kl_dist(transposed[1][0], transposed[2][0]))
        print('  Overlap area:', utils.overlap_area(transposed[1][0], transposed[2][0]))


def main(model, metric):
    root_dir = str(Path(op.abspath(__file__)).parents[2])
    mge = MGEval(pred_dir=op.join(root_dir, "src", "models", ABRV_TO_MODEL[model], "midi"),
                 target_dir=op.join(root_dir, "data", "raw", "midi"))

    # Expected shape of desired metric
    metric_shape = (12, 12)

    # Metric name
    metric_name = "note_length_transition_matrix"

    # Args and kwargs if needed for the desired metric
    args = ()
    kwargs = {"track_num": 1}

    pred_metric, target_metric = mge.get_metric(metric_name, metric_shape, metric_shape, *args, **kwargs)
    inter = mge.inter_set_cross_validation(pred_metric, target_metric)
    pred_intra, target_intra = mge.intra_set_cross_validation(pred_metric, target_metric)
    mge.visualize(metric_name, pred_intra, target_intra, inter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default="nc", type=str,
                    choices=("nc", "ic", "cc", "bc", "cic", "cbc", "ibc", "cibc"),
                    help="which model to evaluate.")
    parser.add_argument('-mt', '--metric', default="all", type=str,
                    choices=("total_used_pitch", "bar_used_pitch", "total_used_note", 
                             "bar_used_note", "total_pitch_class_histogram", "bar_pitch_class_histogram",
                             "pitch_class_transition_matrix", "pitch_range", "avg_pitch_shift",
                             "avg_IOI", "note_length_hist", "note_length_transition_matrix", "all"),
                    help="which model to evaluate.")
    args = parser.parse_args()
    main(args.model, args.metric)
