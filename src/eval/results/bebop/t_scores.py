import os
import os.path as op
import glob
import json
from scipy.stats import ttest_ind
from collections import OrderedDict

DIR = "mgeval_results"

METRICS = ['avg_IOI', 'avg_pitch_shift', 'bar_used_note', 'bar_used_pitch', 'note_length_hist',
           'note_length_transition_matrix', 'pitch_class_transition_matrix', 'pitch_range', 'total_pitch_class_histogram',
           'total_used_note', 'total_used_pitch']

CONDITIONS = ['chord', 'nxtchord', 'barpos', 'inter']

if not op.exists('tscores'):
    os.makedirs('tscores')

for m in METRICS:
    for c in CONDITIONS:
        in_group = OrderedDict({'p2i_kl': [], 'p2i_ovl': [], 't2i_kl': [], 't2i_ovl': [], 'p2t_kl': [], 'p2t_ovl': []})
        out_group = OrderedDict({'p2i_kl': [], 'p2i_ovl': [], 't2i_kl': [], 't2i_ovl': [], 'p2t_kl': [], 'p2t_ovl': []})

        for model in os.listdir(DIR):
            # print('%s is in the %s for condition %s' % (model, ('in_group', 'out_group')[c not in
                # model.split('_')[:-1]], c))
            jsn = json.load(open(op.join(DIR, model, m + '.json'), 'r'))
            group = (in_group, out_group)[c not in model.split('_')[:-1]]
            for k, v in jsn.items():
                group[k].append(v)

        t_scores = OrderedDict({'p2i_kl': None, 'p2i_ovl': None, 't2i_kl': None, 't2i_ovl': None, 'p2t_kl': None, 'p2t_ovl': None})
        for key in in_group.keys():
            t_scores[key] = ttest_ind(in_group[key], out_group[key])
        json.dump(t_scores, open(op.join('tscores', '_'.join([m, c]) + '.json'), 'w'), indent=4)
