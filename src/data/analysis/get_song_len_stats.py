import argparse
import json
# import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as op
import sys
from pathlib import Path

sys.path.append(str(Path(op.abspath(__file__)).parents[1]))
from parsing.harmony import Harmony #noqa

MAJOR_CIRCLE = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'Db', 'Ab', 'Eb', 'Bb', 'F']
MINOR_CIRCLE = ['A', 'D', 'G', 'C', 'F', 'Bb', 'Eb', 'G#', 'C#', 'F#', 'B', 'E']

def get_sorted_tuples(data_dict):
    data_tuples = [(item[0], item[1]['count']) for item in data_dict.items()]
    data_tuples.sort(key=lambda tup: tup[1], reverse=True) 
    return data_tuples

def main(triad=False, simplified=False, histogram=False):
    root_dir = str(Path(op.abspath(__file__)).parents[3])
    json_path = op.join(root_dir, 'data', 'interim')
    analysis_path = op.join(root_dir, 'data', 'analysis')
    if not op.exists(json_path):
        raise Exception("no json directory exists.")

    fpaths = [op.join(json_path, fname) for fname in os.listdir(json_path)]
    js_scores = []
    for fpath in fpaths:
        js = json.load(open(fpath, 'r'))
        js_scores.append(js)

    song_lens = []
    notes_per_measure = []
    for js in js_scores:
        song_lens.append(len(js['part']['measures']))
        for m in js['part']['measures']:
            notes_per_measure.append(sum([len(g['notes']) for g in m['groups']]))

    print('songs: ')
    print('max len - {}, min_len - {}, avg_len - {}'.format(max(song_lens), min(song_lens), np.mean(song_lens)))
    print('notes_per_measure: ')
    print('max len - {}, min_len - {}, avg_len - {}'.format(max(notes_per_measure), min(notes_per_measure), np.mean(notes_per_measure)))


if __name__ == '__main__':
    main()
