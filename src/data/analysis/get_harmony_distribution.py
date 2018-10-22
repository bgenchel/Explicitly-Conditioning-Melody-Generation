import argparse
import json
# import matplotlib.pyplot as plt
import os
import os.path as op
import sys
from pathlib import Path

sys.path.append(str(Path(op.abspath(__file__)).parents[1]))
from parsing.harmony import Harmony #noqa

def get_data(js_scores, triad=False, simplified=False):
    data = {}
    for js in js_scores:
        for measure in js['part']['measures']:
            for harmony_dict in measure["harmonies"]:
                harmony = Harmony(harmony_dict)
                if triad:
                    symbol = harmony.get_triad_chord_symbol()
                    harte = harmony.get_triad_harte_notation()
                    pitch_classes = harmony.get_triad_pitch_classes()
                    pitch_classes_binary = harmony.get_triad_pitch_classes_binary()
                elif simplified:
                    symbol = harmony.get_simple_chord_symbol()
                    harte = harmony.get_simple_harte_notation()
                    pitch_classes = harmony.get_simple_pitch_classes()
                    pitch_classes_binary = harmony.get_simple_pitch_classes_binary()
                else:
                    symbol = Harmony(harmony_dict).get_chord_symbol()
                    harte = harmony.get_harte_notation()
                    pitch_classes = harmony.get_pitch_classes()
                    pitch_classes_binary = harmony.get_pitch_classes_binary()


                if symbol not in data:
                    data[symbol] = {'count': 0,
                                    'harte': str(harte),
                                    'pitch_classes': str(pitch_classes),
                                    'pitch_classes_binary': str(pitch_classes_binary)}

                data[symbol]['count'] += 1

    data_tuples = [(item[0], item[1]['count']) for item in data.items()]
    data_tuples.sort(key=lambda tup: tup[1], reverse=True) 
    return data, data_tuples

def main(triad=False, simplified=False, histogram=False):
    root_dir = str(Path(op.abspath(__file__)).parents[3])
    json_path = op.join(root_dir, 'data', 'raw', 'json')
    analysis_path = op.join(root_dir, 'data', 'analysis')
    if not op.exists(json_path):
        raise Exception("no json directory exists.")

    fpaths = [op.join(json_path, fname) for fname in os.listdir(json_path)]
    js_scores = []
    for fpath in fpaths:
        js = json.load(open(fpath, 'r'))
        js_scores.append(js)

    if triad:
        counts_outfile = "triad_harmony_sorted_dist.txt"
        full_outfile = "triad_harmony_data.json"
    elif simplified:
        counts_outfile = "simple_harmony_sorted_dist.txt"
        full_outfile = "simple_harmony_data.json"
    else:
        counts_outfile = "harmony_sorted_dist.txt"
        full_outfile = "harmony_data.json"

    if op.exists(full_outfile):
        os.remove(full_outfile)

    data, data_tuples = get_data(js_scores, triad, simplified)
    print("%s unique chords found." % len(data.keys()))
    json.dump(data, open(op.join(analysis_path, full_outfile), "w"), indent=4)
    with open(op.join(analysis_path, counts_outfile), 'w') as fp:
        for tup in data_tuples:
            fp.write('%s:\t\t%i\n' % (tup[0], tup[1]))

    # if histogram:
    #     keys = [tup[0] for tup in data_tuples]
    #     values = [tup[1] for tup in data_tuples]
    #     fig = plt.figure()
    #     fig.bar(keys, values, 1, color='g')
    #     fig.plot()
    #     fig.savefig('harmony_hist.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--simplified', action='store_true',
                        help="get simplified chord representations.")
    parser.add_argument('-t', '--triad', action='store_true',
                        help="get triad chord representations. takes precedence over \
                        simplified representation.")
    parser.add_argument('-b', '--barplot', action='store_true',
                        help="make a histogram of chords and counts")
    args = parser.parse_args()
    main(args.triad, args.simplified, args.barplot)
