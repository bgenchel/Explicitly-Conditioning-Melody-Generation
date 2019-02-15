import argparse
import json
# import matplotlib.pyplot as plt
import os
import os.path as op
import sys
from pathlib import Path

sys.path.append(str(Path(op.abspath(__file__)).parents[1]))
from parsing.harmony import Harmony #noqa

def get_sorted_tuples(data_dict):
    data_tuples = [(item[0], item[1]) for item in data_dict.items()]
    data_tuples.sort(key=lambda tup: tup[1], reverse=True) 
    return data_tuples

def main(triad=False, simplified=False, histogram=False):
    root_dir = str(Path(op.abspath(__file__)).parents[3])
    xml_path = op.join(root_dir, 'data', 'raw', 'xml')
    analysis_path = op.join(root_dir, 'data', 'analysis')
    if not op.exists(xml_path):
        raise Exception("no raw data directory exists.")

    artists_dict = {}
    for fname in os.listdir(xml_path):
        artist_string = fname.split('.')[0].split('-')[1]
        artists = artist_string.split(',_')
        for artist in artists:
            if artist not in artists_dict:
                artists_dict[artist] = 0
            artists_dict[artist] += 1

    sorted_tuples = get_sorted_tuples(artists_dict)

    print('{} unique artists'.format(len(sorted_tuples)))
    print('top three most:')
    print('\t{}: {}'.format(sorted_tuples[0][0], sorted_tuples[0][1] / len(os.listdir(xml_path))))
    print('\t{}: {}'.format(sorted_tuples[1][0], sorted_tuples[1][1] / len(os.listdir(xml_path))))
    print('\t{}: {}'.format(sorted_tuples[2][0], sorted_tuples[2][1] / len(os.listdir(xml_path))))

    if histogram:
        keys = [tup[0] for tup in data_tuples]
        values = [tup[1] for tup in data_tuples]
        fig = plt.figure()
        fig.bar(keys, values, 1, color='g')
        fig.plot()
        fig.savefig(op.join(analysis_path, 'artists_dist.png'))

if __name__ == '__main__':
    main()
