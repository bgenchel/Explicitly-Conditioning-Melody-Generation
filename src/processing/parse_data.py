from __future__ import print_function
import json
import numpy as np
import os
import os.path as op
from pathlib import Path

NOTES_DICT = {'B#': 1, 'C': 1, 'C#': 2, 'Db': 2, 'D': 3, 'D#': 4, 'Eb': 4, 'E': 5, 
              'Fb': 5, 'E#': 6, 'F': 6, 'F#': 7, 'Gb': 7, 'G': 8, 'G#': 9, 'Ab': 9, 
              'A': 10, 'A#': 11, 'Bb': 11, 'B': 12, 'Cb': 12, 'rest': 0}

DURATION_DICT = {'whole': 0, 'half': 1, 'quarter': 2, 'eighth': 3, 'sixteenth': 4, 
                 'whole-trip': 5, 'half-trip': 6, 'quar-trip': 7, 'eigh-trip': 8, 
                 'six-trip': 9, 'whole-dot': 10, 'half-dot': 11, 'quar-dot': 12, 
                 'eigh-dot': 13, 'six-dot': 14, 'thirtysec': 15, 
                 'thirtysec-trip': 16, 'thirtysec-dot': 17}

# MAJOR_KEY_DICT = {'0': 1, '1': 8, '2': 3, '3': 10, '4': 5, '5': 12, '6': 7, '-1': 6, 
#                   '-2': 11, '-3': 4, '-4': 9, '-5': 2, '-6': 7 }

# MINOR_KEY_DICT = {'0': 10, '1': 5, '2': 12, '3': 7, '4': 2, '5': 9, '6': 4, '-1': 3, 
#                   '-2': 8, '-3': 1, '-4': 6, '-5': 11, '-6': 4 }

KEYS_DICT = {"major": {'0': 'C', '1': 'G', '2': 'D', '3': 'A', '4': 'E', '5': 'B', '6': 'F#', 
                  '-1': 'F', '-2': 'Bb', '-3': 'Eb', '-4': 'Ab', '-5': 'Db', '-6': 'Gb'},
            "minor": {'0': 'A', '1': 'E', '2': 'B', '3': 'F#', '4': 'C#', '5': 'G#', '6': 'D#', 
                  '-1': 'D', '-2': 'G', '-3': 'C', '-4': 'F', '-5': 'Bb', '-6': 'Eb'}}


def convert_to_harte(harmony_dict):
    pass

def get_key(jsdict):
    key_dict = jsdict["part"]["measures"][0]["attributes"]["key"]
    position = key_dict["fifths"]["text"]
    mode = key_dict["mode"]["text"]
    return "%s %s" % (KEY_DICT[mode][position], mode)

def get_time_signature(jsdict):
    time_dict = jsdict["part"]["measures"][0]["attributes"]["time"]
    return "%s/%s" % (time_dict["beats"], time_dict["beat-type"])

def get_divisions(jsdict):
    return int(jsdict["part"]["measures"][0]["attributes"]["divisions"]["text"])

def parse_measure(measure_dict, divisions=96, **kwargs):
    parsed = {"harmonies": [], "notes": [], "durations": []}

    pass

def parse_json(fpath):
    print("Parsing %s" % fpath)
    jsdict = json.load(fpath)
    divisions = get_divisions(jsdict)
    parsed_data = {"title": jsdict["movement-title"]["text"],
                   "artist": jsdict["identification"]["creator"]["text"],
                   "key": get_key(jsdict),
                   "time_signature": get_time_signature(jsdict),
                   "measures": []}
    for measure in jsdict['part']['measures']:
        parsed_data['measures'].append(parse_measure(measure), divisions)
    


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--song", help="Name of songfile to parse", type=str)
    # args = parser.parse_args()
    # fname = args.song
    # fname = "charlie_parker-moose_the_mooche.json"
    root_dir = str(Path(op.abspath(__file__)).parents[2])
    json_dir = op.join(root_dir, 'data', 'raw', 'json')
    json_paths = [op.join(json_dir, fname) for fname in os.listdir(json_dir)]
    parsed_data = []
    for json_path in json_paths:
        parsed_data.append(parse_json(json_path))
