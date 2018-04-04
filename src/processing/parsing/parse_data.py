from __future__ import print_function
import json
import numpy as np
import os
import os.path as op
import pickle
import pdb
from pathlib import Path
from harmony import Harmony

NOTES_MAP = {'rest': 0, 'B#': 1, 'C': 1, 'C#': 2, 'Db': 2, 'D': 3, 'D#': 4, 'Eb': 4, 
             'E': 5, 'Fb': 5, 'E#': 6, 'F': 6, 'F#': 7, 'Gb': 7, 'G': 8, 'G#': 9, 
             'Ab': 9, 'A': 10, 'A#': 11, 'Bb': 11, 'B': 12, 'Cb': 12}

DURATIONS_MAP = {'whole': 0, 'half': 1, 'quarter': 2, 'eighth': 3, '16th': 4, 
                 'whole-triplet': 5, 'half-triplet': 6, 'quarter-triplet': 7, 
                 'eighth-triplet': 8, '16th-triplet': 9, 'whole-dot': 10, 'half-dot': 11, 
                 'quarter-dot': 12, 'eighth-dot': 13, '16th-dot': 14, '32nd': 15, 
                 '32nd-triplet': 16, '32nd-dot': 17, 'other': -1}

KEYS_DICT = {"major": {'0': 'C', '1': 'G', '2': 'D', '3': 'A', '4': 'E', '5': 'B', '6': 'F#', 
                  '-1': 'F', '-2': 'Bb', '-3': 'Eb', '-4': 'Ab', '-5': 'Db', '-6': 'Gb'},
            "minor": {'0': 'A', '1': 'E', '2': 'B', '3': 'F#', '4': 'C#', '5': 'G#', '6': 'D#', 
                  '-1': 'D', '-2': 'G', '-3': 'C', '-4': 'F', '-5': 'Bb', '-6': 'Eb'}}

def get_key(jsdict):
    """
    I know from analysis that the only keys in my particular dataset are major
    and minor.
    """
    key = None
    multiple = False
    for measure in jsdict["part"]["measures"]:
        if "key" in measure["attributes"].keys():
            if key is not None:
                multiple = True
                break
            else:
                key_dict = jsdict["part"]["measures"][0]["attributes"]["key"]
                position = key_dict["fifths"]["text"]
                if "mode" in key_dict.keys():
                    mode = key_dict["mode"]["text"]
                else:
                    mode = "major" # just assume, it doesn't really matter anyways
                key = "%s%s" % (KEYS_DICT[mode][position], mode)
    return key, multiple

def get_time_signature(jsdict):
    time_dict = jsdict["part"]["measures"][0]["attributes"]["time"]
    return "%s/%s" % (time_dict["beats"], time_dict["beat-type"])

def get_divisions(jsdict):
    return int(jsdict["part"]["measures"][0]["attributes"]["divisions"]["text"])

def get_note_duration(note_dict, division=24):
    dur_dict = {'whole': division*4, 'half':  division*2, 'quarter': division, 
                'eighth': division/2, '16th': division/4, '32nd': division/8}

    if "duration" not in note_dict.keys():
        note_dur = -1
    else:
        note_dur = float(note_dict["duration"]["text"])

    if "type" in note_dict.keys():
        note_type = note_dict["type"]["text"]
        label = note_type
        if note_dur == (3 * dur_dict[note_type] / 2):
            label = '-'.join([label, 'dot'])
        elif note_dur == (dur_dict[note_type] * 2 / 3):
            label = '-'.join([label, 'triplet'])
        elif note_dur != dur_dict[note_type]: 
            print("Undefined %s duration. Entering as regular %s." % (note_type, note_type))
    elif note_dur == dur_dict["whole"]:
        label = "whole"
    elif note_dur == (3 * dur_dict["whole"] / 2):
        label = "whole-dot"
    elif note_dur == (2 * dur_dict["whole"] / 3):
        label = "whole-triplet"
    elif note_dur == dur_dict["half"]:
        label = "half"
    elif note_dur == (3 * dur_dict["half"] / 2):
        label = "half-dot"
    elif note_dur == (2 * dur_dict["half"] / 3):
        label = "half-triplet"
    elif note_dur == dur_dict["quarter"]:
        label = "quarter"
    elif note_dur == (3 * dur_dict["quarter"] / 2):
        label = "quarter-dot"
    elif note_dur == (2 * dur_dict["quarter"] / 3):
        label = "quarter-triplet"
    elif note_dur == dur_dict["eighth"]:
        label = "eighth"
    elif note_dur == (3 * dur_dict["eighth"] / 2):
        label = "eighth-dot"
    elif note_dur == (2 * dur_dict["eighth"] / 3):
        label = "eighth-triplet"
    elif note_dur == dur_dict["16th"]:
        label = "16th"
    elif note_dur == (3 * dur_dict["16th"] / 2):
        label = "16th-dot"
    elif note_dur == (2 * dur_dict["16th"] / 3):
        label = "16th-triplet"
    elif note_dur == dur_dict["32nd"]:
        label = "32nd"
    elif note_dur == (3 * dur_dict["32nd"] / 2):
        label = "32nd-dot"
    elif note_dur == (3 * dur_dict["32nd"] / 3):
        label = "33nd-triplet"
    else:
        print("Undefined duration %.2f. Labeling 'other'." % (note_dur/division))
        label = "other"
    return DURATIONS_MAP[label]

def parse_note(note_dict, division=24):
    if "rest" in note_dict.keys():
        pitch_num = NOTES_MAP["rest"]
        # octave = -1
    elif "pitch" in note_dict.keys():
        note_string = note_dict["pitch"]["step"]["text"]
        if "alter" in note_dict["pitch"].keys():
            note_string += (lambda x: "b" if -1 else ("#" if 1 else ""))(
                                note_dict["pitch"]["alter"]["text"])
        octave = int(note_dict["pitch"]["octave"]["text"])
        pitch_num = (octave + 1)*12 + NOTES_MAP[note_string]

    duration = get_note_duration(note_dict, division)
    return pitch_num, duration

def parse_measure(measure_dict, divisions=24):
    parsed = {"harmonies": [], "pitch_numbers": [], "duration_tags": []}
    for harmony_dict in measure_dict["harmonies"]:
        parsed["harmonies"].append(Harmony(harmony_dict).get_simple_pitch_classes_binary())

    for note_dict in measure_dict["notes"]:
        # want monophonic, so we'll just take the top note
        if "chord" in note_dict.keys() or "grace" in note_dict.keys():
            continue
        else:
            pitch_num, duration = parse_note(note_dict, divisions) 
            parsed["pitch_numbers"].append(pitch_num)
            parsed["duration_tags"].append(duration)

    return parsed

def parse_json(fpath):
    print("Parsing %s" % op.basename(fpath))
    jsdict = json.load(open(fpath))
    divisions = get_divisions(jsdict)

    key, multiple = get_key(jsdict)
    if multiple is True:
        return None

    parsed = {"title": jsdict["movement-title"]["text"],
              "artist": jsdict["identification"]["creator"]["text"],
              "key": key, # skip files with multiple keys
              "time_signature": get_time_signature(jsdict),
              "measures": []}

    for measure in jsdict['part']['measures']:
        parsed['measures'].append(parse_measure(measure, divisions))

    # try to fill in harmonies somewhat niavely
    for i, measure in enumerate(parsed['measures']):
        if not measure['harmonies']:
            if i == 0:
                for after_measure in parsed['measures'][i+1:]:
                    if after_measure['harmonies']:
                        measure['harmonies'].append(after_measure['harmonies'][0])
                        break
            else: 
                for before_measure in parsed['measures'][i-1::-1]:
                    if before_measure['harmonies']:
                        measure['harmonies'].append(before_measure['harmonies'][0])
                        break

    return parsed


if __name__ == '__main__':
    root_dir = str(Path(op.abspath(__file__)).parents[3])
    json_dir = op.join(root_dir, 'data', 'raw', 'json')
    pkl_dir = op.join(root_dir, 'data', 'processed', 'pkl')

    if not op.exists(json_dir):
        raise Exception("Json directory not found.")
    if not op.exists(pkl_dir):
        os.makedirs(pkl_dir)

    json_paths = [op.join(json_dir, fname) for fname in os.listdir(json_dir)]
    parsed_data = []
    for json_path in json_paths:
        parsed = parse_json(json_path)
        if parsed is not None:
            pd = parse_json(json_path)
            outpath = op.join(pkl_dir, op.basename(json_path).replace('.json', '.pkl'))
            pickle.dump(pd, open(outpath, 'wb'))
            parsed_data.append(pd)

    pickle.dump(parsed_data, open(op.join(pkl_dir, "dataset.pkl"), 'wb'))
