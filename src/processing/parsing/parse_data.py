from __future__ import print_function
import copy
import json
import os
import os.path as op
import pickle
import random
import sys
from pathlib import Path
from harmony import Harmony

sys.path.append(str(Path(op.abspath(__file__)).parents[2]))
from utils.constants import NOTES_MAP, DURATIONS_MAP, KEYS_DICT


def rotate(l, x):
    return l[-x:] + l[:-x]

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
                try:
                    key = "%s%s" % (KEYS_DICT[mode][position], mode)
                except KeyError:
                    print("Error!! mode: {}, position: {}".format(mode, position))
                    key = None


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
    return DURATIONS_MAP[label], note_dur

def parse_note(note_dict, division=24):
    if "rest" in note_dict.keys():
        pitch_num = NOTES_MAP["rest"]
    elif "pitch" in note_dict.keys():
        note_string = note_dict["pitch"]["step"]["text"]
        if "alter" in note_dict["pitch"].keys():
            note_string += (lambda x: "b" if -1 else ("#" if 1 else ""))(
                                note_dict["pitch"]["alter"]["text"])
        octave = int(note_dict["pitch"]["octave"]["text"])
        pitch_num = (octave + 1)*12 + NOTES_MAP[note_string]

    dur_tag, dur_ticks = get_note_duration(note_dict, division)
    return pitch_num, dur_tag, dur_ticks

def parse_measure(measure_dict, divisions=24):
    parsed = {"harmonies": [], "pitch_numbers": [], "duration_tags": [], "half_index": 0}
    for harmony_dict in measure_dict["harmonies"]:
        parsed["harmonies"].append(Harmony(harmony_dict).get_simple_pitch_classes_binary())

    dur_ticks_list = []
    for note_dict in measure_dict["notes"]:
        # want monophonic, so we'll just take the top note
        if "chord" in note_dict.keys() or "grace" in note_dict.keys():
            continue
        else:
            pitch_num, dur_tag, dur_ticks = parse_note(note_dict, divisions) 
            parsed["pitch_numbers"].append(pitch_num)
            parsed["duration_tags"].append(dur_tag)
            dur_ticks_list.append(dur_ticks)
    # pdb.set_trace()
    dur_ticks_list = [sum(dur_ticks_list[:i]) for i in range(len(dur_ticks_list))]
    half_way_ticks = dur_ticks_list[-1]/2
    for i, ticks in enumerate(dur_ticks_list):
        if ticks > half_way_ticks:
            parsed["half_index"] = i
            break

    return parsed

def parse_json(fpath):
    print("Parsing %s" % op.basename(fpath))
    jsdict = json.load(open(fpath))
    divisions = get_divisions(jsdict)

    key, multiple = get_key(jsdict)
    if multiple is True:
        return None

    fname = op.basename(fpath)
    parts = fname.split('-')
    artist = parts[0]
    title = '-'.join(parts[1:])
    if "movement-title" in jsdict:
        title = jsdict['movement-title']['text']
    if ("identification" in jsdict) and ("creator" in jsdict["identification"]):
        artist = jsdict["identification"]["creator"]
    parsed = {"title": title,
              "artist": artist,
              "key": key, # skip files with multiple keys
              "time_signature": get_time_signature(jsdict),
              "measures": []}

    for measure in jsdict['part']['measures']:
        parsed['measures'].append(parse_measure(measure, divisions))

    # try to fill in harmonies somewhat niavely
    harmonies_per_measure = set()
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
        harmonies_per_measure.add(len(measure['harmonies']))

    if max(harmonies_per_measure) == 0:
        return None
    elif max(harmonies_per_measure) > 2:
        return None

    return parsed


if __name__ == '__main__':
    root_dir = str(Path(op.abspath(__file__)).parents[3])
    json_dir = op.join(root_dir, 'data', 'raw', 'json')
    song_dir = op.join(root_dir, 'data', 'processed', 'songs')
    dataset_dir = op.join(root_dir, 'data', 'processed', 'datasets')

    if not op.exists(json_dir):
        raise Exception("Json directory not found.")
    if not op.exists(song_dir):
        os.makedirs(song_dir)
    if not op.exists(dataset_dir):
        os.makedirs(dataset_dir)

    json_paths = [op.join(json_dir, fname) for fname in os.listdir(json_dir)]
    parsed_data = []
    charlie_parker_data = []
    for json_path in json_paths:
        parsed = parse_json(json_path)
        if parsed is not None:
            for shift in range(-6, 6):
                transposed = copy.deepcopy(parsed)
                transposed['transposition'] = shift
                print("transposing by %i" % shift)
                for i, measure in enumerate(transposed['measures']):
                    measure['pitch_numbers'] = [
                            (lambda n: n + shift if n != NOTES_MAP["rest"] else n)(pn) 
                            for pn in measure['pitch_numbers']]
                    measure['harmonies'] = [rotate(h, shift) for h in measure['harmonies']]
                    transposed['measures'][i] = measure
                outname = op.basename(json_path).replace('.json', '')
                outpath = op.join(song_dir, '_'.join([outname, str(shift)])) + '.pkl'
                pickle.dump(transposed, open(outpath, 'wb'))

                parsed_data.append(transposed)
                if 'charlie_parker' in op.basename(outpath):
                    charlie_parker_data.append(parsed)

    random.shuffle(parsed_data)
    parsed_data_dict = {'train': parsed_data[:int(0.9*len(parsed_data))],
                        'valid': parsed_data[int(0.9*len(parsed_data)):]}
    pickle.dump(parsed_data_dict, open(op.join(dataset_dir, "dataset.pkl"), 'wb'))

    random.shuffle(charlie_parker_data)
    charlie_parker_dict = {'train': charlie_parker_data[:int(0.9*len(charlie_parker_data))],
                           'valid': charlie_parker_data[int(0.9*len(charlie_parker_data)):]}
    pickle.dump(charlie_parker_dict, open(op.join(dataset_dir, "charlie_parker_dataset.pkl"), 'wb'))
