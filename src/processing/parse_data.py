from __future__ import print_function
import json
import numpy as np
import os
import os.path as op
import pickle
from pathlib import Path
from harmony import Harmony

NOTES_MAP = {'B#': 1, 'C': 1, 'C#': 2, 'Db': 2, 'D': 3, 'D#': 4, 'Eb': 4, 'E': 5, 
              'Fb': 5, 'E#': 6, 'F': 6, 'F#': 7, 'Gb': 7, 'G': 8, 'G#': 9, 'Ab': 9, 
              'A': 10, 'A#': 11, 'Bb': 11, 'B': 12, 'Cb': 12, 'rest': 0}

DURATIONS_MAP = {'whole': 0, 'half': 1, 'quarter': 2, 'eighth': 3, '16th': 4, 
                 'whole-trip': 5, 'half-trip': 6, 'quarter-trip': 7, 'eighth-trip': 8, 
                 '16th-trip': 9, 'whole-dot': 10, 'half-dot': 11, 'quarter-dot': 12, 
                 'eighth-dot': 13, '16th-dot': 14, '32nd': 15, 
                 '32nd-trip': 16, '32nd-dot': 17}

KEYS_DICT = {"major": {'0': 'C', '1': 'G', '2': 'D', '3': 'A', '4': 'E', '5': 'B', '6': 'F#', 
                  '-1': 'F', '-2': 'Bb', '-3': 'Eb', '-4': 'Ab', '-5': 'Db', '-6': 'Gb'},
            "minor": {'0': 'A', '1': 'E', '2': 'B', '3': 'F#', '4': 'C#', '5': 'G#', '6': 'D#', 
                  '-1': 'D', '-2': 'G', '-3': 'C', '-4': 'F', '-5': 'Bb', '-6': 'Eb'}}

def get_key(jsdict):
    key_dict = jsdict["part"]["measures"][0]["attributes"]["key"]
    position = key_dict["fifths"]["text"]
    mode = key_dict["mode"]["text"]
    return "%s %s" % (KEYS_DICT[mode][position], mode)

def get_time_signature(jsdict):
    time_dict = jsdict["part"]["measures"][0]["attributes"]["time"]
    return "%s/%s" % (time_dict["beats"], time_dict["beat-type"])

def get_divisions(jsdict):
    return int(jsdict["part"]["measures"][0]["attributes"]["divisions"]["text"])

def get_note_duration(note_dict, division=24):
    note_dur = int(note_dict["duration"]["text"])
    note_type = note_dict["type"]["text"]

    dur_dict = {'whole': division*4, 'half':  division*2, 'quarter': division, 
                'eighth': division/2, '16th': division/4, '32nd': division/8}

    label = note_type
    if note_dur == dur_dict[note_type]:
        pass
    if note_dur == (3 * dur_dict[note_type] / 2):
        label = '-'.join([label, 'dot'])
    elif note_dur == (dur_dict[note_type] * 2 / 3):
        label = '-'.join([label, 'triplet'])
    else: 
        print("Undefined %s duration. Entering as regular %s." % (note_type, note_type))
        
    return DURATIONS_MAP[label]

def parse_note(note_dict, division=24):
    if "rest" in note_dict.keys():
        note = NOTES_MAP["rest"]
        octave = -1
    elif "pitch" in note_dict.keys():
        note_string = note_dict["pitch"]["step"]["text"]
        if "alter" in note_dict["pitch"].keys():
            note_string += (lambda x: "b" if -1 else ("#" if 1 else ""))(
                                note_dict["pitch"]["alter"]["text"])
        note = NOTES_MAP[note_string]
        octave = note_dict["pitch"]["octave"]["text"]

    duration = get_note_duration(note_dict, division)
    return note, octave, duration

def parse_measure(measure_dict, divisions=24):
    parsed = {"harmonies": [], "notes": [], "octaves": [], "duration-tags": []}
    for harmony_dict in measure_dict["harmonies"]:
        parsed["harmonies"].append(Harmony(harmony_dict).get_pitch_classes_binary())
    
    for note_dict in measure_dict["notes"]:
        note, octave, duration = parse_note(note_dict, divisions) 
        parsed["notes"].append(note)
        parsed["octaves"].append(octave)
        parsed["duration-tags"].append(duration)

    return parsed

def parse_json(fpath):
    print("Parsing %s" % fpath)
    jsdict = json.load(open(fpath))
    divisions = get_divisions(jsdict)

    parsed = {"title": jsdict["movement-title"]["text"],
              "artist": jsdict["identification"]["creator"]["text"],
              "key": get_key(jsdict),
              "time_signature": get_time_signature(jsdict),
              "measures": []}

    for measure in jsdict['part']['measures']:
        parsed['measures'].append(parse_measure(measure), divisions)

    return parsed
    

if __name__ == '__main__':
    root_dir = str(Path(op.abspath(__file__)).parents[2])
    json_dir = op.join(root_dir, 'data', 'raw', 'json')

    json_paths = [op.join(json_dir, fname) for fname in os.listdir(json_dir)]
    parsed_data = []
    for json_path in json_paths:
        import pdb
        parsed_data.append(parse_json(json_path))
        pdb.set_trace()

    pickle.dump(parsed_data, "parsed_data.pkl")
