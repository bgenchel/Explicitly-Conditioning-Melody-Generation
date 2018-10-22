import json
import os
import os.path as op
from pathlib import Path

def main(json_scores):
    data = {'modes': None, 
            'files_without_keys': 0, 
            'files_without_key_mode': 0, 
            'files_with_multiple_keys': {
                'all_keys_have_modes': 0,
                'not_all_keys_have_modes': 0},
            'num_notes_with_chord_tag': 0, 
            'files_with_notes_with_chord_tag': {},
            'num_notes_with_accidental_tag': 0, 
            'files_with_notes_with_accidental_tag': {}}

    modes = set()
    for js in json_scores:
        num_keys = 0
        key_modes = 0
        num_chord_notes = 0
        num_accidental_notes = 0
        for measure in js['part']['measures']:
            if "key" in measure["attributes"]:
                num_keys += 1
                if "mode" in measure['attributes']['key']:
                    modes.add(measure['attributes']['key']['mode']['text'])
                    key_modes += 1

            for note in measure['notes']:
                if 'chord' in note.keys():
                    num_chord_notes += 1
                if 'accidental' in note.keys():
                    num_accidental_notes += 1

        if num_keys == 0:
            data['files_without_keys'] += 1
        if num_keys == 1 and key_modes == 0:
            data['files_without_key_mode'] += 1
        if num_keys > 1:
            if num_keys == key_modes:
                data['files_with_multiple_keys']['all_keys_have_modes'] += 1
            else:
                data['files_with_multiple_keys']['not_all_keys_have_modes'] += 1

        if num_chord_notes > 0:
            data['files_with_notes_with_chord_tag'][js['fname']] = num_chord_notes
            data['num_notes_with_chord_tag'] += num_chord_notes
        
        if num_accidental_notes > 0:
            data['files_with_notes_with_accidental_tag'][js['fname']] = num_accidental_notes
            data['num_notes_with_accidental_tag'] += num_accidental_notes

    data['modes'] = list(modes)
    with open('dataset_problems_profile.json', 'w') as fp:
        json.dump(data, fp, indent=4)


if __name__ == '__main__':
    root_dir = str(Path(op.abspath(__file__)).parents[3])
    json_path = op.join(root_dir, 'data', 'raw', 'json')
    if not op.exists(json_path):
        raise Exception("no json directory exists.")

    fpaths = [op.join(json_path, fname) for fname in os.listdir(json_path)]
    js_scores = []
    for fpath in fpaths:
        js = json.load(open(fpath, 'r'))
        js['fname'] = op.basename(fpath)
        js_scores.append(js)
    main(js_scores)
