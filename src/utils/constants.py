"""
Constant Values
"""
# Strings
PITCH_KEY = "pitch_numbers"
DUR_KEY = "duration_tags"
CHORD_KEY = "harmony"
POS_KEY = "bar_positions"

# Model Params
PITCH_DIM = 128
DUR_DIM = 19
PITCH_EMBED_DIM = 36
DUR_EMBED_DIM = 10
NUM_RNN_LAYERS = 2

# Dictionaries
NOTES_MAP = {'rest': 127, 'B#': 0, 'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 
             'E': 4, 'Fb': 4, 'E#': 5, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 
             'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11}

DURATIONS_MAP = {
    '32nd-triplet': 0, 
    '32nd': 2, 
    '16th-triplet': 3,
    '32nd-dot': 4,
    '16th': 5,
    '8th-triplet': 6,
    '16th-dot': 7,
    '8th': 9,
    'quarter-triplet': 10,
    '8th-dot': 11,
    'quarter': 12,
    'half-triplet': 13,
    'quarter-dot': 14,
    'half': 15,
    'whole-triplet': 16,
    'half-dot': 17,
    'whole': 18,
    'double-triplet': 19,
    'whole-dot': 20,
    'double': 20,
    'double-dot': 21,
    'none': 22
}


KEYS_DICT = {"major": {'0': 'C', '1': 'G', '2': 'D', '3': 'A', '4': 'E', '5': 'B', '6': 'F#', 
                  '-1': 'F', '-2': 'Bb', '-3': 'Eb', '-4': 'Ab', '-5': 'Db', '-6': 'Gb'},
            "minor": {'0': 'A', '1': 'E', '2': 'B', '3': 'F#', '4': 'C#', '5': 'G#', '6': 'D#', 
                  '-1': 'D', '-2': 'G', '-3': 'C', '-4': 'F', '-5': 'Bb', '-6': 'Eb'}}
