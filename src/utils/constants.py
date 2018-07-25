"""
"""
PITCH_DIM = 128
DUR_DIM = 18

DEFAULT_PRINT_EVERY = 50
DEFAULT_WRITE_EVERY = 100

NOTES_MAP = {'rest': 127, 'B#': 0, 'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 
             'E': 4, 'Fb': 4, 'E#': 5, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 
             'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11}

DURATIONS_MAP = {'whole': 0, 'half': 1, 'quarter': 2, 'eighth': 3, '16th': 4, 
                 'whole-triplet': 5, 'half-triplet': 6, 'quarter-triplet': 7, 
                 'eighth-triplet': 8, '16th-triplet': 9, 'whole-dot': 10, 'half-dot': 11, 
                 'quarter-dot': 12, 'eighth-dot': 13, '16th-dot': 14, '32nd': 15, 
                 '32nd-triplet': 16, '32nd-dot': 17, 'other': -1}

KEYS_DICT = {"major": {'0': 'C', '1': 'G', '2': 'D', '3': 'A', '4': 'E', '5': 'B', '6': 'F#', 
                  '-1': 'F', '-2': 'Bb', '-3': 'Eb', '-4': 'Ab', '-5': 'Db', '-6': 'Gb'},
            "minor": {'0': 'A', '1': 'E', '2': 'B', '3': 'F#', '4': 'C#', '5': 'G#', '6': 'D#', 
                  '-1': 'D', '-2': 'G', '-3': 'C', '-4': 'F', '-5': 'Bb', '-6': 'Eb'}}
