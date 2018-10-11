import json
import os.path as op
import os
import sys
from pathlib import Path
from harmony import Harmony
from copy import deepcopy
import pickle

sys.path.append(str(Path(op.abspath(__file__)).parents[2]))
from utils.constants import NOTES_MAP, DURATIONS_MAP, KEYS_DICT


##################################################################
# Parser
#   Loads, parses, and formats JSON data into model-ready inputs
##################################################################
class Parser:
    def __init__(self, output="pitch_duration_tokens", json_dir=None, root_dir=None, song_dir=None, dataset_dir=None):
        self.root_dir, \
        self.json_dir, \
        self.song_dir, \
        self.dataset_dir = self.verify_directories(root_dir, json_dir, song_dir, dataset_dir)

        # Individual file names
        self.json_paths = [op.join(self.json_dir, filename) for filename in os.listdir(self.json_dir)]

        # Storage for the parsed output
        self.parsed = None

        # Parse based on output format
        if output == "pitch_duration_tokens":
            pass
        elif output == "midi_ticks":
            self.ticks = 24
            self.parse_midi_ticks()

    ############################################
    # Ensures that all input/output dirs exist
    ############################################
    def verify_directories(self, root_dir, json_dir, song_dir, dataset_dir):
        # Directory of the project, from which to base other dirs
        if not root_dir:
            # Looks up 3 directories to get project dir
            root_dir = str(Path(op.abspath(__file__)).parents[3])

        # Directory where JSON files to be parsed live
        if json_dir and not op.exists(json_dir):
            raise Exception("JSON directory not found.")
        else:
            json_dir = op.join(root_dir, 'data', 'raw', 'json')
            if not op.exists(json_dir):
                raise Exception("JSON directory not found.")

        # Directory where individual parsed songs get saved to
        if song_dir and not op.exists(song_dir):
            os.makedirs(song_dir)
        else:
            song_dir = op.join(root_dir, 'data', 'processed', 'songs')
            if not op.exists(song_dir):
                os.makedirs(song_dir)

        # Directory where datasets as a whole get saved to
        if dataset_dir and not op.exists(dataset_dir):
            os.makedirs(song_dir)
        else:
            dataset_dir = op.join(root_dir, 'data', 'processed', 'datasets')
            if not op.exists(dataset_dir):
                os.makedirs(dataset_dir)

        return root_dir, json_dir, song_dir, dataset_dir

    ##############################################
    # Given the JSON input as a dict, returns:
    #   {
    #       title,
    #       artist,
    #       key,
    #       time_signature
    #   }
    #   or, None, if a song has more than one key
    ##############################################
    def parse_metadata(self, filename, song_dict):
        # Strip filename of path and file extension (.json)
        filename = filename.split('/')[-1][:-5]

        # Key
        key, multiple = get_key(song_dict)
        if multiple:
            # Disregard songs with multiple keys
            return None

        # Title
        if "movement-title" in song_dict:
            title = song_dict["movement-title"]["text"]
        elif "work" in song_dict:
            if "work-title" in song_dict["work"]:
                title = song_dict["work"]["work-title"]["text"]
        else:
            title = filename.split('-')[1]

        # Artist
        if ("identification" in song_dict) and ("creator" in song_dict["identification"]):
            artist = song_dict["identification"]["creator"]["text"]
        else:
            artist = filename.split('-')[0]

        # Time signature
        time_dict = song_dict["part"]["measures"][0]["attributes"]["time"]
        time_signature = "%s/%s" % (time_dict["beats"]["text"], time_dict["beat-type"]["text"])

        return {
            "title": title,
            "artist": artist,
            "key": key,
            "time_signature": time_signature
        }

    ###########################################################
    # Parses the JSON, and returns a list of dicts for each
    #   song in the following format:
    #   {
    #       metadata: {
    #           title,
    #           artist,
    #           key,
    #           time_signature
    #       },
    #       measures: [[{
    #           harmony,
    #           ticks (num_ticks x 38) (F3-E6)
    #       }], ...]
    #   }
    #   or, None, if a song has more than one key
    ###########################################################
    def parse_midi_ticks(self):
        songs = []

        for filename in self.json_paths:
            # Load song dict
            try:
                song_dict = json.load(open(filename))
            except:
                print("Unable to load JSON for %s" % filename)
                continue

            # Get song metadata
            metadata = self.parse_metadata(filename, song_dict)

            # Get song divisions
            divisions = get_divisions(song_dict)

            # Calculate scale factor from divisions to ticks
            #  i.e. scale_factor * divisions = num_ticks
            scale_factor = self.ticks / divisions

            if scale_factor > 1:
                raise Exception("Error: MusicXML has lower resolution than desired MIDI ticks.")

            # Parse each measure
            measures = []
            last_harmony = []
            for measure in song_dict["part"]["measures"]:
                parsed_measure, last_harmony = self.parse_measure_midi_ticks(measure, scale_factor, last_harmony)
                measures.append(parsed_measure)

            songs.append({
                "metadata": metadata,
                "measures": measures
            })

        self.parsed = songs

    ##########################################################################
    # For a measure, returns a set of ticks grouped by associated harmony in
    #   the following format:
    #   [{
    #      harmony,
    #      ticks (num_ticks x 38) (F3-E6)
    #   },
    #   ...]
    ##########################################################################
    def parse_measure_midi_ticks(self, measure, scale_factor, last_harmony):
        parsed_measure = []
        new_last_harmony = last_harmony

        for group in measure["groups"]:
            # Set note value for each tick in the measure
            ticks_by_note = []
            for note in group["notes"]:
                if not "duration" in note:
                    print("Skipping grace note...")
                    continue
                note_divisions = int(note["duration"]["text"])
                note_ticks = round(scale_factor * note_divisions)
                note_index = get_note_index(note)

                for i in range(note_ticks):
                    tick = [0 for _ in range(37)]
                    tick[note_index] = 1
                    ticks_by_note.append(tick)

            if not group["harmony"]:
                parsed_measure.append({
                    "harmony": last_harmony,
                    "ticks": ticks_by_note
                })
            else:
                harmony = Harmony(group["harmony"]).get_seventh_pitch_classes_binary()
                new_last_harmony = harmony
                parsed_measure.append({
                    "harmony": harmony,
                    "ticks": ticks_by_note
                })

        return parsed_measure, new_last_harmony

    def save_parsed(self, transpose=False):
        # Ensure parsing has happened
        if not self.parsed:
            print("Nothing has been parsed.")
            return

        for song in self.parsed:
            if transpose:
                for steps in range(-6, 6):
                    transposed = transpose_song_midi_ticks(song, steps)
                    filename = "-".join([
                        "_".join(transposed["metadata"]["title"].split(" ")),
                        "_".join(transposed["metadata"]["artist"].split(" "))]) + "_%d" % steps + ".pkl"
                    filename = filename.replace("/", ",")
                    outpath = op.join(self.song_dir, filename)
                    pickle.dump(transposed, open(outpath, 'wb'))
            else:
                filename = "-".join([
                    "_".join(song["metadata"]["title"].split(" ")),
                    "_".join(song["metadata"]["artist"].split(" "))]) + ".pkl"
                filename = filename.replace("/", ",")
                outpath = op.join(self.song_dir, filename)
                pickle.dump(song, open(outpath, 'wb'))


#############################################
# Returns:
#   If there's only one key: key, False
#   If there's more than one key: None, True
##############################################
def get_key(song_dict):
    """
    I know from analysis that the only keys in my particular dataset are major
    and minor.
    """
    key = None
    multiple = False

    # Check each measure to ensure there isn't a key change, if so, return
    for measure in song_dict["part"]["measures"]:
        if "key" in measure["attributes"].keys():
            if key is not None:
                multiple = True
                return key, multiple

    # Get key from first measure
    key_dict = song_dict["part"]["measures"][0]["attributes"]["key"]
    position = key_dict["fifths"]["text"]
    if "mode" in key_dict.keys():
        mode = key_dict["mode"]["text"]
    else:
        mode = "major"  # just assume, it doesn't really matter anyways
    try:
        key = "%s%s" % (KEYS_DICT[mode][position], mode)
    except KeyError:
        print("Error!! mode: {}, position: {}".format(mode, position))
        key = None

    return key, multiple


##########################################################################
# Returns the divisions per quarter note in the JSON represented MusicXML
##########################################################################
def get_divisions(song_dict):
    return int(song_dict["part"]["measures"][0]["attributes"]["divisions"]["text"])


##################################################################
# Given a MusicXML note element, squashes the note between F3-E6,
#   returning an index where 0 is F3, 35 is E6, and 36 is 'rest'
##################################################################
def get_note_index(note):
    if "rest" in note.keys():
        return 36
    else:
        note_string = note["pitch"]["step"]["text"]
        if "alter" in note["pitch"].keys():
            note_string += (lambda x: "b" if -1 else ("#" if 1 else ""))(
                note["pitch"]["alter"]["text"])
        note_int = NOTES_MAP[note_string]
        octave = int(note["pitch"]["octave"]["text"])

        # Squash to F3-E6
        if octave < 3:
            if note_int < 5:
                print("Note %s out of range, transposing..." % note_string)
                octave = 4
            else:
                octave = 3
        elif octave > 5:
            if note_int > 5:
                print("Note %s out of range, transposing..." % note_string)
                octave = 5
            else:
                octave = 6

        # 0 - 35
        return (((octave - 3) * 12) + note_int) - 5


########################################
# Transposes a song in MIDI ticks form
########################################
def transpose_song_midi_ticks(song, steps):
    sign = lambda x: (1, -1)[x < 0]

    transposed = deepcopy(song)
    for measure in transposed["measures"]:
        for group in measure:
            if group["harmony"]:
                # Transpose harmony
                group["harmony"] = rotate(group["harmony"], steps)

            # Transpose ticks
            direction = sign(steps)
            new_ticks = group["ticks"]
            for _ in range(abs(steps)):
                ticks = []
                for tick in new_ticks:
                    ticks.append(transpose_ticks(tick, direction))
                new_ticks = ticks
            group["ticks"] = new_ticks

    return transposed


###################################################
# Transposes a MIDI tick array one step up or down
###################################################
def transpose_ticks(ticks, direction):
    note = ticks.index(1)

    # Don't have to transpose a rest
    if note == len(ticks) - 1:
        return ticks

    # Transpose up a step
    if direction > 0:
        if note == len(ticks) - 2:
            # Make E6 transpose "up" to F5
            transposed = [0 for i in range(len(ticks))]
            transposed[note - 11] = 1
            return transposed
        else:
            transposed = ticks
            transposed.insert(0, transposed.pop(len(transposed) - 2))
            return transposed
    else:
        # Transpose down a step
        if note == 0:
            # Make F3 transpose "down" to E4
            transposed = [0 for i in range(len(ticks))]
            transposed[note + 11] = 1
            return transposed
        else:
            transposed = ticks
            transposed.insert(len(transposed) - 2, transposed.pop(0))
            return transposed


#########################################
# Rotates a list a given number of steps
#########################################
def rotate(l, x):
    return l[-x:] + l[:-x]


if __name__ == '__main__':
    parser = Parser(output="midi_ticks")
    parser.save_parsed()
