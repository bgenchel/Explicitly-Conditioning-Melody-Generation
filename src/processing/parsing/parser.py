import json
import os.path as op
import os
import sys
from pathlib import Path
from harmony import Harmony
from copy import deepcopy
import pickle
import argparse

sys.path.append(str(Path(op.abspath(__file__)).parents[2]))
from utils.constants import NOTES_MAP, DURATIONS_MAP, KEYS_DICT


class Parser:
    def __init__(self, output="pitch_duration_tokens", root_dir=None, json_dir=None, song_dir=None, dataset_dir=None):
        """
        Loads, parses, and formats JSON data into model-ready inputs.
        :param output: the desired parsing format, either "pitch_duration_tokens" or "midi_ticks"
        :param root_dir: the project directory
        :param json_dir: the directory containing the JSON formatted MusicXML
        :param song_dir: the directory to save parsed individual songs
        :param dataset_dir: the directory to save full datasets
        """
        self.output = output

        self.root_dir, \
        self.json_dir, \
        self.song_dir, \
        self.dataset_dir = self.verify_directories(root_dir, json_dir, song_dir, dataset_dir)

        # Individual file names
        self.json_paths = [op.join(self.json_dir, filename) for filename in os.listdir(self.json_dir)]

        # Storage for the parsed output
        self.parsed = None

        # Parse based on output format
        if self.output == "pitch_duration_tokens":
            self.parse_pitch_duration_tokens()
        elif self.output == "midi_ticks":
            self.ticks = 24
            self.parse_midi_ticks()
        else:
            raise Exception("Unrecognized output format.")

    def verify_directories(self, root_dir, json_dir, song_dir, dataset_dir):
        """
        Ensures that all input/output directories exist.
        :param root_dir: the project directory
        :param json_dir: the directory containing the JSON formatted MusicXML
        :param song_dir: the directory to save parsed individual songs
        :param dataset_dir: the directory to save full datasets
        :return:
        """
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

    def parse_metadata(self, filename, song_dict):
        """
        Given the JSON input as a dict, returns.
        :param filename: name of the file from which song_dict is loaded, used as a backup for title/artist
        :param song_dict:  a song dict in the format created by src/processing/conversion/xml_to_json.py
        :return: an object containing metadata for the song in the following format:
        {
          title,
          artist,
          key,
          time_signature
        }
        """
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

    def parse_midi_ticks(self):
        """
        Parses the JSON formatted MusicXML into MIDI ticks format.
        :return: a list of parsed songs in the following format:
        {
          metadata: {
            title,
            artist,
            key,
            time_signature,
            ticks_per_measure
          },
          measures: [[{
            harmony: {
              root,
              pitch_classes
            },
            ticks (num_ticks x 38) (F3-E6)
          }], ...]
        }
        or, None, if a song has more than one key
        """
        songs = []

        for filename in self.json_paths:
            # Load song dict
            try:
                song_dict = json.load(open(filename))
            except:
                print("Unable to load JSON for %s" % filename)
                continue

            print("Parsing %s" % op.basename(filename))

            # Get song metadata
            metadata = self.parse_metadata(filename, song_dict)

            # Add ticks per measure to metadata
            time_signature = [int(n) for n in metadata["time_signature"].split("/")]
            metadata["ticks_per_measure"] = int(time_signature[0] * (4 / time_signature[1]) * self.ticks)

            # Get song divisions
            divisions = get_divisions(song_dict)

            # Calculate scale factor from divisions to ticks
            #  i.e. scale_factor * divisions = num_ticks
            scale_factor = self.ticks / divisions

            if scale_factor > 1:
                raise Exception("Error: MusicXML has lower resolution than desired MIDI ticks.")

            # Parse each measure
            measures = []
            last_harmony = {}
            for measure in song_dict["part"]["measures"]:
                parsed_measure, last_harmony = self.parse_measure_midi_ticks(measure, scale_factor, last_harmony)
                measures.append(parsed_measure)

            songs.append({
                "metadata": metadata,
                "measures": measures
            })

        self.parsed = songs

    def parse_measure_midi_ticks(self, measure, scale_factor, last_harmony):
        """
        For a measure, returns a set of ticks grouped by associated harmony in.
        :param measure: a measure dict in the format created by src/processing/conversion/xml_to_json.py
        :param scale_factor: the scale factor between XML divisions and midi ticks
        :param last_harmony: a reference to the last harmony used in case a measure has none
        :return: a dict containing a list of groups that contains a harmony and the midi ticks associated with that harmony
        """
        parsed_measure = {
            "groups": []
        }
        new_last_harmony = last_harmony

        num_ticks = 0

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

            num_ticks += len(ticks_by_note)

            if not group["harmony"]:
                parsed_measure["groups"].append({
                    "harmony": last_harmony,
                    "ticks": ticks_by_note
                })
            else:
                harmony = Harmony(group["harmony"])
                harmony_dict = {
                    "root": harmony.get_one_hot_root(),
                    "pitch_classes": harmony.get_seventh_pitch_classes_binary()
                }
                new_last_harmony = harmony_dict
                parsed_measure["groups"].append({
                    "harmony": harmony_dict,
                    "ticks": ticks_by_note
                })

        parsed_measure["num_ticks"] = num_ticks

        return parsed_measure, new_last_harmony

    def parse_pitch_duration_tokens(self):
        """
        Parses the JSON formatted MusicXML into pitch duration tokens format.
        :return: a list of parsed songs in the following format:
        {
          metadata: {
            title,
            artist,
            key,
            time_signature
          },
          measures: [{
            groups: {
              harmony: {
                root,
                pitch_classes
              },
              pitch_numbers,
              duration_tags,
              bar_position
            }, ...
          }, ...]
        }
        or, None, if a song has more than one key
        """
        songs = []

        for filename in self.json_paths:
            # Load song dict
            try:
                song_dict = json.load(open(filename))
            except:
                print("Unable to load JSON for %s" % filename)
                continue

            print("Parsing %s" % op.basename(filename))

            # Get song metadata
            metadata = self.parse_metadata(filename, song_dict)

            # Get song divisions
            divisions = get_divisions(song_dict)

            # Parse each measure
            measures = []
            for measure in song_dict["part"]["measures"]:
                parsed_measure = self.parse_measure_pitch_duration_tokens(measure, divisions)
                measures.append(parsed_measure)

            # try to fill in harmonies somewhat naively
            max_harmonies_per_measure = 0
            for i, measure in enumerate(measures):
                for j, group in enumerate(measure["groups"]):
                    if not group["harmony"]:
                        if i == 0 and j == 0:
                            for after_group in measures[i]["groups"][j + 1:]:
                                if after_group["harmony"]:
                                    measure["groups"][j]["harmony"] = after_group["harmony"]
                                    break
                            for after_measure in measures[i + 1:]:
                                for after_measure_group in after_measure["groups"]:
                                    if after_measure_group["harmony"]:
                                        measure["groups"][j]["harmony"] = after_measure_group["harmony"]
                                        break
                        elif i == 0:
                            for before_group in measure["groups"][j - 1::-1]:
                                if before_group["harmony"]:
                                    measure["groups"][j]["harmony"] = before_group["harmony"]
                                    break
                        else:
                            for before_group in measure["groups"][j - 1::-1]:
                                if before_group["harmony"]:
                                    measure["groups"][j]["harmony"] = before_group["harmony"]
                                    break
                            for before_measure in measures[i - 1::-1]:
                                for before_measure_group in before_measure["groups"]:
                                    if before_measure_group["harmony"]:
                                        measure["groups"][j]["harmony"] = before_measure_group["harmony"]
                                        break
                max_harmonies_per_measure = max(len(measure["groups"]), max_harmonies_per_measure)

            if max_harmonies_per_measure == 0:
                continue

            songs.append({
                "metadata": metadata,
                "measures": measures
            })

        self.parsed = songs

    def parse_measure_pitch_duration_tokens(self, measure, divisions):
        """
        Parses a measure according to the pitch duration token parsing process.
        :param measure: a measure dict in the format created by src/processing/conversion/xml_to_json.py
        :param divisions: the number of divisions a quarter note is split into in this song's MusicXML representation
        :return: a dict containing a list of groups containing the pitch numbers, durations, and bar positions for each
                harmony in a measure
        """
        parsed_measure = {
            "groups": []
        }

        for group in measure["groups"]:
            parsed_group = {
                "harmony": {},
                "pitch_numbers": [],
                "duration_tags": [],
                "bar_position": []
            }
            harmony = Harmony(group["harmony"])
            parsed_group["harmony"]["root"] = harmony.get_one_hot_root()
            parsed_group["harmony"]["pitch_classes"] = harmony.get_seventh_pitch_classes_binary()
            dur_ticks_list = []
            for note_dict in group["notes"]:
                # want monophonic, so we'll just take the top note
                if "chord" in note_dict.keys() or "grace" in note_dict.keys():
                    continue
                else:
                    pitch_num, dur_tag, dur_ticks = self.parse_note_pitch_duration_tokens(note_dict, divisions)
                    parsed_group["pitch_numbers"].append(pitch_num)
                    parsed_group["duration_tags"].append(dur_tag)
                    dur_ticks_list.append(dur_ticks)
            dur_ticks_list = [sum(dur_ticks_list[:i]) for i in range(len(dur_ticks_list))]
            dur_to_next_bar = [4 * divisions - dur_ticks for dur_ticks in dur_ticks_list]
            parsed_group["bar_position"] = dur_to_next_bar
            parsed_measure["groups"].append(parsed_group)
        return parsed_measure

    def parse_note_pitch_duration_tokens(self, note_dict, divisions):
        """
        Parses a note according to the pitch duration token parsing process.
        :param note_dict: a note dict in the format created by src/processing/conversion/xml_to_json.py
        :param divisions: the number of divisions a quarter note is split into in this song's MusicXML representation
        :return: a number reflecting the pitch, a string representing the duration, and the duration in divisions
        """
        if "rest" in note_dict.keys():
            pitch_num = NOTES_MAP["rest"]
        elif "pitch" in note_dict.keys():
            note_string = note_dict["pitch"]["step"]["text"]
            if "alter" in note_dict["pitch"].keys():
                note_string += (lambda x: "b" if -1 else ("#" if 1 else ""))(
                    note_dict["pitch"]["alter"]["text"])
            octave = int(note_dict["pitch"]["octave"]["text"])
            pitch_num = (octave + 1) * 12 + NOTES_MAP[note_string]

        dur_tag, dur_ticks = get_note_duration(note_dict, divisions)
        return pitch_num, dur_tag, dur_ticks

    def save_parsed(self, transpose=False):
        """
        Saves the parsed songs as .pkl to song_dir.
        :param transpose: if True, transposes and saves each song in all 12 keys.
        :return: None
        """
        # Ensure parsing has happened
        if not self.parsed:
            print("Nothing has been parsed.")
            return

        for song in self.parsed:
            if transpose:
                for steps in range(-6, 6):
                    if self.output == "pitch_duration_tokens":
                        transposed = transpose_song_pitch_duration_tokens(song, steps)
                    else:
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


def get_key(song_dict):
    """
    Fetches a key from the JSON representation of MusicXML for a song.

    I know from analysis that the only keys in my particular dataset are major
    and minor.
    :param song_dict: a song dict in the format created by src/processing/conversion/xml_to_json.py
    :return: ff there's only one key: (key, False), if there's more than one key: (None, True)
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


def get_divisions(song_dict):
    """
    Fetch the divisions per quarter note in the JSON represented MusicXML
    :param song_dict: a song dict in the format created by src/processing/conversion/xml_to_json.py
    :return: the number of divisions a quarter note is split into in this song's MusicXML representation
    """
    return int(song_dict["part"]["measures"][0]["attributes"]["divisions"]["text"])


def get_note_index(note):
    """
    Fetches an index value for encoding a note's pitch for MIDI tick pitch formatting.
    :param note: a note dict in the format created by src/processing/conversion/xml_to_json.py
    :return: an index representing a note value, where 0 is F3, 35 is E6, and 36 is 'rest'
    """
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


def transpose_song_midi_ticks(song, steps):
    """
    Transposes a song that has been parsed into MIDI ticks form.
    :param song: a song parsed into MIDI ticks form
    :param steps: a positive or negative number representing how many steps to transpose
    :return: a transposed song in MIDI ticks form
    """

    sign = lambda x: (1, -1)[x < 0]

    transposed = deepcopy(song)
    print("transposing by %i" % steps)
    transposed["transposition"] = steps

    for measure in transposed["measures"]:
        for group in measure["groups"]:
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


def transpose_ticks(ticks, direction):
    """
    Transposes a MIDI tick array one step up or down.
    :param ticks: a one-hot array representing a pitch F3-E6 or rest
    :param direction: either -1 or 1, representing the direction to transpose
    :return: a transposed MIDI tick array
    """
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


def get_note_duration(note_dict, divisions=24):
    """
    Fetches the duration of a note in JSON formatted MusicXML.
    :param note_dict: a note dict in the format created by src/processing/conversion/xml_to_json.py
    :param divisions: the number of divisions a quarter note is split into in this song's MusicXML representation
    :return: a string representing the duration, and the duration in divisions
    """
    dur_dict = {'double': divisions * 8, 'whole': divisions * 4, 'half': divisions * 2,
                'quarter': divisions, '8th': divisions / 2, '16th': divisions / 4, '32nd': divisions / 8}

    if "duration" not in note_dict.keys():
        note_dur = -1
    else:
        note_dur = float(note_dict["duration"]["text"])

    if "type" in note_dict.keys():
        note_type = note_dict["type"]["text"]
        if note_type == "eighth":
            note_type = "8th"
        label = note_type
        if note_dur == (3 * dur_dict[note_type] / 2):
            label = '-'.join([label, 'dot'])
        elif note_dur == (dur_dict[note_type] * 2 / 3):
            label = '-'.join([label, 'triplet'])
        elif note_dur != dur_dict[note_type]:
            print("Undefined %s duration. Entering as regular %s." % (note_type, note_type))
    elif note_dur == dur_dict["double"]:
        label = "double"
    elif note_dur == (3 * dur_dict["double"] / 2):
        label = "double-dot"
    elif note_dur == (2 * dur_dict["double"] / 3):
        label = "double-triplet"
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
    elif note_dur == dur_dict["8th"]:
        label = "8th"
    elif note_dur == (3 * dur_dict["8th"] / 2):
        label = "8th-dot"
    elif note_dur == (2 * dur_dict["8th"] / 3):
        label = "8th-triplet"
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
        label = "32nd-triplet"
    else:
        print("Undefined duration %.2f. Labeling 'other'." % (note_dur / divisions))
        label = "other"
    return DURATIONS_MAP[label], note_dur


def transpose_song_pitch_duration_tokens(song, steps):
    """
    Transposes a song in the pitch duration token format.
    :param song: a song in pitch duration token format
    :param steps: positive or negative integer number of steps to transpose
    :return: a transposed song in pitch duration token format
    """
    transposed = deepcopy(song)
    transposed["transposition"] = steps
    print("transposing by %i" % steps)
    for i, measure in enumerate(transposed["measures"]):
        for j, group in enumerate(measure["groups"]):
            group["pitch_numbers"] = [
                (lambda n: n + steps if n != NOTES_MAP["rest"] else n)(pn)
                for pn in group["pitch_numbers"]]
            group["harmony"]["root"] = rotate(group["harmony"]["root"], steps)
            group["harmony"]["pitch_classes"] = rotate(group["harmony"]["pitch_classes"], steps)
            transposed["measures"][i]["groups"][j] = group
    return transposed


def rotate(l, x):
    """
    Rotates a list a given number of steps
    :param l: the list to rotate
    :param x: a positive or negative integer steps to rotate
    :return: the rotated list
    """
    return l[-x:] + l[:-x]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="pitch_duration_tokens",
                        help="The output format of the processed data.")
    parser.add_argument("--transpose", default=False,
                        help="Whether or not to transpose the parsed songs into all 12 keys.")
    args = parser.parse_args()

    p = Parser(output=args.output)
    p.save_parsed(transpose=args.transpose)
