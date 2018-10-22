import copy
import numpy as np
import chord_labels as cl

ROOT_DICT = {'B#': 0, 'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
        'Fb': 4, 'E#': 5, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 
        'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11}
             

CHORD_DICT = {"major": {"label": "maj", 
                         "components": {3: 0, 5: 0},
                         "triad_label": "maj",
                         "triad_components": {3: 0, 5: 0},
                         "seventh_label": "maj",
                         "seventh_components": {3: 0, 5: 0}},
               "major-seventh": {"label": "maj7",
                                 "components": {3: 0, 5: 0, 7: 0},
                                 "triad_label": "maj",
                                 "triad_components": {3: 0, 5: 0},
                                 "seventh_label": "maj7",
                                 "seventh_components": {3: 0, 5: 0, 7: 0}},
               "minor": {"label": "m",
                         "components": {3: -1, 5: 0},
                         "triad_label": "m",
                         "triad_components": {3: -1, 5: 0},
                         "seventh_label": "m",
                         "seventh_components": {3: -1, 5: 0}},
               "minor-seventh": {"label": "m7",
                                 "components": {3: -1, 5: 0, 7: -1},
                                 "triad_label": "m",
                                 "triad_components": {3: -1, 5: 0},
                                 "seventh_label": "m7",
                                 "seventh_components": {3: -1, 5: 0, 7: -1}},
               "augmented": {"label": "aug",
                             "components": {3: 0, 5: 1},
                             "triad_label": "aug",
                             "triad_components": {3: 0, 5: 1},
                             "seventh_label": "aug7",
                             "seventh_components": {3: 0, 5: 1, 7: -1}},
               "augmented-seventh": {"label": "aug7",
                                     "components": {3: 0, 5: 1, 7: -1},
                                     "triad_label": "aug",
                                     "triad_components": {3: 0, 5: 1},
                                     "seventh_label": "aug7",
                                     "seventh_components": {3: 0, 5: 1, 7: -1}},
               "augmented-ninth": {"label": "aug9", 
                                   "components": {3: 0, 5: 1, 7: -1, 9: 0},
                                   "triad_label": "aug",
                                   "triad_components": {3: 0, 5: 1},
                                   "seventh_label": "aug7",
                                   "seventh_components": {3: 0, 5: 1, 7: -1}},
               "diminished": {"label": "dim",
                              "components": {3: -1, 5: -1},
                              "triad_label": "dim",
                              "triad_components": {3: -1, 5: -1},
                              "seventh_label": "m7b5",
                              "seventh_components": {3: -1, 5: -1, 7: -1}},
               "half-diminished": {"label": "m7b5",
                                   "components": {3: -1, 5: -1, 7: -1},
                                   "triad_label": "dim",
                                   "triad_components": {3: -1, 5: -1},
                                   "seventh_label": "m7b5",
                                   "seventh_components": {3: -1, 5: -1, 7: -1}},
               "diminished-seventh": {"label": "dim7",
                                      "components": {3: -1, 5: -1, 7: -2},
                                      "triad_label": "dim",
                                      "triad_components": {3: -1, 5: -1},
                                      "seventh_label": "m7b5",
                                      "seventh_components": {3: -1, 5: -1, 7: -1}},
               "dominant": {"label": "7",
                            "components": {3: 0, 5: 0, 7: -1},
                            "triad_label": "maj",
                            "triad_components": {3: 0, 5: 0},
                            "seventh_label": "7",
                            "seventh_components": {3: 0, 5: 0, 7: -1}},
               "dominant-seventh": {"label": "7",
                                    "components": {3: 0, 5: 0, 7: -1},
                                    "triad_label": "maj",
                                    "triad_components": {3: 0, 5: 0},
                                    "seventh_label": "7",
                                    "seventh_components": {3: 0, 5: 0, 7: -1}},
               "7": {"label": "7",
                     "components": {3: 0, 5: 0, 7: -1},
                     "triad_label": "maj",
                     "triad_components": {3: 0, 5: 0},
                     "seventh_label": "7",
                     "seventh_components": {3: 0, 5: 0, 7: -1}},
               "minor-major": {"label": "m(maj7)",
                               "components": {3: -1, 5: 0, 7: 0},
                               "triad_label": "m",
                               "triad_components": {3: -1, 5: 0},
                               "seventh_label": "m",
                               "seventh_components": {3: -1, 5: 0}},
               "major-minor": {"label": "m(maj7)",
                               "components": {3: -1, 5: 0, 7: 0},
                               "triad_label": "m",
                               "triad_components": {3: -1, 5: 0},
                               "seventh_label": "m",
                               "seventh_components": {3: -1, 5: 0}},
               "major-sixth": {"label": "6",
                               "components": {3: 0, 5: 0, 6: 0},
                               "triad_label": "maj",
                               "triad_components": {3: 0, 5: 0},
                               "seventh_label": "maj",
                               "seventh_components": {3: 0, 5: 0}},
               "minor-sixth": {"label": "m6",
                               "components": {3: -1, 5: 0, 6: 0},
                               "triad_label": "m",
                               "triad_components": {3: -1, 5: 0},
                               "seventh_label": "m",
                               "seventh_components": {3: -1, 5: 0}},
               "dominant-ninth": {"label": "9",
                                  "components": {3: 0, 5: 0, 7: -1, 9: 0},
                                  "triad_label": "maj",
                                  "triad_components": {3: 0, 5: 0},
                                  "seventh_label": "7",
                                  "seventh_components": {3: 0, 5: 0, 7: -1}},
               "major-ninth": {"label": "maj9",
                               "components": {3: 0, 5: 0, 7: 0, 9: 0},
                               "triad_label": "maj",
                               "triad_components": {3: 0, 5: 0},
                               "seventh_label": "maj7",
                               "seventh_components": {3: 0, 5: 0, 7: 0}},
               "minor-ninth": {"label": "m9",
                               "components": {3: -1, 5: 0, 7: -1, 9:0},
                               "triad_label": "m", 
                               "triad_components": {3: -1, 5: 0},
                               "seventh_label": "m7",
                               "seventh_components": {3: -1, 5: 0, 7: -1}},
                "maj69": {"label": "maj69",
                          "components": {3: 0, 5: 0, 6: 0, 9: 0},
                          "triad_label": "maj",
                          "triad_components": {3: 0, 5: 0},
                          "seventh_label": "maj",
                          "seventh_components": {3: 0, 5: 0}},
               "dominant-11th": {"label": "11",
                                 "components": {3: 0, 5: 0, 7: -1, 9: 0, 11: 0},
                                 "triad_label": "maj",
                                 "triad_components": {3: 0, 5: 0},
                                 "seventh_label": "7",
                                 "seventh_components": {3: 0, 5: 0, 7: -1}},
               "major-11th": {"label": "maj11",
                              "components": {3: 0, 5: 0, 7: 0, 9: 0, 11: 0},
                              "triad_label": "maj",
                              "triad_components": {3: 0, 5: 0},
                              "seventh_label": "maj7",
                              "seventh_components": {3: 0, 5: 0, 7: 0}},
               "minor-11th": {"label": "m11",
                              "seventh_label": "m7",
                              "triad_label": "m",
                              "triad_components": {3: -1, 5: 0},
                              "components": {3: -1, 5: 0, 7: -1, 9: 0, 11: 0},
                              "seventh_components": {3: -1, 5: 0, 7: -1}},
               "dominant-13th": {"label": "13",
                                 "components": {3: 0, 5: 0, 7: -1, 9: 0, 11: 0, 13: 0},
                                 "triad_label": "maj",
                                 "triad_components": {3: 0, 5: 0},
                                 "seventh_label": "7",
                                 "seventh_components": {3: 0, 5: 0, 7: -1}},
               "major-13th": {"label": "maj13",
                              "components": {3: 0, 5: 0, 7: 0, 9: 0, 11: 0, 13: 0},
                              "triad_label": "maj",
                              "triad_components": {3: 0, 5: 0},
                              "seventh_label": "maj7",
                              "seventh_components": {3: 0, 5: 0, 7: 0}},
               "minor-13th": {"label": "m13",
                              "components": {3: -1, 5: 0, 7: -1, 9: 0, 11: 0, 13: 0},
                              "triad_label": "m",
                              "triad_components": {3: -1, 5: 0},
                              "seventh_label": "m7",
                              "seventh_components": {3: -1, 5: 0, 7: -1}},
               "suspended-second": {"label": "sus2",
                                    "components": {2: 0, 5: 0},
                                    "triad_label": "sus2",
                                    "triad_components": {2: 0, 5: 0},
                                    "seventh_label": "sus2",
                                    "seventh_components": {2: 0, 5: 0}},
               "suspended-fourth": {"label": "sus4",
                                    "components": {4: 0, 5: 0},
                                    "triad_label": "sus4",
                                    "triad_components": {4: 0, 5: 0},
                                    "seventh_label": "sus4",
                                    "seventh_components": {4: 0, 5: 0}}
}

class Harmony(object):
    _chord_root = None
    _chord_symbol = None
    _harte_notation = None
    _pitch_classes = None
    _pitch_classes_binary = None
    _triad_chord_symbol = None
    _triad_harte_notation = None
    _triad_pitch_classes = None
    _triad_pitch_classes_binary = None
    _seventh_chord_symbol = None
    _seventh_harte_notation = None
    _seventh_pitch_classes = None
    _seventh_pitch_classes_binary = None

    def __init__(self, harmony_dict):
        self.harmony_dict = harmony_dict

    def _get_alter_label(self, alter):
        label = ''
        while int(alter) < 0:
            label += 'b'
            alter += 1
        while int(alter) > 0:
            label += '#'
            alter -= 1
        return label

    def _get_root_label(self):
        if self.harmony_dict is None:
            return None

        root_dict = self.harmony_dict["root"]
        if self._chord_root is None:
            self._chord_root = root_dict["root-step"]["text"]
            if "root-alter" in root_dict.keys():
                self._chord_root += self._get_alter_label(int(root_dict["root-alter"]["text"]))
        return self._chord_root

    def get_one_hot_root(self):
        vector = [0]*12
        if self.harmony_dict is not None:
            vector[ROOT_DICT[self._get_root_label()]] = 1
        return vector

    def _get_degree_label(self, degree_dict):
        degree = degree_dict["degree-value"]["text"]
        if "degree-alter" in degree_dict.keys():
            alter_label = self._get_alter_label(int(degree_dict["degree-alter"]["text"])) 
        return (int(degree), alter_label)

    def _get_bass_label(self, bass_dict):
        retval = bass_dict["bass-step"]["text"]
        if "bass-alter" in bass_dict.keys():
            retval += self._get_alter_label(int(bass_dict["bass-alter"]["text"]))
        return retval

    def get_chord_symbol(self):
        if self.harmony_dict is None:
            return ""

        if self._chord_symbol is None:
            self._chord_symbol = self._get_root_label()
            kind = self.harmony_dict["kind"]["text"]
            if kind in CHORD_DICT.keys():
                self._chord_symbol += CHORD_DICT[kind]["label"]
            else:
                print("Unknown chord kind: %s, igoring." %
                        str(self.harmony_dict["kind"]["text"]))

            if self.harmony_dict["degrees"]:
                degree_labels = []
                for degree in self.harmony_dict["degrees"]:
                    degree_labels.append(self._get_degree_label(degree))
                degree_labels.sort(key=lambda tup: tup[0])
                degree_labels = ["%s%i" % (dl_tup[1], dl_tup[0]) for dl_tup in degree_labels]
                self._chord_symbol += "(%s)" % ",".join(degree_labels)

            if "bass" in self.harmony_dict.keys():
                self._chord_symbol += "/%s" % (self._get_bass_label(self.harmony_dict["bass"]))

        return self._chord_symbol

    def get_triad_chord_symbol(self):
        if self.harmony_dict is None:
            return ""

        if self._triad_chord_symbol is None:
            self._triad_chord_symbol = self._get_root_label()
            kind = self.harmony_dict["kind"]["text"]
            if kind in CHORD_DICT.keys():
                self._triad_chord_symbol += CHORD_DICT[kind]["triad_label"]
            else:
                print("Unknown chord kind: %s, igoring." %
                        str(self.harmony_dict["kind"]["text"]))

        return self._triad_chord_symbol

    def get_seventh_chord_symbol(self):
        if self.harmony_dict is None:
            return ""

        if self._seventh_chord_symbol is None:
            self._seventh_chord_symbol = self._get_root_label()
            kind = self.harmony_dict["kind"]["text"]
            if kind in CHORD_DICT.keys():
                self._seventh_chord_symbol += CHORD_DICT[kind]["seventh_label"]
            else:
                print("Unknown chord kind: %s, igoring." %
                        str(self.harmony_dict["kind"]["text"]))

        return self._seventh_chord_symbol

    def get_harte_notation(self):
        if self.harmony_dict is None:
            return ""

        if self._harte_notation is None:
            self._harte_notation = "%s:" % self._get_root_label()
            kind = self.harmony_dict["kind"]["text"]
            if kind in CHORD_DICT.keys():
                components = copy.deepcopy(CHORD_DICT[kind]["components"])
            else:
                components = copy.deepcopy(CHORD_DICT["major"]["components"])
                print("Unknown chord kind: %s, using \"major\"." %
                        str(self.harmony_dict["kind"]["text"]))

            if self.harmony_dict["degrees"]:
                for degree in self.harmony_dict["degrees"]:
                    value = int(degree["degree-value"]["text"])
                    if "degree-alter" in degree.keys():
                        alter = int(degree["degree-alter"]["text"])
                        components[value] = alter
                    else:
                        components[alter] = 0

            component_strings = []
            for i in range(2, 14):
                if i in components.keys():
                    alter_label = self._get_alter_label(components[i])
                    component_strings.append("%s%i" % (alter_label, i))
            self._harte_notation += "(%s)" % (",".join(component_strings))

            # ignoring bass notes for now. I'm not sure how to implement that in
            # a not super complex way.

        return self._harte_notation

    def get_triad_harte_notation(self):
        if self.harmony_dict is None:
            return ""

        if self._triad_harte_notation is None:
            self._triad_harte_notation = "%s:" % self._get_root_label()
            kind = self.harmony_dict["kind"]["text"]
            if kind in CHORD_DICT.keys():
                components = copy.deepcopy(CHORD_DICT[kind]["triad_components"])
            else:
                components = copy.deepcopy(CHORD_DICT["major"]["triad_components"])
                print("Unknown chord kind: %s, using \"major\"." %
                        str(self.harmony_dict["kind"]["text"]))

            component_strings = []
            for i in range(2, 14):
                if i in components.keys():
                    alter_label = self._get_alter_label(components[i])
                    component_strings.append("%s%i" % (alter_label, i))
            self._triad_harte_notation += "(%s)" % (",".join(component_strings))

        return self._triad_harte_notation

    def get_seventh_harte_notation(self):
        if self.harmony_dict is None:
            return ""

        if self._seventh_harte_notation is None:
            self._seventh_harte_notation = "%s:" % self._get_root_label()
            kind = self.harmony_dict["kind"]["text"]
            if kind in CHORD_DICT.keys():
                components = copy.deepcopy(CHORD_DICT[kind]["seventh_components"])
            else:
                components = copy.deepcopy(CHORD_DICT["major"]["seventh_components"])
                print("Unknown chord kind: %s, using \"major\"." %
                        str(self.harmony_dict["kind"]["text"]))

            component_strings = []
            for i in range(2, 14):
                if i in components.keys():
                    alter_label = self._get_alter_label(components[i])
                    component_strings.append("%s%i" % (alter_label, i))
            self._seventh_harte_notation += "(%s)" % (",".join(component_strings))

        return self._seventh_harte_notation

    def get_pitch_classes(self):
        if self.harmony_dict is None:
            return [] 

        if self._pitch_classes is None:
            self._pitch_classes = cl.parse_chord(self.get_harte_notation()).tones
        return self._pitch_classes

    def get_triad_pitch_classes(self):
        if self.harmony_dict is None:
            return [] 

        if self._triad_pitch_classes is None:
            self._triad_pitch_classes = cl.parse_chord(self.get_triad_harte_notation()).tones
        return self._triad_pitch_classes

    def get_seventh_pitch_classes(self):
        if self.harmony_dict is None:
            return [] 

        if self._seventh_pitch_classes is None:
            self._seventh_pitch_classes = cl.parse_chord(self.get_seventh_harte_notation()).tones
        return self._seventh_pitch_classes

    def get_pitch_classes_binary(self):
        if self.harmony_dict is None:
            return [0]*12

        if self._pitch_classes_binary is None:
            self._pitch_classes_binary = cl.parse_chord(self.get_harte_notation()).tones_binary
        return self._pitch_classes_binary

    def get_triad_pitch_classes_binary(self):
        if self.harmony_dict is None:
            return [0]*12

        if self._triad_pitch_classes_binary is None:
            self._triad_pitch_classes_binary = cl.parse_chord(self.get_triad_harte_notation()).tones_binary
        return self._triad_pitch_classes_binary

    def get_seventh_pitch_classes_binary(self):
        if self.harmony_dict is None:
            return [0]*12

        if self._seventh_pitch_classes_binary is None:
            self._seventh_pitch_classes_binary = cl.parse_chord(self.get_seventh_harte_notation()).tones_binary
        return self._seventh_pitch_classes_binary
