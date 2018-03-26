import copy
from chord_labels import parse_chord

KIND_LABELS = {"major": "maj",
               "major-seventh": "maj7",
               "minor": "",
               "minor-seventh": "m7",
               "augmented": "aug",
               "augmented-seventh": "aug7",
               "diminished": "dim",
               "half-diminished": "m7b5",
               "diminished-seventh": "dim7",
               "dominant": "7",
               "major-minor": "min(maj7)",
               "major-sixth": "6",
               "minor-sixth": "m6",
               "dominant-ninth": "9",
               "major-ninth": "maj9",
               "minor-ninth": "m9",
               "dominant-11th": "11",
               "major-11th": "maj11",
               "minor-11": "m11",
               "dominant-13th": "13",
               "major-13th": "maj13",
               "minor-13th": "m13",
               "suspended-second": "sus2",
               "suspended-fourth": "sus4"}

KIND_COMPONENTS = {"major": {3: 0, 5: 0},
                   "major-seventh": {3: 0, 5: 0, 7: 0},
                   "minor": {3: -1, 5: 0},
                   "minor-seventh": {3: -1, 5: 0, 7: -1},
                   "augmented": {3: 0, 5: 1}, 
                   "augmented-seventh": {3: 0, 5: 1, 7: -1},
                   "diminished": {3: -1, 5: -1},
                   "half-diminished": {3: -1, 5: -1, 7: -1},
                   "diminished-seventh": {3: -1, 5: -1, 7: -2},
                   "dominant": {3: 0, 5: 0, 7: -1},
                   "major-minor": {3: -1, 5: 0, 7: 0},
                   "major-sixth": {3: 0, 5: 0, 6: 0},
                   "minor-sixth": {3: -1, 5: 0, 6: 0},
                   "dominant-ninth": {3: 0, 5: 0, 7: -1, 9: 0},
                   "major-ninth": {3: 0, 5: 0, 7: 0, 9: 0},
                   "minor-ninth": {3: -1, 5: 0, 7: -1, 9: 0},
                   "dominant-11th": {3: 0, 5: 0, 7: -1, 9: 0, 11: 0},
                   "major-11th": {3: 0, 5: 0, 7: 0, 9: 0, 11: 0},
                   "minor-11": {3: -1, 5: 0, 7: -1, 9: 0, 11: 0},
                   "dominant-13th": {3: 0, 5: 0, 7: -1, 9: 0, 11: 0, 13: 0},
                   "major-13th": {3: 0, 5: 0, 7: 0, 9: 0, 11: 0, 13: 0},
                   "minor-13th": {3: -1, 5: 0, 7: -1, 9: 0, 11: 0, 13: 0},
                   "suspended-second": {2: 0, 5: 0},
                   "suspended-fourth": {4: 0, 5: 0}}

class Harmony(object):
    _chord_symbol = None
    _harte_notation = None
    _pitch_classes = None
    _pitch_classes_binary = None

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

    def _get_root_label(self, root_dict):
        retval = root_dict["root-step"]["text"]
        if "root-alter" in root_dict.keys():
            retval += self._get_alter_label(int(root_dict["root-alter"]["text"]))
        return retval

    def _get_degree_label(self, degree_dict):
        retval = degree_dict["degree-value"]["text"]
        if "degree-alter" in degree_dict.keys():
            alter_label = self._get_alter_label(int(degree_dict["degree-alter"]["text"])) 
            retval = alter_label + retval
        return retval

    def _get_bass_label(self, bass_dict):
        retval = bass_dict["bass-step"]["text"]
        if "bass-alter" in bass_dict.keys():
            retval += self._get_alter_label(int(bass_dict["bass-alter"]["text"]))
        return retval

    def get_chord_symbol(self):
        if self._chord_symbol is None:
            self._chord_symbol = self._get_root_label(self.harmony_dict['root'])
            self._chord_symbol += KIND_LABELS[self.harmony_dict["kind"]["text"]]

            if self.harmony_dict["degrees"]:
                for degree in self.harmony_dict["degrees"]:
                    self._chord_symbol += self._get_degree_label(degree)

            if self.harmony_dict["bass"]:
                self._chord_symbol += "/%s" % (self._get_bass_label(self.harmony_dict["bass"]))

        return self._chord_symbol

    def get_harte_notation(self):
        if self._harte_notation is None:
            self._harte_notation = "%s:" % self._get_root_label(self.harmony_dict['root'])
            components = copy.deepcopy(KIND_COMPONENTS[self.harmony_dict["kind"]["text"]])

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

    def get_pitch_classes(self):
        if self._pitch_classes is None:
            self._pitch_classes = parse_chord(self.get_harte_notation()).tones
        return self._pitch_classes

    def get_pitch_classes_binary(self):
        if self._pitch_classes_binary is None:
            self._pitch_classes_binary = parse_chord(self.get_harte_notation()).tones_binary
        return self._pitch_classes_binary
