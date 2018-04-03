import copy
from chord_labels import parse_chord

CHORD_DICT = {"major": {"label": "maj", 
                         "components": {3: 0, 5: 0},
                         "triad_label": "maj",
                         "triad_components": {3: 0, 5: 0},
                         "simple_label": "maj",
                         "simple_components": {3: 0, 5: 0}},
               "major-seventh": {"label": "maj7",
                                 "components": {3: 0, 5: 0, 7: 0},
                                 "triad_label": "maj",
                                 "triad_components": {3: 0, 5: 0},
                                 "simple_label": "maj7",
                                 "simple_components": {3: 0, 5: 0, 7: 0}},
               "minor": {"label": "m",
                         "components": {3: -1, 5: 0},
                         "triad_label": "m",
                         "triad_components": {3: -1, 5: 0},
                         "simple_label": "m",
                         "simple_components": {3: -1, 5: 0}},
               "minor-seventh": {"label": "m7",
                                 "components": {3: -1, 5: 0, 7: -1},
                                 "triad_label": "m",
                                 "triad_components": {3: -1, 5: 0},
                                 "simple_label": "m7",
                                 "simple_components": {3: -1, 5: 0, 7: -1}},
               "augmented": {"label": "aug",
                             "components": {3: 0, 5: 1},
                             "triad_label": "aug",
                             "triad_components": {3: 0, 5: 1},
                             "simple_label": "aug7",
                             "simple_components": {3: 0, 5: 1, 7: -1}},
               "augmented-seventh": {"label": "aug7",
                                     "components": {3: 0, 5: 1, 7: -1},
                                     "triad_label": "aug",
                                     "triad_components": {3: 0, 5: 1},
                                     "simple_label": "aug7",
                                     "simple_components": {3: 0, 5: 1, 7: -1}},
               "augmented-ninth": {"label": "aug9", 
                                   "components": {3: 0, 5: 1, 7: -1, 9: 0},
                                   "triad_label": "aug",
                                   "triad_components": {3: 0, 5: 1},
                                   "simple_label": "aug7",
                                   "simple_components": {3: 0, 5: 1, 7: -1}},
               "diminished": {"label": "dim",
                              "components": {3: -1, 5: -1},
                              "triad_label": "dim",
                              "triad_components": {3: -1, 5: -1},
                              "simple_label": "m7b5",
                              "simple_components": {3: -1, 5: -1, 7: -1}},
               "half-diminished": {"label": "m7b5",
                                   "components": {3: -1, 5: -1, 7: -1},
                                   "triad_label": "dim",
                                   "triad_components": {3: -1, 5: -1},
                                   "simple_label": "m7b5",
                                   "simple_components": {3: -1, 5: -1, 7: -1}},
               "diminished-seventh": {"label": "dim7",
                                      "components": {3: -1, 5: -1, 7: -2},
                                      "triad_label": "dim",
                                      "triad_components": {3: -1, 5: -1},
                                      "simple_label": "m7b5",
                                      "simple_components": {3: -1, 5: -1, 7: -1}},
               "dominant": {"label": "7",
                            "components": {3: 0, 5: 0, 7: -1},
                            "triad_label": "maj",
                            "triad_components": {3: 0, 5: 0},
                            "simple_label": "7",
                            "simple_components": {3: 0, 5: 0, 7: -1}},
               "dominant-seventh": {"label": "7",
                                    "components": {3: 0, 5: 0, 7: -1},
                                    "triad_label": "maj",
                                    "triad_components": {3: 0, 5: 0},
                                    "simple_label": "7",
                                    "simple_components": {3: 0, 5: 0, 7: -1}},
               "7": {"label": "7",
                     "components": {3: 0, 5: 0, 7: -1},
                     "triad_label": "maj",
                     "triad_components": {3: 0, 5: 0},
                     "simple_label": "7",
                     "simple_components": {3: 0, 5: 0, 7: -1}},
               "minor-major": {"label": "m(maj7)",
                               "components": {3: -1, 5: 0, 7: 0},
                               "triad_label": "m",
                               "triad_components": {3: -1, 5: 0},
                               "simple_label": "m",
                               "simple_components": {3: -1, 5: 0}},
               "major-minor": {"label": "m(maj7)",
                               "components": {3: -1, 5: 0, 7: 0},
                               "triad_label": "m",
                               "triad_components": {3: -1, 5: 0},
                               "simple_label": "m",
                               "simple_components": {3: -1, 5: 0}},
               "major-sixth": {"label": "6",
                               "components": {3: 0, 5: 0, 6: 0},
                               "triad_label": "maj",
                               "triad_components": {3: 0, 5: 0},
                               "simple_label": "maj",
                               "simple_components": {3: 0, 5: 0}},
               "minor-sixth": {"label": "m6",
                               "components": {3: -1, 5: 0, 6: 0},
                               "triad_label": "m",
                               "triad_components": {3: -1, 5: 0},
                               "simple_label": "m",
                               "simple_components": {3: -1, 5: 0}},
               "dominant-ninth": {"label": "9",
                                  "components": {3: 0, 5: 0, 7: -1, 9: 0},
                                  "triad_label": "maj",
                                  "triad_components": {3: 0, 5: 0},
                                  "simple_label": "7",
                                  "simple_components": {3: 0, 5: 0, 7: -1}},
               "major-ninth": {"label": "maj9",
                               "components": {3: 0, 5: 0, 7: 0, 9: 0},
                               "triad_label": "maj",
                               "triad_components": {3: 0, 5: 0},
                               "simple_label": "maj7",
                               "simple_components": {3: 0, 5: 0, 7: 0}},
               "minor-ninth": {"label": "m9",
                               "components": {3: -1, 5: 0, 7: -1, 9:0},
                               "triad_label": "m", 
                               "triad_components": {3: -1, 5: 0},
                               "simple_label": "m7",
                               "simple_components": {3: -1, 5: 0, 7: -1}},
                "maj69": {"label": "maj69",
                          "components": {3: 0, 5: 0, 6: 0, 9: 0},
                          "triad_label": "maj",
                          "triad_components": {3: 0, 5: 0},
                          "simple_label": "maj",
                          "simple_components": {3: 0, 5: 0}},
               "dominant-11th": {"label": "11",
                                 "components": {3: 0, 5: 0, 7: -1, 9: 0, 11: 0},
                                 "triad_label": "maj",
                                 "triad_components": {3: 0, 5: 0},
                                 "simple_label": "7",
                                 "simple_components": {3: 0, 5: 0, 7: -1}},
               "major-11th": {"label": "maj11",
                              "components": {3: 0, 5: 0, 7: 0, 9: 0, 11: 0},
                              "triad_label": "maj",
                              "triad_components": {3: 0, 5: 0},
                              "simple_label": "maj7",
                              "simple_components": {3: 0, 5: 0, 7: 0}},
               "minor-11th": {"label": "m11",
                              "simple_label": "m7",
                              "triad_label": "m",
                              "triad_components": {3: -1, 5: 0},
                              "components": {3: -1, 5: 0, 7: -1, 9: 0, 11: 0},
                              "simple_components": {3: -1, 5: 0, 7: -1}},
               "dominant-13th": {"label": "13",
                                 "components": {3: 0, 5: 0, 7: -1, 9: 0, 11: 0, 13: 0},
                                 "triad_label": "maj",
                                 "triad_components": {3: 0, 5: 0},
                                 "simple_label": "7",
                                 "simple_components": {3: 0, 5: 0, 7: -1}},
               "major-13th": {"label": "maj13",
                              "components": {3: 0, 5: 0, 7: 0, 9: 0, 11: 0, 13: 0},
                              "triad_label": "maj",
                              "triad_components": {3: 0, 5: 0},
                              "simple_label": "maj7",
                              "simple_components": {3: 0, 5: 0, 7: 0}},
               "minor-13th": {"label": "m13",
                              "components": {3: -1, 5: 0, 7: -1, 9: 0, 11: 0, 13: 0},
                              "triad_label": "m",
                              "triad_components": {3: -1, 5: 0},
                              "simple_label": "m7",
                              "simple_components": {3: -1, 5: 0, 7: -1}},
               "suspended-second": {"label": "sus2",
                                    "components": {2: 0, 5: 0},
                                    "triad_label": "sus2",
                                    "triad_components": {2: 0, 5: 0},
                                    "simple_label": "sus2",
                                    "simple_components": {2: 0, 5: 0}},
               "suspended-fourth": {"label": "sus4",
                                    "components": {4: 0, 5: 0},
                                    "triad_label": "sus4",
                                    "triad_components": {4: 0, 5: 0},
                                    "simple_label": "sus4",
                                    "simple_components": {4: 0, 5: 0}}
}
                

class Harmony(object):
    _chord_symbol = None
    _harte_notation = None
    _pitch_classes = None
    _pitch_classes_binary = None
    _triad_chord_symbol = None
    _triad_harte_notation = None
    _triad_pitch_classes = None
    _triad_pitch_classes_binary = None
    _simple_chord_symbol = None
    _simple_harte_notation = None
    _simple_pitch_classes = None
    _simple_pitch_classes_binary = None

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
        if self._chord_symbol is None:
            self._chord_symbol = self._get_root_label(self.harmony_dict['root'])
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
        if self._triad_chord_symbol is None:
            self._triad_chord_symbol = self._get_root_label(self.harmony_dict['root'])
            kind = self.harmony_dict["kind"]["text"]
            if kind in CHORD_DICT.keys():
                self._triad_chord_symbol += CHORD_DICT[kind]["triad_label"]
            else:
                print("Unknown chord kind: %s, igoring." %
                        str(self.harmony_dict["kind"]["text"]))

            # if self.harmony_dict["degrees"]:
            #     degree_labels = []
            #     for degree in self.harmony_dict["degrees"]:
            #         degree_labels.append(self._get_degree_label(degree))
            #     degree_labels.sort(key=lambda tup: tup[0])
            #     degree_labels = [str(dl_tup[1]) + dl_tup[0] for dl_tup in degree_labels]
            #     self._chord_symbol += "(%s)" % ",".join(degree_labels)

        return self._triad_chord_symbol

    def get_simple_chord_symbol(self):
        if self._simple_chord_symbol is None:
            self._simple_chord_symbol = self._get_root_label(self.harmony_dict['root'])
            kind = self.harmony_dict["kind"]["text"]
            if kind in CHORD_DICT.keys():
                self._simple_chord_symbol += CHORD_DICT[kind]["simple_label"]
            else:
                print("Unknown chord kind: %s, igoring." %
                        str(self.harmony_dict["kind"]["text"]))

            # if self.harmony_dict["degrees"]:
            #     degree_labels = []
            #     for degree in self.harmony_dict["degrees"]:
            #         degree_labels.append(self._get_degree_label(degree))
            #     degree_labels.sort(key=lambda tup: tup[0])
            #     degree_labels = [str(dl_tup[1]) + dl_tup[0] for dl_tup in degree_labels]
            #     self._chord_symbol += "(%s)" % ",".join(degree_labels)

        return self._simple_chord_symbol

    def get_harte_notation(self):
        if self._harte_notation is None:
            self._harte_notation = "%s:" % self._get_root_label(self.harmony_dict['root'])
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
        if self._triad_harte_notation is None:
            self._triad_harte_notation = "%s:" % self._get_root_label(self.harmony_dict['root'])
            kind = self.harmony_dict["kind"]["text"]
            if kind in CHORD_DICT.keys():
                components = copy.deepcopy(CHORD_DICT[kind]["triad_components"])
            else:
                components = copy.deepcopy(CHORD_DICT["major"]["triad_components"])
                print("Unknown chord kind: %s, using \"major\"." %
                        str(self.harmony_dict["kind"]["text"]))

            # if self.harmony_dict["degrees"]:
            #     for degree in self.harmony_dict["degrees"]:
            #         value = int(degree["degree-value"]["text"])
            #         if value not in (5, 7):
            #             continue

            #         if "degree-alter" in degree.keys():
            #             alter = int(degree["degree-alter"]["text"])
            #             components[value] = alter
            #         else:
            #             components[alter] = 0

            component_strings = []
            for i in range(2, 14):
                if i in components.keys():
                    alter_label = self._get_alter_label(components[i])
                    component_strings.append("%s%i" % (alter_label, i))
            self._triad_harte_notation += "(%s)" % (",".join(component_strings))

        return self._triad_harte_notation

    def get_simple_harte_notation(self):
        if self._simple_harte_notation is None:
            self._simple_harte_notation = "%s:" % self._get_root_label(self.harmony_dict['root'])
            kind = self.harmony_dict["kind"]["text"]
            if kind in CHORD_DICT.keys():
                components = copy.deepcopy(CHORD_DICT[kind]["simple_components"])
            else:
                components = copy.deepcopy(CHORD_DICT["major"]["simple_components"])
                print("Unknown chord kind: %s, using \"major\"." %
                        str(self.harmony_dict["kind"]["text"]))

            # if self.harmony_dict["degrees"]:
            #     for degree in self.harmony_dict["degrees"]:
            #         value = int(degree["degree-value"]["text"])
            #         if value not in (5, 7):
            #             continue

            #         if "degree-alter" in degree.keys():
            #             alter = int(degree["degree-alter"]["text"])
            #             components[value] = alter
            #         else:
            #             components[alter] = 0

            component_strings = []
            for i in range(2, 14):
                if i in components.keys():
                    alter_label = self._get_alter_label(components[i])
                    component_strings.append("%s%i" % (alter_label, i))
            self._simple_harte_notation += "(%s)" % (",".join(component_strings))

        return self._simple_harte_notation

    def get_pitch_classes(self):
        if self._pitch_classes is None:
            self._pitch_classes = parse_chord(self.get_harte_notation()).tones
        return self._pitch_classes

    def get_triad_pitch_classes(self):
        if self._triad_pitch_classes is None:
            self._triad_pitch_classes = parse_chord(self.get_triad_harte_notation()).tones
        return self._triad_pitch_classes

    def get_simple_pitch_classes(self):
        if self._simple_pitch_classes is None:
            self._simple_pitch_classes = parse_chord(self.get_simple_harte_notation()).tones
        return self._simple_pitch_classes

    def get_pitch_classes_binary(self):
        if self._pitch_classes_binary is None:
            self._pitch_classes_binary = parse_chord(self.get_harte_notation()).tones_binary
        return self._pitch_classes_binary

    def get_triad_pitch_classes_binary(self):
        if self._triad_pitch_classes_binary is None:
            self._triad_pitch_classes_binary = parse_chord(self.get_triad_harte_notation()).tones_binary
        return self._triad_pitch_classes_binary

    def get_simple_pitch_classes_binary(self):
        if self._simple_pitch_classes_binary is None:
            self._simple_pitch_classes_binary = parse_chord(self.get_simple_harte_notation()).tones_binary
        return self._simple_pitch_classes_binary
