import argparse
import xml.etree.ElementTree as ET
import os.path as op
import json
from pathlib import Path

NOTES_DICT = {'B#': 1, 'C': 1, 'C#': 2, 'Db': 2, 'D': 3, 'D#': 4, 'Eb': 4, 'E': 5, 
              'Fb': 5, 'E#': 6, 'F': 6, 'F#': 7, 'Gb': 7, 'G': 8, 'G#': 9, 'Ab': 9, 
              'A': 10, 'A#': 11, 'Bb': 11, 'B': 12, 'Cb': 12, '': 0}

DURATION_DICT = {'whole': 0, 'half': 1, 'quarter': 2, 'eighth': 3, 'sixteenth': 4, 
                 'whole-trip': 5, 'half-trip': 6, 'quar-trip': 7, 'eigh-trip': 8, 
                 'six-trip': 9, 'whole-dot': 10, 'half-dot': 11, 'quar-dot': 12, 
                 'eigh-dot': 13, 'six-dot': 14, 'thirtysec': 15, 
                 'thirtysec-trip': 16, 'thirtysec-dot': 17}

MAJOR_KEY_DICT = {'0': 1, '1': 8, '2': 3, '3': 10, '4': 5, '5': 12, '6': 7, '-1': 6, 
                  '-2': 11, '-3': 4, '-4': 9, '-5': 2, '-6': 7 }

MINOR_KEY_DICT = {'0': 10, '1': 5, '2': 12, '3': 7, '4': 2, '5': 9, '6': 4, '-1': 3, 
                  '-2': 8, '-3': 1, '-4': 6, '-5': 11, '-6': 4 }

def xml_to_dict(fpath):
    def recurse(root, root_dict):
        if root.tag == "part":
            root_dict["measures"] = []
        if root.tag == "measure":
            root_dict["harmonies"] = []
            root_dict["notes"] = []

        for child in root:
            child_dict = {'attributes': child.attrib}
            if len(list(child)) == 0:
                child_dict['text'] = child.text
            else:
                child_dict = recurse(child, child_dict)

            if child.tag == "measure":
                root_dict["measures"].append(child_dict)
            elif child.tag == "harmony":
                root_dict["harmonies"].append(child_dict)
            elif child.tag == "note":
                root_dict["notes"].append(child_dict)
            else:
                root_dict[child.tag] = child_dict
        return root_dict

    xml_dict = {}
    tree = ET.parse(fpath)
    root = tree.getroot()
    xml_dict[root.tag] = {"attributes": root.attrib}
    return recurse(root, xml_dict)
                


# def parse_file(fpath):
#     print("Parsing %s" % fpath)
#     parsed_data = {}
#     tree = ET.parse(fpath)
#     root = tree.getroot()
#     for child in root:
#         if child.tag == "movement-title":
#             parsed_data["title"] = child.text
#         if child.tag == "identification":
#             for subchild in child:
#                 if subchild.tag == "creator":
#                     parsed_data["artist"] = subchild.text
#                     break
#         if child.tag == "part":
#             for measure in part:
#                 for mchild in measure:
#                     if mchild.tag == "attributes":
#                         for attrib in mchild:
#                             if attrib.tag == "key":
#                                 for 

    # import pdb
    # pdb.set_trace()

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--song", help="Name of songfile to parse", type=str)
    # args = parser.parse_args()
    # fname = args.song
    fname = "charlie_parker-moose_the_mooche.xml"
    xml_path = op.join(str(Path(op.abspath(__file__)).parents[2]), 'data', 'raw', 'xml')
    fpaths = [op.join(xml_path, fname)]
    parsed_data = []
    for fpath in fpaths:
        with open(fname.split('.')[0] + '.json', 'w') as fp:
            # parsed_data.append(parse_file(fpath))
            json.dump(xml_to_dict(fpath), fp, indent=4)
