from __future__ import print_function
from collections import OrderedDict
import json
import os
import os.path as op
import pdb
import requests as r

DATASET_LOCATION = op.expanduser(op.join('~', 'Dropbox', 'Data', 'Wikifonia'))
files = [f.strip().lower() for f in os.listdir(DATASET_LOCATION)]
files.sort()
pdb.set_trace()

with open('wikifonia_songs.json', 'w') as outfile:
    data = OrderedDict()
    for f in files:
        try:
            artist, song = f[:-4].split(' - ')
            # outfile.write("artist:\t%s\tsong:\t%s\n" % (artist.lower(), song.lower()))
            if artist not in data:
                data[artist] = []
            data[artist].append(song)
        except Exception as e:
            print("file %s caused error: %s" % (f, e.message))
    outfile.write(json.dumps(data, indent=4))
