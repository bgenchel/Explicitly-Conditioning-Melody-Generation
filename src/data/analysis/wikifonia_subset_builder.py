
from __future__ import print_function
import json
import os
import os.path as op
import pdb
import requests as req
import sys

WIKIFONIA_DIR = op.expanduser(op.join('~', 'Dropbox', 'Data', 'Wikifonia'))
DEST_DIR = op.join(os.getcwd(), 'dataset')

if not op.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

files = [f.strip().lower() for f in os.listdir(WIKIFONIA_DIR)]

genre_data = json.load(open('wikifonia_genre_data.json', 'rb'), encoding="utf-8")
dataset = set()
for f in files:
    try:
        # pdb.set_trace()
        if unicode(f[:-4]) in genre_data['jazz']:
            os.rename(op.join(WIKIFONIA_DIR, f), op.join(DEST_DIR, f))
    except Exception as e:
        print("problem processing file %s: %s" % (f, e.message))
