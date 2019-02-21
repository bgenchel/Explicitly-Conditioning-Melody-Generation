import os
import os.path as op
import json
import pdb

for f in os.listdir('tscores'):
    jsn = json.load(open(op.join('tscores', f), 'r'))
    for k, v in jsn.items():
        if v[1] < .05:
            input()
            print('%s::%s - tscore: %.3f' % (f, k, v[1]))
