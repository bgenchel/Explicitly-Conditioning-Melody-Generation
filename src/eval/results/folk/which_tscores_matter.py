import os
import os.path as op
import json
import pdb

for f in os.listdir('tscores'):
    jsn = json.load(open(op.join('tscores', f), 'r'))
    for k, v in jsn.items():
        if v[1] < .05:
            input()
            model_name = '_'.join(f.split('.')[0].split('_')[:-1])
            cond_name = f.split('.')[0].split('_')[-1]
            print('%s :: %s :: %s - tscore: %.3f' % (model_name, k, cond_name,  v[1]))
