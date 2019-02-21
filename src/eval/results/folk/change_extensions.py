import os
import os.path as op

DIR = 'mgeval_results'

for f in os.listdir(DIR):
    fpath = op.join(DIR, f)
    if op.isdir(fpath):
        for ff in os.listdir(fpath):
            components = ff.split('.')
            new_name = '.'.join([components[0], components[-1]])
            os.rename(op.join(fpath, ff), op.join(fpath, new_name))
