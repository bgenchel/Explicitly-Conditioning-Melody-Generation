import os
import os.path as op
import shutil
import time
from stat import ST_CTIME


def get_test_runs(runspath, key=""):
    trdir = op.join(runspath, key, 'test_runs')
    return [op.join(trdir, fp) for fp in os.listdir(trdir)]

def main():
    runspath = op.join(op.abspath(os.getcwd()), 'runs') 
    paths = get_test_runs(runspath, 'pitch') + get_test_runs(runspath, 'duration')
    data = [(os.stat(path)[ST_CTIME], path) for path in paths] 
    shutil.rmtree(sorted(data, key=lambda t: t[0], reverse=True)[0][1])
    return 

if __name__ == '__main__':
    main()
