import sys
import re
import numpy as np

if __name__ == '__main__':
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        epochs = f.read().split('main loss')
        speeds = [float(re.search(r'\, ([0-9\.]+)s', e.splitlines()[-2]).group(1)) for e in epochs[:-1]]
        print(f"Average speed: {np.mean(speeds)}")