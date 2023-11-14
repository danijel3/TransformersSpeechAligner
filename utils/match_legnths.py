import argparse
import json
from pathlib import Path

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=Path)

    args = parser.parse_args()

    with open(args.json) as f:
        match = json.load(f)

    lengths = [x["audio_offsets"][-1][1] - x["audio_offsets"][0][0] for x in match]

    print(lengths)
    print(f'Num: {len(lengths)}')
    print(f'Mean: {np.mean(lengths)}')
    print(f'Min:{np.min(lengths)}')
    print(f'Max: {np.max(lengths)}')
