import argparse
import json
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=Path)
    parser.add_argument('txt', type=Path)

    args = parser.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    with open(args.txt, 'w') as g:
        for seg in data:
            g.write(f'{seg["norm"]}\n')
