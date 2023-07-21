import argparse
import json
import re
from pathlib import Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('orig', type=Path)
    parser.add_argument('json', type=Path)

    args = parser.parse_args()

    orig_file = args.orig
    norm_file = args.json

    norms = []
    with open(orig_file) as f:
        for l in f:
            norm = {}
            l = l.strip()
            t = l.index('\t')
            uid = l[:t]
            l = l[t + 1:]
            norm['uid'] = uid
            norm['text'] = l
            l = re.sub(r'[^\w]', ' ', l.lower())
            text = []
            offsets = []
            for m in re.finditer(r'[^ ]+', l):
                text.append(m[0])
                offsets.append((m.start(), m.end()))
            norm['norm'] = ' '.join(text)
            norm['normoff'] = offsets
            norms.append(norm)

    with open(norm_file, 'w') as f:
        json.dump(norms, f, indent=2)
