import argparse
import json
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm


def match(trans: List[Path], reco: List[Path], ngram: int = 3) -> List:
    trans_id = {}
    vocab = {}
    bow = {}
    ret = []
    print('Loadin trans...', flush=True)
    for file in tqdm(trans):
        tid = len(trans_id)
        trans_id[tid] = str(file)
        with open(file) as f:
            for l in f:
                tok = l.strip().split()
                words = []
                for w in tok[1:]:
                    if w not in vocab:
                        vocab[w] = len(vocab)
                    words.append(vocab[w])
                for i in range(len(words) - ngram):
                    ng = '_'.join([str(x) for x in words[i:i + ngram]])
                    if ng not in bow:
                        bow[ng] = set()
                    bow[ng].add(tid)

    print('Matching...', flush=True)

    for file in tqdm(reco):
        matches = [0] * len(trans_id)
        ng_num = 0
        with open(file) as f:
            data = json.load(f)
        for seg in data.values():
            words = []
            for w in seg['text'].split():
                if w in vocab:
                    words.append(vocab[w])
                else:
                    words.append(-1)
            for i in range(len(words) - ngram):
                ng = '_'.join([str(x) for x in words[i:i + ngram]])
                ng_num += 1
                if ng in bow:
                    for tid in bow[ng]:
                        matches[tid] += 1
        matches = [x / ng_num for x in matches]
        best_tid = max(enumerate(matches), key=lambda x: x[1])[0]
        ret.append({
            'reco': str(file),
            'trans': trans_id[best_tid],
            'score': matches[best_tid],
            'matches': matches
        })
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_json', type=Path)
    parser.add_argument('files', nargs='+', type=Path)

    args = parser.parse_args()

    if args.out_json.exists():
        print(f'Output file {args.out_json} already exists! Please remove it first.')
        sys.exit(1)

    trans = []
    reco = []
    for file in args.files:
        if file.suffix == '.txt':
            trans.append(file)
        elif file.suffix == '.json':
            reco.append(file)
        else:
            print(f'Unknown suffix in file: {file}')

    out = match(trans, reco)
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=4)
