import argparse
import json
from pathlib import Path
from typing import List

from tqdm import tqdm


def match(trans: List[Path], reco: List[Path], ngram: int = 3):
    trans_id = {}
    vocab = {}
    bow = {}
    print('Loadin trans...')
    for file in tqdm(trans):
        tid = len(trans_id)
        trans_id[tid] = str(file)
        with open(file) as f:
            for l in f:
                tok = l.strip().split()
                words = []
                for w in tok:
                    if w not in vocab:
                        vocab[w] = len(vocab)
                    words.append(vocab[w])
                for i in range(len(words) - ngram):
                    ng = '_'.join([str(x) for x in words[i:i + ngram]])
                    if ng not in bow:
                        bow[ng] = set()
                    bow[ng].add(tid)

    print('Matches:')

    for file in reco:
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
        print(f'JSON {file} matches TRANS {trans_id[best_tid]}')
        print(f'STAT score {matches[best_tid]:0.2%} other matches [{", ".join([f"{x:0.2%}" for x in matches])}]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', type=Path)

    args = parser.parse_args()

    trans = []
    reco = []
    for file in args.files:
        if file.suffix == '.txt':
            trans.append(file)
        elif file.suffix == '.json':
            reco.append(file)
        else:
            print(f'Unknown suffix in file: {file}')

    match(trans, reco)
