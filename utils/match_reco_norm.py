import argparse
import json
from pathlib import Path

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('reco', type=Path)
    parser.add_argument('norm', nargs='+', type=Path)

    args = parser.parse_args()

    vocab = {}

    # with open(args.reco) as f:
    #     reco = json.load(f)
    #
    # words = []
    # for s in reco:
    #     for w in s['text'].split():
    #         if w not in vocab:
    #             x = len(vocab)
    #             vocab[w] = x
    #             words.append(x)
    #         else:
    #             words.append(vocab[w])
    #
    # ngram_list = set()
    # for i in range(len(words) - 3):
    #     ngram_list.add(tuple(words[i:i + 3]))
    #
    # print(f'Found {len(ngram_list)} ngrams in reco')

    ngrams = {}
    norm_list = {}
    for norm_id, norm in enumerate(tqdm(args.norm)):
        with open(norm) as f:
            norm_data = json.load(f)

        norm_list[norm_id] = norm.stem

        words = []
        for s in norm_data:
            for w in s['norm'].split():
                try:
                    words.append(vocab[w])
                except KeyError:
                    x = len(vocab)
                    vocab[w] = x
                    words.append(x)

        for i in range(len(words) - 3):
            x = tuple(words[i:i + 3])
            try:
                ngrams[x].add(norm_id)
            except KeyError:
                ngrams[x] = {norm_id}

    print(f'Found {len(ngrams)} ngrams in all norm files')
    print(f'Found {len(vocab)} words in all norm files')

    with open(args.reco) as f:
        reco = json.load(f)

    words = []
    for s in reco:
        for w in s['text'].split():
            if w not in vocab:
                words.append(-1)
            else:
                words.append(vocab[w])

    match_count = {}
    for i in range(len(words) - 3):
        x = tuple(words[i:i + 3])
        if x not in ngrams:
            continue
        for norm_id in ngrams[x]:
            try:
                match_count[norm_id] += 1
            except KeyError:
                match_count[norm_id] = 1

    match_count = sorted(match_count.items(), key=lambda x: x[1], reverse=True)

    print('Whole file matches:')
    for norm_id, count in match_count[:10]:
        print(f'{norm_list[norm_id]}: {count / len(words):%}')

    print('Per segment matches:')
    seg_matches = {}
    for uid, s in enumerate(reco):
        words = []

        for w in s['text'].split():
            if w not in vocab:
                words.append(-1)
            else:
                words.append(vocab[w])

        match_count = {}
        for i in range(len(words) - 3):
            x = tuple(words[i:i + 3])
            if x not in ngrams:
                continue
            for norm_id in ngrams[x]:
                try:
                    match_count[norm_id] += 1
                except KeyError:
                    match_count[norm_id] = 1

        if len(match_count) == 0:
            continue

        best_match = max(match_count.items(), key=lambda x: x[1])[0]
        try:
            seg_matches[best_match] += 1
        except KeyError:
            seg_matches[best_match] = 1

    all_count = sum(seg_matches.values())
    for norm_id, count in sorted(seg_matches.items(), key=lambda x: x[1], reverse=True):
        print(f'{norm_list[norm_id]}: {count / all_count:%}')
