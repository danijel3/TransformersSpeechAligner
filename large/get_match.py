import argparse
import json
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional

from Levenshtein import distance, opcodes


@dataclass
class Dictionary:
    word2id: Dict[str, int] = field(default_factory=lambda: {'<unk>': 0})
    id2word: Dict[int, str] = field(default_factory=lambda: {0: '<unk>'})

    def put(self, words: Set):
        for id, word in enumerate(sorted(list(words))):
            self.word2id[word] = id + 1
            self.id2word[id + 1] = word

    def get_id(self, word: str, warn_oov: bool = False) -> int:
        if word not in self.word2id:
            if warn_oov:
                print(f'WARN: missing word "{word}"')
            return 0
        return self.word2id[word]

    def get_word(self, id: int) -> str:
        return self.id2word[id]

    def get_text(self, ids: List[int]) -> str:
        return ' '.join([self.get_word(x) for x in ids])

    def get_ids(self, text: str, warn_oov: bool = False) -> List[int]:
        return self.to_ids(text.strip().split(), warn_oov)

    def to_ids(self, text: List[str], warn_oov: bool = False) -> List[int]:
        return [self.get_id(x, warn_oov) for x in text]

    def to_words(self, ids: List[int]) -> List[str]:
        return [self.get_word(x) for x in ids]

    def save(self, file: Path):
        with open(file, 'w') as f:
            for id, word in self.id2word.items():
                f.write(f'{id} {word}\n')

    def load(self, file: Path):
        self.word2id = {}
        self.id2word = {}
        with open(file) as f:
            for l in f:
                tok = l.strip().split()
                id = int(tok[0])
                word = tok[1]
                self.id2word[id] = word
                self.word2id[word] = id


@dataclass
class Word:
    text: str
    start: float
    end: float


class Matcher:
    '''Matcher utility class

    This class loads a large text corpus and allows matching short text segments to it.
    '''

    def __init__(self, word_seq: List[str]):
        self.corpus = []

        self.vocab = Dictionary()

        self.vocab.put(set(word_seq))

        for w in word_seq:
            self.corpus.append(self.vocab.get_id(w))

    def _initial_match(self, text: str, threshold: float = 0.9) -> List[Tuple[int, float]]:
        """
        Find locations of best BOW match by calculating a moving-windows histogram over data. The threshold is used to
        filter locations that are greater-equal than that value product of the best location match.
        :param text: input text
        :param threshold: filter threshold
        :return: list of (location, score) tuples
        """
        text_ids = self.vocab.get_ids(text)
        hist = set(text_ids)
        hist_sum = 0
        best_sum = 0
        N = int(len(text_ids))
        for i in self.corpus[:N]:
            if i in hist:
                hist_sum += 1
        dist = [(0, hist_sum)]
        for i in range(N, len(self.corpus)):
            if self.corpus[i - N] in hist:
                hist_sum -= 1
            if self.corpus[i] in hist:
                hist_sum += 1
            if hist_sum >= best_sum * 0.9:
                dist.append((i, hist_sum))
                if hist_sum > best_sum:
                    best_sum = hist_sum
        return list(filter(lambda x: x[1] >= best_sum * threshold, dist))

    def _find_min_diff(self, dist: List[Tuple[int, float]], text: str) -> Tuple[int, float]:
        """
        Use Levenshtein distance to find the location with minimum difference from the list of candiadates.
        :param dist: candidates returned by _initial_match
        :param text: input text
        :return: (location, distance) tuple
        """
        N = len(text.split())
        min_d = 10e10
        min_i = -1
        for i, _ in dist:
            ref = self.vocab.get_text(self.corpus[i - N:i])
            d = distance(ref, text)
            if d < min_d:
                min_d = d
                min_i = i - N
        return min_i, min_d

    def _find_matching_seq(self, text: str, min_i: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Finds sequence in both reference and hypothesis that is a good match. Throws away any initial/final
        insertion/deletion and leaves the mathching middle.
        :param text: input text
        :param min_i: the best location as found by _find_min_diff
        :return: (hyp start, hyp end) , (ref start, ref end) tuple of tuples
        """
        text_ids = self.vocab.get_ids(text)
        N = int(len(text_ids))
        m = list(filter(lambda x: x[0] == 'equal', opcodes(text_ids, self.corpus[min_i:min_i + N])))
        if not m:
            return None
        hb = m[0][1]
        he = m[-1][2]
        rb = m[0][3]
        re = m[-1][4]
        return (hb, he), (rb, re)

    def get_corpus_chunk(self, start: int, end: int) -> str:
        """
        Get a portion of the corpus as a string.
        :param start: start index
        :param end: end index
        :return: text
        """
        return self.vocab.get_text(self.corpus[start:end])

    def run(self, words: List[Word], chunk_len: int = 200, chunk_stride: int = 200) -> List:

        ret = []

        chunk_num = ceil(len(words) / chunk_stride)
        for c in range(chunk_num):
            cs = c * chunk_stride
            ce = cs + chunk_len

            words_chunk = words[cs:ce]
            text = ' '.join([x.text for x in words_chunk])

            locs = self._initial_match(text)
            min_i, min_d = self._find_min_diff(locs, text)
            if min_d / len(text) > 0.5:
                continue
            res = self._find_matching_seq(text, min_i)
            if not res:
                continue
            (hb, he), (rb, re) = res
            ref_text = self.get_corpus_chunk(min_i + rb, min_i + re)

            reco_words = words_chunk[hb:he]
            reco_text = ' '.join([x.text for x in reco_words])

            ret.append({'ref_text': ref_text, 'ref_offset': {'start': min_i + rb, 'end': min_i + re},
                        'reco_text': reco_text, 'reco_offset': {'start': cs + hb, 'end': cs + he},
                        'audio_offsets': [(x.start, x.end) for x in reco_words]})

        return ret


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('reco', type=Path)
    parser.add_argument('norm', type=Path)
    parser.add_argument('out', type=Path)

    args = parser.parse_args()

    ref_words = []
    with open(args.norm) as f:
        for l in json.load(f):
            ref_words.extend(l['norm'].split())

    reco_words = []
    with open(args.reco) as f:
        for s in json.load(f):
            for t, w in zip(s['text'].split(), s['words']):
                reco_words.append(Word(t, w['start'], w['end']))

    matcher = Matcher(ref_words)

    ret = matcher.run(reco_words)

    with open(args.out, 'w') as f:
        json.dump(ret, f, indent=4)
