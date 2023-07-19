import argparse
import json
import logging
import sys
from dataclasses import field, dataclass
from json import JSONDecodeError
from math import ceil
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional

from Levenshtein import distance, opcodes
from tqdm import tqdm

from src.align import align, fix_times, convert_ali_to_segments
from src.data_loaders import Word, load_audio, extract_audio, load_reco

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


class Matcher:
    '''Matcher utility class

    This class loads a large text corpus and allows matching short text segments to it.
    '''

    def __init__(self, corpus: Path):
        lines = []
        orig_lines = None
        self.unnorm = None
        self.corpus = []
        words = set()
        self.vocab = Dictionary()
        try:
            with open(corpus) as f:
                ali = json.load(f)
            orig_lines = []
            for l in ali:
                lines.append(l[1])
                orig_lines.append(l[0])
                words.update(l[0])
                words.update(l[1])

        except JSONDecodeError:
            with open(corpus) as f:
                for l in f:
                    tok = l.strip().split()
                    lines.append(tok)
                    words.update(tok)

        self.vocab.put(words)

        for l in lines:
            for w in l:
                self.corpus.append(self.vocab.get_id(w))
        if orig_lines:
            self.unnorm = []
            for l in orig_lines:
                for w in l:
                    self.unnorm.append(self.vocab.get_id(w))

    def get_corpus_chunk(self, start: int, end: int) -> Tuple[str, Optional[str]]:
        norm = self.vocab.get_text(self.corpus[start:end])
        orig = None
        if self.unnorm:
            orig = self.vocab.get_text(self.unnorm[start:end])
        return norm, orig

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

    def run(self, words: List[Word], audio_file: Path, model: str,
            chunk_len: int = 200, chunk_stride: int = 200) -> List[Dict]:
        ali_segs = []
        ali_masks = []
        ali_ref_texts = []
        ali_orig_texts = []

        logger.info('Loading audio...')
        audio = load_audio(audio_file)

        logger.info('Making chunks...')
        chunk_num = ceil(len(words) / chunk_stride)
        for c in tqdm(range(chunk_num)):
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
            ref_text, orig_text = self.get_corpus_chunk(min_i + rb, min_i + re)

            seg, mask = extract_audio(audio['input_values'], words_chunk[hb:he], audio['samp_freq'])

            ali_segs.append(seg)
            ali_masks.append(mask)
            ali_ref_texts.append(ref_text)
            if orig_text:
                ali_orig_texts.append(orig_text)

        if not ali_segs:
            return []

        logger.info('Aligning reference to audio...')
        ali_res = align(model, ali_segs, ali_ref_texts)

        logger.info('Fixing times...')
        ali_all = []
        if len(ali_orig_texts) > 0:
            for ali_words, mask, orig in zip(ali_res.values(), ali_masks, ali_orig_texts):
                ali_fixed = fix_times(ali_words, mask, audio['samp_freq'])
                for w, o in zip(ali_fixed, orig.split()):
                    w['norm'] = w['text']
                    w['text'] = o
                ali_all.extend(ali_fixed)
        else:
            for ali_words, mask in zip(ali_res.values(), ali_masks):
                ali_fixed = fix_times(ali_words, mask, audio['samp_freq'])
                ali_all.extend(ali_fixed)

        logger.info('Removing overlapping words...')
        ali_all = sorted(ali_all, key=lambda x: x['timestamp'][0])
        ret = [ali_all[0]]
        for x in ali_all[1:]:
            if x['timestamp'][0] >= ret[-1]['timestamp'][1]:
                ret.append(x)

        return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reco_out', type=Path)
    parser.add_argument('reco_id', type=str)
    parser.add_argument('audio', type=Path)
    parser.add_argument('transcript', type=Path)
    parser.add_argument('model', type=str)
    parser.add_argument('output', type=Path)
    parser.add_argument('--chunk-len', type=int, default=200)
    parser.add_argument('--chunk-stride', type=int, default=150)
    parser.add_argument('--sil-gap', type=float, default=1)
    parser.add_argument('--max-len', type=float, default=-1)

    args = parser.parse_args()

    logger.info('Loading reco data...')
    words = load_reco(args.words)

    if args.recoid not in words:
        logger.error(f'Recording with id {args.recoid} is not present in the {args.words} file.')
        return

    words = words[args.recoid]

    logger.info('Loading transcript...')
    matcher = Matcher(args.transcript)

    logger.info('Matching reco and transcript...')
    ali = matcher.run(words, args.audio, args.model, args.chunk_len, args.chunk_stride)

    logger.info('Grouping segments...')
    segs = convert_ali_to_segments(ali, words, args.sil_gap, args.max_len)

    with open(args.output, 'w') as f:
        json.dump(segs, f)


if __name__ == '__main__':
    main()
