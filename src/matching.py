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

from src.align import align, fix_times, convert_ali_to_segments, get_errors
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

    def run(self, words: List[Word], audio_file: Path, model: str,
            chunk_len: int = 200, chunk_stride: int = 200) -> List[Dict]:
        ali_segs = []
        ali_masks = []
        ali_ref_texts = []
        ali_ref_pos = []

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
            ref_text = self.get_corpus_chunk(min_i + rb, min_i + re)

            seg, mask = extract_audio(audio['input_values'], words_chunk[hb:he], audio['samp_freq'])

            ali_segs.append(seg)
            ali_masks.append(mask)
            ali_ref_texts.append(ref_text)
            ali_ref_pos.append((min_i + rb, min_i + re))

        if not ali_segs:  # TODO this eats up too much words sometimes
            return []

        # ali_dbg = Path('data/ali_dbg.json')
        # if ali_dbg.exists():
        #     logger.info('Loading ali from dbg...')
        #     ali_res = json.load(ali_dbg.open('r'))
        # else:
        #     logger.info('Aligning reference to audio...')
        #     ali_res = align(model, ali_segs, ali_ref_texts)
        #     with open(ali_dbg, 'w') as f:
        #         json.dump(ali_res, f)
        #
        logger.info('Aligning reference to audio...')
        ali_res = align(model, ali_segs, ali_ref_texts)

        for ali, pos in zip(ali_res.values(), ali_ref_pos):
            for w, p in zip(ali, range(pos[0], pos[1])):
                w['ref_pos'] = p

        logger.info('Fixing times...')
        ali_all = []
        for ali_words, mask in zip(ali_res.values(), ali_masks):
            ali_fixed = fix_times(ali_words, mask, audio['samp_freq'])
            for a, w in zip(ali_fixed, ali_words):
                a['ref_pos'] = w['ref_pos']
            ali_all.extend(ali_fixed)

        logger.info('Removing overlapping words...')
        ali_all = sorted(ali_all, key=lambda x: x['timestamp'][0])
        ret = [ali_all[0]]
        for x in ali_all[1:]:
            if x['timestamp'][0] >= ret[-1]['timestamp'][1]:
                ret.append(x)

        return ret


def convert_ali_to_corpus_lines(ali: List[Dict], reco: List[Dict], norm: List, sil_gap: float = 2) -> List[Dict]:
    norm_words = []
    segs = []
    for ln, l in enumerate(norm):
        segs.append({
            'id': l['uid'],
            'text': l['text'],
            'w': []
        })
        for n in l['normoff']:
            norm_words.append((ln, n))

    for w in ali:
        ln, n = norm_words[w['ref_pos']]
        segs[ln]['w'].append((w, n))

    for s in segs:
        words = sorted(s['w'], key=lambda x: x[0]['timestamp'][0])
        s['norm'] = ' '.join([x[0]['text'] for x in words])
        s['words'] = [{
            'time_s': round(x[0]['timestamp'][0], 3),
            'time_e': round(x[0]['timestamp'][1], 3),
            'char_s': x[1][0],
            'char_e': x[1][1],
        } for x in words]
        if len(s['words']) > 0:
            s['start'] = s['words'][0]['time_s']
            s['end'] = s['words'][-1]['time_e']
        else:
            s['start'] = s['end'] = 0
        s['w'] = []

    unaligned = []
    for w in reco:
        for s in segs:
            mp = (w.start + w.end) / 2
            if s['start'] <= mp <= s['end']:
                s['w'].append(w)
                break
        else:
            unaligned.append(w)

    for s in segs:
        words = sorted(s['w'], key=lambda x: x.start)
        s['reco'] = ' '.join([x.text for x in words])
        s['reco_words'] = [{
            'time_s': round(x.start, 3),
            'time_e': round(x.end, 3),
        } for x in words]
        s.pop('w')
        s['errors'] = get_errors(s['norm'].split(), s['reco'].split())
        s['errors']['cer'] = get_errors(s['norm'], s['reco'])['wer']

        # combine unaligned reco words into segments
    unaligned = sorted(unaligned, key=lambda x: x.start)
    unaligned_segs = []
    seg = [unaligned[0]]
    for w in unaligned[1:]:
        if w.start - seg[-1].end > sil_gap:
            unaligned_segs.append(seg)
            seg = [w]
        else:
            seg.append(w)
    unaligned_segs.append(seg)

    for seg in unaligned_segs:
        segs.append({'reco': ' '.join([x.text for x in seg]),
                     'reco_words': [{
                         'time_s': round(x.start, 3),
                         'time_e': round(x.end, 3)
                     } for x in seg],
                     'start': round(seg[0].start, 3),
                     'end': round(seg[-1].end, 3),
                     'unaligned': True})

    return sorted(segs, key=lambda x: x['start'])


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
    words = load_reco(args.reco_out)

    if args.reco_id not in words:
        logger.error(f'Recording with id {args.reco_id} is not present in the {args.reco_out} file.')
        return

    words = words[args.reco_id]

    word_seq = []
    norm = None
    try:  # try and treat the input transcript as JSON file
        with open(args.transcript) as f:
            norm = json.load(f)
            for l in norm:
                tok = l['norm'].strip().split()
                word_seq.extend(tok)

    except JSONDecodeError:  # if it fails read it as a normal text file
        with open(args.transcript) as f:
            for l in f:
                tok = l.strip().split()
                word_seq.extend(tok)

    logger.info('Loading transcript...')
    matcher = Matcher(word_seq)

    logger.info('Matching reco and transcript...')
    ali = matcher.run(words, args.audio, args.model, args.chunk_len, args.chunk_stride)

    if norm:
        logger.info('Organizing tokens by corpus lines...')
        segs = convert_ali_to_corpus_lines(ali, words, norm)
    else:
        logger.info('Grouping segments...')
        segs = convert_ali_to_segments(ali, words, args.sil_gap, args.max_len)

    with open(args.output, 'w') as f:
        json.dump(segs, f, indent=2)


if __name__ == '__main__':
    main()
