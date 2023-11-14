import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from Levenshtein import distance, opcodes
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class RecoWord:
    word: int
    text: str
    start: float
    end: float
    seg: int
    pos: int


@dataclass
class NormWord:
    word: int
    text: str
    seg: int
    pos: int


def _initial_match(reco: List[RecoWord], norm: List[NormWord], threshold: float = 0.9, threshold2: float = 0.99) -> \
        List[Tuple[int, float]]:
    """
    Find locations of best BOW match by calculating a moving-windows histogram over data. The threshold is used to
    filter locations that are greater-equal than that value product of the best location match.
    :param text: input text
    :param threshold: filter threshold
    :return: list of (location, score) tuples
    """
    hist = set([reco.word for reco in reco])
    hist_sum = 0
    best_sum = 0
    N = len(reco)
    for i in norm[:N]:
        if i.word in hist:
            hist_sum += 1
    dist = [(0, hist_sum)]
    for i in range(N, len(norm)):
        if norm[i - N].word in hist:
            hist_sum -= 1
        if norm[i].word in hist:
            hist_sum += 1
        if hist_sum >= best_sum * 0.9:
            dist.append((i, hist_sum))
            if hist_sum > best_sum:
                best_sum = hist_sum
    ret = list(filter(lambda x: x[1] >= best_sum * threshold, dist))
    if len(ret) > 1000:
        ret = list(filter(lambda x: x[1] >= best_sum * threshold2, dist))
    return ret


def _find_min_diff(dist: List[Tuple[int, float]], reco: List[RecoWord], norm: List[NormWord]) -> Tuple[
    int, float, int]:
    """
    Use Levenshtein distance to find the location with minimum difference from the list of candiadates.
    :param dist: candidates returned by _initial_match
    :param text: input text
    :return: (location, distance) tuple
    """
    N = len(reco)
    reco_text = ' '.join([x.text for x in reco])
    norm_text = [x.text for x in norm]
    min_d = 10e10
    min_i = -1
    for i, _ in dist:
        norm_sub = ' '.join(norm_text[i - N:i])
        d = distance(norm_sub, reco_text)
        if d < min_d:
            min_d = d
            min_i = i - N
    return min_i, min_d, len(reco_text)


def _find_matching_seq_words(reco: List[RecoWord], norm: List[NormWord]) -> Optional[
    Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Finds sequence in both reference and hypothesis that is a good match. Throws away any initial/final
    insertion/deletion and leaves the mathching middle.
    :param text: input text
    :return: (hyp start, hyp end) , (ref start, ref end) tuple of tuples
    """
    reco_ids = [x.word for x in reco]
    norm_ids = [x.word for x in norm]
    reco_off = reco[0].pos
    norm_off = norm[0].pos
    m = list(filter(lambda x: x[0] == 'equal', opcodes(reco_ids, norm_ids)))
    if not m:
        return None
    rb = m[0][1]
    re = m[-1][2]
    nb = m[0][3]
    ne = m[-1][4]
    return (reco_off + rb, reco_off + re), (norm_off + nb, norm_off + ne)


def char_to_word_idx(text: str, char_pos: int) -> int:
    """
    Converts character position to word position.
    :param text: input text
    :param char_pos: character position
    :return: word position
    """
    words = text.split()
    pos = 0
    for i, w in enumerate(words):
        if pos + len(w) > char_pos:
            return i
        pos += len(w) + 1
    return len(words) - 1


def _find_matching_seq_chars(reco: List[RecoWord], norm: List[NormWord]) -> Optional[
    Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Finds sequence in both reference and hypothesis that is a good match. Throws away any initial/final
    insertion/deletion and leaves the mathching middle.
    :param text: input text
    :return: (hyp start, hyp end) , (ref start, ref end) tuple of tuples
    """
    reco_text = ' '.join([x.text for x in reco])
    norm_text = ' '.join([x.text for x in norm])
    reco_off = reco[0].pos
    norm_off = norm[0].pos
    m = list(filter(lambda x: x[0] == 'equal' and x[2] - x[1] > 2, opcodes(reco_text, norm_text)))
    if not m:
        return None
    rb = m[0][1]
    re = m[-1][2]
    nb = m[0][3]
    ne = m[-1][4]

    rb = char_to_word_idx(reco_text, rb)
    re = char_to_word_idx(reco_text, re - 1) + 1
    nb = char_to_word_idx(norm_text, nb)
    ne = char_to_word_idx(norm_text, ne - 1) + 1

    return (reco_off + rb, reco_off + re), (norm_off + nb, norm_off + ne)


def reco_to_chunks(reco_words: List, reco_gap: float = 15.0, min_len: int = 15, max_len: int = 1000) -> List:
    reco_chunks = []
    last_chunk = []
    last_end = reco_words[0].start
    for w in reco_words:
        if w.start - last_end > reco_gap or len(last_chunk) >= max_len:
            if len(last_chunk) >= min_len:
                reco_chunks.append((last_chunk[0].pos, last_chunk[-1].pos + 1))
            last_chunk = [w]
            last_end = w.start
        else:
            last_chunk.append(w)
            last_end = w.end
    if len(last_chunk) >= min_len:
        reco_chunks.append((last_chunk[0].pos, last_chunk[-1].pos + 1))
    return reco_chunks


def search_initial_matches(reco_words: List, reco_chunks: List, norm_words: List, good_th: float = 0.1,
                           segs: List = []) -> List[Dict]:
    for cs, ce in tqdm(reco_chunks):
        im = _initial_match(reco_words[cs:ce], norm_words)
        min_i, min_d, len_reco = _find_min_diff(im, reco_words[cs:ce], norm_words)
        if min_d / len_reco < good_th:
            m = _find_matching_seq_words(reco_words[cs:ce], norm_words[min_i:min_i + (ce - cs)])
            if not m:
                continue
            (rb, re), (nb, ne) = m
            segs.append({'reco': {'beg': rb, 'end': re}, 'norm': {'beg': nb, 'end': ne}})

    segs = sorted(segs, key=lambda x: x['norm']['beg'])
    seg_fix = [segs[0]]
    for seg in segs[1:]:
        if seg['norm']['beg'] > seg_fix[-1]['norm']['end']:
            seg_fix.append(seg)
    segs = sorted(seg_fix, key=lambda x: x['reco']['beg'])

    return segs


def search_between_segs(segs: List, reco_words: List, norm_words: List, ib_score_th: float = 0.5,
                        max_len: float = 2000) -> List[Dict]:
    ib_segs = [{'reco': {'beg': p['reco']['end'], 'end': n['reco']['beg']},
                'norm': {'beg': p['norm']['end'], 'end': n['norm']['beg']}} for p, n in zip(segs[:-1], segs[1:])]
    ib_segs = list(
        filter(lambda x: x['reco']['end'] - x['reco']['beg'] > 0 and x['norm']['end'] - x['norm']['beg'] > 0, ib_segs))

    nsegs = sorted(segs, key=lambda x: x['norm']['beg'])

    reco_end = segs[0]['reco']['beg']
    norm_beg = nsegs[0]['norm']['beg'] - reco_end
    norm_end = nsegs[0]['norm']['beg']
    if norm_beg < 0:
        norm_beg = 0
    if reco_end > 0 and norm_end > norm_beg:
        ib_segs.insert(0, {'reco': {'beg': 0, 'end': reco_end}, 'norm': {'beg': norm_beg, 'end': norm_end}})

    reco_beg = segs[-1]['reco']['end']
    reco_end = len(reco_words)
    norm_beg = nsegs[-1]['norm']['end']
    norm_end = norm_beg + (reco_end - reco_beg)
    if norm_end > len(norm_words):
        norm_end = len(norm_words)
    if reco_end > reco_beg and norm_end > norm_beg:
        ib_segs.append({'reco': {'beg': reco_beg, 'end': reco_end}, 'norm': {'beg': norm_beg, 'end': norm_end}})

    for i, seg in enumerate(ib_segs):

        len_ratio = (seg['reco']['end'] - seg['reco']['beg']) / (seg['norm']['end'] - seg['norm']['beg'])

        if len_ratio < (1 / 5) or len_ratio > 5:
            continue

        reco_chunk = reco_words[seg['reco']['beg']:seg['reco']['end']]
        norm_chunk = norm_words[seg['norm']['beg']:seg['norm']['end']]

        m = _find_matching_seq_words(reco_chunk, norm_chunk)
        if not m:
            continue
        (rb, re), (nb, ne) = m

        reco_text = ' '.join([x.text for x in reco_words[rb:re]])
        norm_text = ' '.join([x.text for x in norm_words[nb:ne]])
        score = distance(reco_text, norm_text) / len(norm_text)

        if score < ib_score_th and reco_words[re - 1].end - reco_words[rb].start < max_len:
            segs.append({'reco': {'beg': rb, 'end': re}, 'norm': {'beg': nb, 'end': ne}})

    return sorted(segs, key=lambda x: x['reco']['beg'])


def seg_stats(segs: List, reco_words: List, norm_words: List):
    print(f'Reco: {sum([x["reco"]["end"] - x["reco"]["beg"] for x in segs]) / len(reco_words):%}')
    print(f'Norm: {sum([x["norm"]["end"] - x["norm"]["beg"] for x in segs]) / len(norm_words):%}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('reco', type=Path)
    parser.add_argument('norm', type=Path)
    parser.add_argument('out', type=Path)

    args = parser.parse_args()

    with open(args.reco) as f:
        reco = json.load(f)

    with open(args.norm) as f:
        norm = json.load(f)

    vocab = {}
    reco_words = []
    norm_words = []
    pos = 0

    for si, s in enumerate(reco):
        for w, t in zip(s['text'].split(), s['words']):
            if w not in vocab:
                vocab[w] = len(vocab)
            reco_words.append(RecoWord(vocab[w], w, t['start'], t['end'], si, pos))
            pos += 1

    pos = 0
    for si, s in enumerate(norm):
        for w in s['norm'].split():
            if w not in vocab:
                vocab[w] = len(vocab)
            norm_words.append(NormWord(vocab[w], w, si, pos))
            pos += 1

    vocab_decode = {v: k for k, v in vocab.items()}

    reco_chunks = reco_to_chunks(reco_words)

    segs = search_initial_matches(reco_words, reco_chunks, norm_words)

    prev = 0
    for seg in sorted(segs, key=lambda x: x['norm']['beg']):
        if seg['norm']['beg'] < prev:
            print('overlap')
        prev = seg['norm']['end']

    print('Stats after initial match:')
    seg_stats(segs, reco_words, norm_words)

    segs = search_between_segs(segs, reco_words, norm_words)

    prev = 0
    for seg in sorted(segs, key=lambda x: x['norm']['beg']):
        if seg['norm']['beg'] < prev:
            print('overlap')
        prev = seg['norm']['end']

    print('Stats after aligning in-between segments:')
    seg_stats(segs, reco_words, norm_words)

    ib_chunks = [(p['reco']['end'], n['reco']['beg']) for p, n in zip(segs[:-1], segs[1:])]
    if segs[0]['reco']['beg'] > 0:
        ib_chunks.insert(0, (0, segs[0]['reco']['beg']))
    if segs[-1]['reco']['end'] < len(reco_words):
        ib_chunks.append((segs[-1]['reco']['end'], len(reco_words)))
    ib_chunks = list(filter(lambda x: x[1] - x[0] > 0, ib_chunks))
    ib_subchunks = []
    for ch in ib_chunks:
        ib_subchunks.extend(reco_to_chunks(reco_words[ch[0]:ch[1]]))
    segs = search_initial_matches(reco_words, ib_subchunks, norm_words, good_th=0.5, segs=segs)

    prev = 0
    for seg in sorted(segs, key=lambda x: x['norm']['beg']):
        if seg['norm']['beg'] < prev:
            print('overlap')
        prev = seg['norm']['end']

    print('Stats after aligning with no sequence constraint:')
    seg_stats(segs, reco_words, norm_words)

    segs = search_between_segs(segs, reco_words, norm_words)

    prev = 0
    for seg in sorted(segs, key=lambda x: x['norm']['beg']):
        if seg['norm']['beg'] < prev:
            print('overlap')
        prev = seg['norm']['end']

    print('Stats after aligning in-between segments one more time:')
    seg_stats(segs, reco_words, norm_words)

    # generate output

    output = []
    for seg in segs:
        reco_chunk = reco_words[seg['reco']['beg']:seg['reco']['end']]
        norm_chunk = norm_words[seg['norm']['beg']:seg['norm']['end']]
        reco_text = ' '.join([x.text for x in reco_chunk])
        norm_text = ' '.join([x.text for x in norm_chunk])
        output.append({'ref_text': norm_text,
                       'ref_offset': {'start': seg['norm']['beg'], 'end': seg['norm']['end']},
                       'reco_text': reco_text,
                       'reco_offset': {'start': seg['reco']['beg'], 'end': seg['reco']['end']},
                       'audio_offsets': [(x.start, x.end) for x in reco_chunk],
                       'cer': distance(reco_text, norm_text) / len(norm_text)})

    with open(args.out, 'w') as f:
        json.dump(output, f, indent=4)
