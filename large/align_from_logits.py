import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

from src.align import viterbi


def extract_logits(logits: np.ndarray, words: List[Tuple], lps: float = 50, margin: float = 0.25) -> Tuple[
    np.ndarray, np.ndarray, int]:
    offset = int((words[0][0] - margin) * lps)
    if offset < 0:
        offset = 0
    length = int((words[-1][1] + margin) * lps) - offset
    if offset + length > logits.shape[0]:
        length = logits.shape[0] - offset
    segment = logits[offset:offset + length]
    mask = np.zeros(segment.shape[0], dtype=bool)
    for word in words:
        s = int((word[0] - margin) * lps) - offset
        e = int((word[1] + margin) * lps) - offset
        mask[s:e] = True
    return segment[mask, :], mask, offset


def align(match: List, logits: np.ndarray, vps: float = 50, pad_token_id: int = 0,
          word_delimiter_token_id: int = 4) -> List:
    ali = []
    for seg in tqdm(match):
        l, m, o = extract_logits(logits, seg['audio_offsets'])
        labels = processor(text=seg['ref_text'])['input_ids']
        words = seg['ref_text'].split()
        pos = list(range(seg['ref_offset']['start'], seg['ref_offset']['end']))
        labels = (word_delimiter_token_id,) + tuple(labels) + (word_delimiter_token_id,)
        bestpath, _ = viterbi(l, np.array(labels), pad_id=pad_token_id)

        times = np.cumsum(m) / vps
        o = o / vps

        ls = 0
        wc = 0
        for t, p in enumerate(bestpath):
            if p >= 0 and labels[p] == word_delimiter_token_id:
                if t > ls:
                    start = ls / vps
                    end = t / vps
                    dur = end - start
                    fix = np.searchsorted(times, start) / vps
                    ali.append({'text': words[wc], 'pos': pos[wc], 'start': fix + o, 'end': fix + dur + o})
                    wc += 1
                ls = t + 1

    ali = sorted(ali, key=lambda x: x['start'])
    ret = [ali[0]]
    for w in ali[1:]:
        if w['start'] >= ret[-1]['end']:
            ret.append(w)
    return ret


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('match', type=Path)
    parser.add_argument('logits', type=Path)
    parser.add_argument('model', type=str)
    parser.add_argument('output', type=Path)
    parser.add_argument('--vps', type=int, default=50, help='provide -1 to recalculate from model (slow)')

    args = parser.parse_args()

    with open(args.match) as f:
        match = json.load(f)

    logits = np.load(args.logits)['logits']

    processor = AutoProcessor.from_pretrained(args.model)

    pad_token_id = processor.tokenizer.pad_token_id
    word_delimiter_token_id = processor.tokenizer.word_delimiter_token_id

    vps = args.vps
    if vps < 0:
        processor = AutoProcessor.from_pretrained(args.w2v2_model)
        model = AutoModel.from_pretrained(args.w2v2_model)
        fps = processor.feature_extractor.sampling_rate / model.config.num_codevectors_per_group

    ali = align(match, logits, vps, pad_token_id, word_delimiter_token_id)

    with open(args.output, 'w') as f:
        json.dump(ali, f, indent=4)
