import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from pyctcdecode import build_ctcdecoder
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel


def reco(logits: np.ndarray, vad_segments: List, ngram_model: Path, unigrams_list: List, w2v2_model: str,
         output_reco: Path,
         alpha: float = 0.5, beta: float = 1.5, unk_score_offset: float = -10.0,
         beam_width: int = 100, beam_prune_logp: float = -10.0, token_min_logp: float = -5.0,
         vps: int = 50):
    processor = AutoProcessor.from_pretrained(w2v2_model)
    labels = [x[0] for x in sorted(processor.tokenizer.get_vocab().items(), key=lambda x: x[1])]

    if len(unigrams_list) == 0:
        unigrams_list = None

    decoder = build_ctcdecoder(
        labels,
        kenlm_model_path=str(ngram_model),
        unigrams=unigrams_list,
        alpha=alpha,
        beta=beta,
        unk_score_offset=unk_score_offset
    )

    ret = []
    for seg in tqdm(vad_segments):
        start = int(seg[0] * vps)
        end = int(seg[1] * vps)
        output = decoder.decode_beams(logits[start:end], beam_width=beam_width,
                                      beam_prune_logp=beam_prune_logp,
                                      token_min_logp=token_min_logp)
        text, _, words, ac_sc, lm_sc = output[0]
        assert (len(words) == len(text.split())), 'word count doesn\'t match between text and word list'
        offsets = [{'start': seg[0] + (w[1][0] / float(vps)), 'end': seg[0] + (w[1][1] / float(vps))} for w in words]
        ret.append(
            {'start': seg[0], 'end': seg[1], 'text': text, 'words': offsets, 'acoustic_score': ac_sc,
             'lm_score': lm_sc})

    with open(output_reco, 'w') as f:
        json.dump(ret, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logits_npyz', type=Path)
    parser.add_argument('vad_json', type=Path)
    parser.add_argument('ngram_model', type=Path)
    parser.add_argument('unigrams_list', type=Path)
    parser.add_argument('w2v2_model', type=str)
    parser.add_argument('output_reco', type=Path)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1.5)
    parser.add_argument('--unk-score-offset', type=float, default=-10.0)
    parser.add_argument('--vps', type=int, default=50, help='provide -1 to recalculate from model (slow)')

    args = parser.parse_args()

    logits = np.load(args.logits_npyz)['logits']

    with open(args.vad_json) as f:
        vad_segments = json.load(f)

    with open(args.unigrams_list) as f:
        unigrams_list = [x.strip() for x in f]

    vps = args.vps
    if vps < 0:
        processor = AutoProcessor.from_pretrained(args.w2v2_model)
        model = AutoModel.from_pretrained(args.w2v2_model)
        fps = processor.feature_extractor.sampling_rate / model.config.num_codevectors_per_group

    reco(logits, vad_segments, args.ngram_model, unigrams_list, args.w2v2_model, args.output_reco, args.alpha,
         args.beta, args.unk_score_offset, vps)
