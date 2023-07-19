import argparse
import json
import logging
import sys
import time
from math import ceil
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union

import numpy as np
import torch
import torchaudio
from Levenshtein import editops
from datasets import Dataset, disable_caching
from pympi import TextGrid
from tqdm import tqdm
from transformers import Wav2Vec2Processor, AutoModelForCTC
from transformers.pipelines.automatic_speech_recognition import chunk_iter

from src.data_loaders import Word

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def viterbi(logits: np.ndarray, labels: np.ndarray, pad_id: int = 0) -> Tuple[List[int], np.ndarray]:
    T = logits.shape[0]
    N = len(labels)
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    delta[0, :] = -np.inf
    delta[0, 0] = logits[0][pad_id]

    a = np.zeros((N,))
    b = np.zeros((N,))
    o = np.zeros((N,))
    M = np.arange(N)

    for t in range(1, T):
        delta[t, :] = -np.inf
        beg = np.clip(t - T + N, 0, N - 1)
        end = np.clip(t + 1, 0, N)

        r = np.arange(beg, end)

        a[r] = delta[t - 1][r - 1]
        b[r] = delta[t - 1][r]

        o[r] = logits[t][labels[r]]
        opad = logits[t][pad_id]

        m = np.zeros(N, dtype=bool)
        m[r] = (a > b)[r]
        delta[t][m] = a[m] + o[m]
        psi[t][m] = M[m] - 1

        m = np.zeros(N, dtype=bool)
        m[r] = (a <= b)[r]
        delta[t][m] = b[m] + opad
        psi[t][m] = M[m]

    bestpath = [delta[T - 1].argmax()]
    for t in range(T - 2, -1, -1):
        bestpath.append(psi[t + 1][bestpath[-1]])
    bestpath.reverse()
    return bestpath, delta


def align_logits(logits: List[np.ndarray], labels: List[List[int]],
                 fr_len: float = 0.02, wl_id: int = 4, pad_id: int = 0) -> List[List[Tuple[float, float]]]:
    ret = []
    for log, lab in zip(logits, labels):
        lab = (wl_id,) + tuple(lab) + (wl_id,)
        bestpath, _ = viterbi(log, np.array(lab), pad_id=pad_id)

        dur = []
        ls = 0
        for t, p in enumerate(bestpath):
            if lab[p] == wl_id:
                if t > ls:
                    dur.append((ls * fr_len, t * fr_len))
                ls = t + 1
        ret.append(dur)
    return ret


def align(w2v2_model: str, audio: List[np.ndarray], text: List[str], ids: Optional[List[str]] = None,
          chunk_len: float = 10, left_stride: float = 4, right_stride: float = 2, batch_size: int = 4) -> Dict:
    if not audio:
        logger.warning('No data given!')
        return {}

    logger.info('Loading models...')

    processor = Wav2Vec2Processor.from_pretrained(w2v2_model)
    model = AutoModelForCTC.from_pretrained(w2v2_model).cuda()

    pad_token_id = processor.tokenizer.pad_token_id
    word_delimiter_token_id = processor.tokenizer.word_delimiter_token_id

    align_to = getattr(model.config, "inputs_to_logits_ratio", 1)
    chunk_len = int(round(chunk_len * processor.feature_extractor.sampling_rate / align_to) * align_to)
    stride_left = int(round(left_stride * processor.feature_extractor.sampling_rate / align_to) * align_to)
    stride_right = int(round(right_stride * processor.feature_extractor.sampling_rate / align_to) * align_to)

    start_time = time.time()

    disable_caching()

    logger.info("Loading data...")

    if not ids:
        ids = [f'audio_{x:04d}' for x in range(len(audio))]

    data = Dataset.from_dict({'id': ids, 'input_values': audio, 'ref': text}).with_format('np')

    logger.info(f'Loaded {data.num_rows} files!')

    def process_labels(batch):
        batch["labels"] = processor(text=batch["ref"]).input_ids
        batch["length"] = len(batch['input_values'])
        return batch

    logger.info('Processing labels...')

    data = data.map(process_labels)

    len_s = sum(data["length"]) / 16000
    logger.info(f'Total audio length: {len_s:0.2f}s == {len_s / 60:0.2f}min == {len_s / 3600:0.2f}h')

    def chunking(dataset):
        for sample in dataset:
            for seq, chunk in enumerate(
                    chunk_iter(sample['input_values'], processor.feature_extractor, chunk_len, stride_left,
                               stride_right)):
                yield {'id': sample['id'], 'chunk_seq': seq, 'stride': chunk['stride'],
                       'input_values': chunk['input_values'][0],
                       'attention_mask': chunk['attention_mask'][0]}

    logger.info('Splitting data into chunks...')

    chunks_in = Dataset.from_generator(chunking, gen_kwargs={'dataset': data})

    logger.info(f'Divided into {chunks_in.num_rows} chunks!')

    def process(batch):
        processed = processor.pad({'input_values': batch['input_values'], 'attention_mask': batch['attention_mask']},
                                  padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(input_values=processed['input_values'].cuda(),
                           attention_mask=processed['attention_mask'].cuda()).logits.cpu()

        return {'logits': logits, 'stride': batch['stride']}

    logger.info('Processing chunks using the W2V2 model...')

    chunks_out = chunks_in.map(process, batched=True, batch_size=batch_size)

    ratio = 1 / model.config.inputs_to_logits_ratio

    logger.info('Merging chunks back into files...')

    file_logits = {}
    for chunk in tqdm(chunks_out.sort('chunk_seq')):
        if chunk['id'] not in file_logits:
            file_logits[chunk['id']] = []
        stride = chunk['stride']
        token_n = int(round(stride[0] * ratio))
        left = int(round(stride[1] / stride[0] * token_n))
        right = int(round(stride[2] / stride[0] * token_n))
        right_n = token_n - right
        file_logits[chunk['id']].append(np.array(chunk['logits'])[left:right_n, :])

    def process_alignment(sample):
        labels = processor(text=sample["ref"]).input_ids
        word_lens = align_logits([np.vstack(file_logits[sample['id']])], [labels],
                                 wl_id=word_delimiter_token_id, pad_id=pad_token_id)[0]
        return {'word_dur': word_lens}

    logger.info('Performing forced alignment...')

    data_ali = data.map(process_alignment, batched=False)

    logger.info('Saving output...')

    alignment = {}
    for file in data_ali:
        alignment[file['id']] = [{'text': t, 'timestamp': ts} for t, ts in
                                 zip(file['ref'].split(), file['word_dur'].tolist())]

    logger.info('Done!')

    data.cleanup_cache_files()
    data_ali.cleanup_cache_files()
    chunks_in.cleanup_cache_files()
    chunks_out.cleanup_cache_files()

    took_s = time.time() - start_time
    logger.info(f'Took {took_s:0.2f}s == {took_s / 60:0.2f}min == {took_s / 3600:0.2f}h')

    return alignment


def fix_times(ali: List[Dict], mask: np.ndarray, samp_freq: float):
    ret = []
    times = np.cumsum(mask) / samp_freq
    for seg in ali:
        start = seg['timestamp'][0]
        if start == 0:
            start = 0.01
        end = seg['timestamp'][1]
        dur = end - start
        fix = np.searchsorted(times, start) / samp_freq
        ret.append({'text': seg['text'], 'timestamp': [fix, fix + dur]})
    return ret


def save_ali_to_textgrid(file: Path, ali: List[Dict]):
    tg = TextGrid(xmin=0, xmax=ali[-1]['timestamp'][1])
    tier = tg.add_tier('Words')
    for tok in ali:
        tier.add_interval(tok['timestamp'][0], tok['timestamp'][1], tok['text'])
    tg.to_file(str(file))


def convert_ali_to_segments(ali: List[Dict], reco: List[Word], sil_gap=2, max_len=-1) -> List[Dict]:
    '''
    Converts alignment as a list of words to a list of segments. Also matches the reco words to alignment and computes
    error rates based on the comparison.
    :param ali: alignment as produced by `align` method
    :param reco: reco output to match to the segments
    :param sil_gap: min silence gap used to divide segments
    :param max_len: max length of segments; longer segments are split into roughly equal parts shorter than max_len
    :return: list of segments
    '''
    # combine words into segments based on sil_gap
    segs = []
    if ali:
        seg = [ali[0]]
        for w in ali[1:]:
            if w['timestamp'][0] - seg[-1]['timestamp'][1] > sil_gap:
                segs.append(seg)
                seg = [w]
            else:
                seg.append(w)
        segs.append(seg)

    # split long segments (longer than max_len) into equal sub-segments
    if max_len and max_len > 0:
        sp_seg = []
        for seg in segs:
            dur = seg[-1]['timestamp'][1] - seg[0]['timestamp'][0]
            if dur > max_len:
                seg_dur = dur / ceil(dur / max_len)
                ss = [seg[0]]
                for w in seg[1:]:
                    if w['timestamp'][1] - ss[0]['timestamp'][0] > seg_dur:
                        sp_seg.append(ss)
                        ss = [w]
                    else:
                        ss.append(w)
                sp_seg.append(ss)
            else:
                sp_seg.append(seg)
        segs = sp_seg

    # create output list of dicts
    ret = [{'norm': ' '.join([x['text'] for x in seg]),
            'start': round(seg[0]['timestamp'][0], 3),
            'end': round(seg[-1]['timestamp'][1], 3),
            'words': [{
                'time_s': round(x['timestamp'][0], 3),
                'time_e': round(x['timestamp'][1], 3)
            } for x in seg],
            'reco_words': [],
            'reco': []} for seg in segs]

    # match reco segments to ali
    unaligned_reco = []
    for w in reco:
        found = False
        for seg in ret:
            if seg['start'] <= w.start <= seg['end'] or seg['start'] <= w.end <= seg['end']:
                seg['reco'].append(w)
                found = True
                break
        if not found:
            unaligned_reco.append(w)

    # compute errors
    def get_errors(source: Union[List[str], str], dest: Union[List[str], str]) -> Dict:
        err = {'delete': 0, 'insert': 0, 'replace': 0}
        for op, _, _ in editops(source, dest):
            err[op] += 1
        err['num'] = len(source)
        err['corr'] = err['num'] - err['delete'] - err['replace']
        err['wer'] = round((err['delete'] + err['insert'] + err['replace']) / err['num'], 3)
        return err

    for seg in ret:
        reco = sorted(seg['reco'], key=lambda x: x.start)
        seg['reco'] = ' '.join([x.text for x in reco])
        seg['reco_words'] = [{
            'time_s': round(x.start, 3),
            'time_e': round(x.end, 3)
        } for x in reco]
        seg['errors'] = get_errors(seg['norm'].split(), seg['reco'].split())
        seg['errors']['cer'] = get_errors(seg['norm'], seg['reco'])['wer']

    # combine unaligned reco words into segments
    unaligned_reco = sorted(unaligned_reco, key=lambda x: x.start)
    segs = []
    seg = [unaligned_reco[0]]
    for w in unaligned_reco[1:]:
        if w.start - seg[-1].end > sil_gap:
            segs.append(seg)
            seg = [w]
        else:
            seg.append(w)
    segs.append(seg)

    for seg in segs:
        ret.append({'reco': ' '.join([x.text for x in seg]),
                    'reco_words': [{
                        'time_s': round(x.start, 3),
                        'time_e': round(x.end, 3)
                    } for x in seg],
                    'start': round(seg[0].start, 3),
                    'end': round(seg[-1].end, 3),
                    'unaligned': True})

    return sorted(ret, key=lambda x: x['start'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('audio', type=Path)
    parser.add_argument('text', type=Path)
    parser.add_argument('out_json', type=Path)
    parser.add_argument('out_textgrid', type=Path)
    parser.add_argument('--chunk-len', type=float, default=10)
    parser.add_argument('--left-stride', type=float, default=4)
    parser.add_argument('--right-stride', type=float, default=2)
    parser.add_argument('--batch-size', type=int, default=8)

    args = parser.parse_args()

    audio, fs = torchaudio.load(args.audio)
    audio = audio.flatten().numpy()

    with open(args.text) as f:
        text = f.readline().strip()

    ts = align(args.model, [audio], [text], ids=[args.audio.stem],
               chunk_len=args.chunk_len,
               left_stride=args.left_stride,
               right_stride=args.right_stride,
               batch_size=args.batch_size)

    with open(args.out_json, 'w') as f:
        json.dump(ts, f, indent=4)

    save_ali_to_textgrid(args.out_textgrid, list(ts.values())[0])
