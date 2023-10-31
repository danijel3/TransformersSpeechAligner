import argparse
import json
from pathlib import Path
from typing import List, Dict, Union

from Levenshtein import editops


def get_errors(source: Union[List[str], str], dest: Union[List[str], str]) -> Dict:
    err = {'delete': 0, 'insert': 0, 'replace': 0}
    for op, _, _ in editops(source, dest):
        err[op] += 1
    err['num'] = len(source)
    if err['num'] == 0:
        err['num'] = 1
    err['corr'] = err['num'] - err['delete'] - err['replace']
    err['wer'] = round((err['delete'] + err['insert'] + err['replace']) / err['num'], 3)
    return err


def convert_ali_to_corpus_lines(ali: List[Dict], reco: List[Dict], norm: List[Dict], sil_gap: float = 2) -> List[Dict]:
    segs = []  # final segments list
    norm_words = []  # map of words to line and position in line
    # create list of segments with ids and original (reference) text
    for ln, l in enumerate(norm):
        segs.append({
            'id': l['uid'],
            'text': l['text'],
            'w': []
        })
        for n in l['normoff']:
            norm_words.append((ln, n))

    # add words from alignment to segments
    for w in ali:
        ln, n = norm_words[w['pos']]
        segs[ln]['w'].append((w, n))

    # get time start/end for each word in alignment with text of the word in norm
    # and position of the word in reference text by char start/end
    for s in segs:
        words = sorted(s['w'], key=lambda x: x[0]['start'])
        if not words:
            continue
        s['norm'] = ' '.join([x[0]['text'] for x in words])
        s['words'] = [{
            'time_s': round(x[0]['start'], 3),
            'time_e': round(x[0]['end'], 3),
            'char_s': x[1][0],
            'char_e': x[1][1],
        } for x in words]
        s['start'] = s['words'][0]['time_s']
        s['end'] = s['words'][-1]['time_e']
        s['w'] = []

    # add words from recognition to segments
    unaligned = []
    for w in reco:
        mp = (w['start'] + w['end']) / 2
        found = False
        for s in segs:
            if 'start' in s and s['start'] <= mp <= s['end']:
                s['w'].append(w)
                found = True
                break
        if not found:
            unaligned.append(w)

    # for reco words add start/end times and compute WER
    for s in segs:
        words = sorted(s['w'], key=lambda x: x['start'])
        if not words:
            continue
        s['reco'] = ' '.join([x['text'] for x in words])
        s['reco_words'] = [{
            'time_s': round(x['start'], 3),
            'time_e': round(x['end'], 3),
        } for x in words]
        s.pop('w')
        s['errors'] = get_errors(s['norm'].split(), s['reco'].split())
        s['errors']['cer'] = get_errors(s['norm'], s['reco'])['wer']
        s['match_error'] = None

    # combine unaligned reco words into segments
    unaligned = sorted(unaligned, key=lambda x: x['start'])
    unaligned_segs = []
    if unaligned:
        seg = [unaligned[0]]
        for w in unaligned[1:]:
            if w['start'] - seg[-1]['end'] > sil_gap:
                unaligned_segs.append(seg)
                seg = [w]
            else:
                seg.append(w)
        unaligned_segs.append(seg)

    # add unaligned segments to final segments list
    for seg in unaligned_segs:
        segs.append({'reco': ' '.join([x['text'] for x in seg]),
                     'reco_words': [{
                         'time_s': round(x['start'], 3),
                         'time_e': round(x['end'], 3)
                     } for x in seg],
                     'start': round(seg[0]['start'], 3),
                     'end': round(seg[-1]['end'], 3),
                     'match_error': 'only in reco'})

    for seg in segs:
        if 'norm' not in segs and 'match_error' not in seg:
            seg['match_error'] = 'only in reference'
            seg.pop('w')

    segs = sorted(segs, key=lambda x: x['start'] if 'start' in x else 0)

    lk = ''
    for s in segs:
        if 'id' in s:
            s['k'] = s['id']
            lk = s['id']
        else:
            s['k'] = lk
        s['k'] = s['k'].split('.')
        s['k'][1] = int(s['k'][1][1:])

    segs = sorted(segs, key=lambda x: x['k'])

    for s in segs:
        s.pop('k')

    for s in segs:
        if 'reco' in s and 'norm' in s:
            s['errors'] = get_errors(s['norm'].split(), s['reco'].split())
            s['errors']['cer'] = get_errors(s['norm'], s['reco'])['wer']

    return segs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('ali', type=Path, help='Path to alignment JSON file')
    parser.add_argument('reco', type=Path, help='Path to recognition JSON file')
    parser.add_argument('norm', type=Path, help='Path to reference/normalization JSON file')
    parser.add_argument('output', type=Path, help='Path to output JSON file')

    args = parser.parse_args()

    with open(args.ali) as f:
        ali = json.load(f)

    with open(args.reco) as f:
        reco = []
        for s in json.load(f):
            for i, w in enumerate(s['text'].split()):
                reco.append({'text': w, 'start': s['words'][i]['start'], 'end': s['words'][i]['end']})

    with open(args.norm) as f:
        norm = json.load(f)

    segs = convert_ali_to_corpus_lines(ali, reco, norm)

    with open(args.output, 'w') as f:
        json.dump(segs, f, indent=4)
