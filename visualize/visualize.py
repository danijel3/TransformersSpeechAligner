import argparse
import json
import os
import re
from pathlib import Path

from flask import Flask, render_template, make_response, request, abort

app = Flask('visualize')

audio = None
audio_size = 0

annotations = None


def load_data(wav: Path, json_file: Path):
    global audio, audio_size, annotations
    with open(wav, 'rb') as f:
        audio = f.read()
        audio_size = len(audio)

    with open(json_file) as f:
        annotations = json.load(f)


if 'WAV_PATH' in os.environ and 'JSON_PATH' in os.environ:
    load_data(Path(os.environ['WAV_PATH']), Path(os.environ['JSON_PATH']))


@app.route('/')
def index():
    if not audio or not annotations:
        return abort(404)

    stats = {'ref_only_seg': 0, 'ref_only_words': 0,
             'reco_only_seg': 0, 'reco_only_words': 0,
             'ok_match_seg': 0, 'ok_match_words': 0}
    corr = 0
    num = 0
    for seg in annotations:
        if seg['match_error'] == 'only in reference':
            stats['ref_only_seg'] += 1
            stats['ref_only_words'] += len(seg['text'].split())
        elif seg['match_error'] == 'only in reco':
            stats['reco_only_seg'] += 1
            stats['reco_only_words'] += len(seg['reco'].split())
        else:
            stats['ok_match_seg'] += 1
            stats['ok_match_words'] += len(seg['norm'].split())

        if 'errors' in seg:
            corr += seg['errors']['corr']
            num += seg['errors']['num']

    stats['overall_wer'] = round(100 * (num - corr) / num, 2)

    annot_filt = annotations
    if 'noref' in request.args:
        annot_filt = list(filter(lambda x: x['match_error'] != 'only in reference', annotations))

    return render_template('visualize.html', audio=audio, annotations=annot_filt, stats=stats)


@app.route('/audio')
def get_audio():
    headers = []
    end = audio_size - 1
    if 'Range' in request.headers:
        status = 206
        headers.append(('Accept-Ranges', 'bytes'))
        ranges = re.findall(r'\d+', request.headers['Range'])
        begin = int(ranges[0])
        if len(ranges) > 1:
            end = int(ranges[1])
        headers.append(('Content-Range', f'bytes {begin}-{end}/{audio_size}'))
        buf = audio[begin:end + 1]
    else:
        status = 200
        buf = audio

    response = make_response(buf)
    response.status = status
    response.headers['Content-Type'] = 'audio/x-wav'
    response.headers['Content-Disposition'] = 'inline; filename=audio.wav'
    for k, v in headers:
        response.headers[k] = v
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=Path)
    parser.add_argument('wav', type=Path)

    args = parser.parse_args()

    load_data(args.wav, args.json)

    app.run()
