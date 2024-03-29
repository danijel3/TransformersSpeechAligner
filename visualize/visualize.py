import argparse
import json
import re
from pathlib import Path
from typing import List, Dict

from flask import Flask, render_template, make_response, request, abort, jsonify, redirect

app = Flask('visualize')

settings_file = Path('settings.json')
index_file = Path('index.json')


@app.route('/')
def index():
    audio_dir = Path('debug-croatian/wav')
    json_dir = Path('debug-croatian/output')

    if settings_file.exists():
        with open(settings_file) as f:
            settings = json.load(f)
            if 'audio_dir' in settings:
                audio_dir = Path(settings['audio_dir'])
            if 'json_dir' in settings:
                json_dir = Path(settings['json_dir'])

    files = []

    if index_file.exists():
        with open(index_file) as f:
            files = json.load(f)

    return render_template('index.html', files=files, audio_dir=audio_dir, json_dir=json_dir)


def get_stats(annotations: List) -> Dict:
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

    if num > 0:
        stats['overall_wer'] = round(100 * (num - corr) / num, 2)
    else:
        stats['overall_wer'] = 0
    return stats


@app.route('/reindex', methods=['POST'])
def reindex():
    settings = {}
    audio_dir = Path('debug-croatian/wav')
    json_dir = Path('debug-croatian/output')

    if settings_file.exists():
        with open(settings_file) as f:
            settings = json.load(f)
            if 'audio_dir' in settings:
                audio_dir = Path(settings['audio_dir'])
            if 'json_dir' in settings:
                json_dir = Path(settings['json_dir'])

    if 'audio_dir' in request.form:
        audio_dir = Path(request.form['audio_dir'])
    if 'json_dir' in request.form:
        json_dir = Path(request.form['json_dir'])

    settings['audio_dir'] = str(audio_dir)
    settings['json_dir'] = str(json_dir)
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=4)

    utt_to_date = {}
    if Path('utt_to_date.json').exists():
        with open('utt_to_date.json') as f:
            utt_to_date = json.load(f)

    yt = re.compile(r'[0-9a-zA-Z\_\-]{11}')
    files = []
    for json_f in json_dir.glob('*.json'):
        utt = {'utt': json_f.stem}
        if utt['utt'] in utt_to_date:
            utt['date'] = utt_to_date[utt['utt']]
        else:
            utt['date'] = ''

        m = yt.findall(utt['utt'])
        if m:
            utt['yt'] = m[-1]
        else:
            utt['yt'] = ''

        with open(json_f) as f:
            annotations = json.load(f)
        utt.update(get_stats(annotations))
        files.append(utt)

    with open(index_file, 'w') as f:
        json.dump(files, f, indent=4)

    return redirect('/')


@app.route('/visualize/<utt>', defaults={'yt_id': None})
@app.route('/visualize/<utt>/yt/<yt_id>')
def visualize(utt, yt_id):
    audio_dir = Path('debug-croatian/wav')
    json_dir = Path('debug-croatian/output')

    if settings_file.exists():
        with open(settings_file) as f:
            settings = json.load(f)
            if 'audio_dir' in settings:
                audio_dir = Path(settings['audio_dir'])
            if 'json_dir' in settings:
                json_dir = Path(settings['json_dir'])

    audio_file = audio_dir / (utt + '.wav')
    json_file = json_dir / (utt + '.json')

    if not json_file.exists():
        abort(404)

    if not yt_id and not audio_file.exists():
        abort(404)

    with open(json_file) as f:
        annotations = json.load(f)

    stats = get_stats(annotations)

    annot_filt = annotations
    if 'noref' in request.args:
        annot_filt = list(filter(lambda x: x['match_error'] != 'only in reference', annotations))

    doc = 'audio_player.html'
    if yt_id:
        doc = 'youtube_player.html'

    return render_template(doc, utt=utt, annotations=annot_filt, stats=stats, yt=yt_id)


@app.route('/audio/<utt>')
def get_audio(utt):
    audio_dir = Path('debug-croatian/wav')

    if settings_file.exists():
        with open(settings_file) as f:
            settings = json.load(f)
            if 'audio_dir' in settings:
                audio_dir = Path(settings['audio_dir'])

    audio_file = audio_dir / (utt + '.wav')

    if not audio_file.exists():
        abort(404)

    headers = []
    if 'Range' in request.headers:
        audio_size = audio_file.stat().st_size
        end = audio_size - 1
        status = 206
        headers.append(('Accept-Ranges', 'bytes'))
        ranges = re.findall(r'\d+', request.headers['Range'])
        begin = int(ranges[0])
        if len(ranges) > 1:
            end = int(ranges[1])
        headers.append(('Content-Range', f'bytes {begin}-{end}/{audio_size}'))
        with open(audio_file, 'rb') as f:
            f.seek(begin)
            buf = f.read(end - begin + 1)
    else:
        status = 200
        with open(audio_file, 'rb') as f:
            buf = f.read()

    response = make_response(buf)
    response.status = status
    response.headers['Content-Type'] = 'audio/x-wav'
    response.headers['Content-Disposition'] = 'inline; filename=audio.wav'
    for k, v in headers:
        response.headers[k] = v
    return response


if __name__ == '__main__':
    app.run()
