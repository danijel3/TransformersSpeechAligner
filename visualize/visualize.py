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
    return render_template('visualize.html', audio=audio, annotations=annotations)


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
