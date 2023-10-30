import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection


def run(audio: Path, output: Path,
        model: str = 'pytorch_model.bin', min_dur: float = 1.0, min_gap: float = 1.0, db_floor: float = -45.0):
    model = Model.from_pretrained(model)

    pipeline = VoiceActivityDetection(segmentation=model).to(torch.device('cuda'))
    pipeline.instantiate({
        "min_duration_on": min_dur,  # remove speech regions shorter than that many seconds.
        "min_duration_off": min_gap  # fill non-speech regions shorter than that many seconds.
    })

    s = time.perf_counter()
    vad = pipeline(str(audio))
    print(f'Took {time.perf_counter() - s} seconds')

    segs = [(x.start, x.end) for x in vad.itersegments()]

    wav, fs = torchaudio.load(audio)
    wav = wav.flatten().numpy()

    def rms(seg):
        s = int(seg[0] * fs)
        e = int(seg[1] * fs)
        return 20 * np.log10(np.sqrt(np.mean(wav[s:e] ** 2)))

    segs = list(filter(lambda x: rms(x) >= db_floor, segs))

    with open(output, 'w') as f:
        json.dump(segs, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('audio', type=Path, help='Path to audio file')
    parser.add_argument('output', type=Path, help='Path to output JSON file')
    parser.add_argument('--model', type=str, default='pytorch_model.bin', help='Path or hub URL to model file')
    parser.add_argument('--min-dur', type=float, default=1.0, help='Minimum duration of speech segments')
    parser.add_argument('--min-gap', type=float, default=1.0, help='Minimum duration of non-speech segments')
    parser.add_argument('--db-floor', type=float, default=-45.0, help='Minimum RMS dB of speech segments')

    args = parser.parse_args()

    run(args.audio, args.output, args.model, args.min_dur, args.min_gap, args.db_floor)
