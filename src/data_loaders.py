import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torchaudio


def load_audio(file: Path):
    fid = file.stem
    wav, fs = torchaudio.load(file)
    wav = wav.flatten()
    np_wav = wav.numpy()
    return {'input_values': wav,
            'id': fid,
            'samp_freq': fs,
            'lenght': np_wav.size / fs}


def load_audio_gen(files: List[Path]):
    for file in files:
        yield load_audio(file)


def extract_audio(audio: np.ndarray, words: List, fs: float, margin: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(audio.shape, dtype=bool)
    for word in words:
        s = int((word.start - margin) * fs)
        e = int((word.end + margin) * fs)
        mask[s:e] = True
    return audio[mask], mask


@dataclass
class Word:
    text: str
    start: float
    end: float


def load_reco(file: Path) -> Dict[str, List[Word]]:
    with open(file) as f:
        segs = json.load(f)

    words = {}
    for seg in segs.values():
        rid = seg['reco_id']
        if rid not in words:
            words[rid] = []
        for word in seg['chunks']:
            off = seg['start']
            words[rid].append(Word(word['text'], off + word['timestamp'][0], off + word['timestamp'][1]))
    return words
