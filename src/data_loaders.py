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


def extract_audio(audio: np.ndarray, words: List, fs: float, margin: float = 0.25) -> Tuple[
    np.ndarray, np.ndarray, int]:
    """
    Extract audio from given list of words.
    :param audio: whole audio file
    :param words: list of words
    :param fs: sampling frequency
    :param margin: margin in seconds
    :return: extracted audio, boolean mask, mask offset
    """
    offset = int((words[0].start - margin) * fs)
    length = int((words[-1].end + margin) * fs - offset)
    segment = audio[offset:offset + length]
    mask = np.zeros(segment.shape, dtype=bool)
    for word in words:
        s = int((word.start - margin) * fs) - offset
        e = int((word.end + margin) * fs) - offset
        mask[s:e] = True
    return segment[mask], mask, offset


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
