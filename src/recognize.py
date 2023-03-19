import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

from src.data_loaders import load_audio_gen

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def recognize(audio: Path, model: str, perform_vad: bool = True, batch_size: int = 16, db_floor: float = -45) -> Dict[
    str, Dict]:
    if audio.is_dir():
        audio_list = list(audio.glob('*.wav'))
    else:
        audio_list = [audio]

    logger.info('Loading data...')

    data = Dataset.from_generator(load_audio_gen, gen_kwargs={'files': audio_list}).with_format('np')

    logger.info(f'Loaded {len(data)} files!')

    if perform_vad:
        def process_vad(dataset):
            logger.info('Loading VAD model...')
            model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
            get_speech_timestamps = utils[0]
            logger.info('Processing VAD...')
            for file in tqdm(dataset, total=len(dataset)):
                wav = file['input_values']
                fs = file['samp_freq']
                for i, ts in enumerate(get_speech_timestamps(wav, model, sampling_rate=fs,
                                                             min_silence_duration_ms=500,
                                                             speech_pad_ms=200)):
                    fid = file['id']
                    beg = ts['start']
                    end = ts['end']
                    seg = wav[beg:end]
                    yield {
                        'input_values': seg,
                        'id': f'{fid}_seg_{i:04d}',
                        'reco_id': fid,
                        'samp_freq': fs,
                        'length': len(seg),
                        'rms': 20 * np.log10(np.sqrt(np.mean(seg ** 2))),
                        'start': beg / fs,
                        'end': end / fs
                    }

        data = Dataset.from_generator(process_vad, gen_kwargs={'dataset': data}).with_format('np'). \
            filter(lambda x: x['rms'] >= db_floor)

        logger.info(f'Obtained {len(data)} segments!')

    kd = KeyDataset(data, 'input_values')

    logger.info('Loading ASR model...')

    pipe = pipeline('automatic-speech-recognition', model, device=0, chunk_length_s=10)

    logger.info('Starting recognition...')

    ret = {}
    count = 0
    for out in tqdm(pipe(kd, return_timestamps='word', batch_size=batch_size), total=data.shape[0]):
        seg = data[count]
        ret[seg['id']] = {
            'reco_id': seg['reco_id'],
            'samp_freq': float(seg['samp_freq']),
            'length': int(seg['length']),
            'rms': float(seg['rms']),
            'start': float(seg['start']),
            'end': float(seg['end']),
            'text': out['text']
        }
        if 'chunks' in out:
            ret[seg['id']]['chunks'] = out['chunks']
        count += 1

    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audio', type=Path, help='Audio file or directory containing WAV files.')
    parser.add_argument('model', type=str)
    parser.add_argument('out_json', type=Path)

    args = parser.parse_args()

    logger.info(f'Starting recognition of {args.audio} using {args.model} and saving result to {args.out_json}...')

    data = recognize(args.audio, args.model)

    logger.info('Saving output...')

    with open(args.out_json, 'w') as f:
        json.dump(data, f, indent=4)

    logger.info('Done!')
