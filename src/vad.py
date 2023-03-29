import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm

from src.data_loaders import load_audio_gen

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('audio', type=Path, nargs='+')
    parser.add_argument('--db-floor', type=float, default=-45)

    args = parser.parse_args()

    data = Dataset.from_generator(load_audio_gen, gen_kwargs={'files': args.audio}).with_format('np')

    logger.info(f'Loaded {len(data)} files!')

    data = Dataset.from_generator(process_vad, gen_kwargs={'dataset': data}).with_format('np'). \
        filter(lambda x: x['rms'] >= args.db_floor)
    logger.info(f'Obtained {len(data)} segments!')

    logger.info('Saving to disk...')
    data.remove_columns('input_values').to_json(args.out)

    logger.info('Done!')
