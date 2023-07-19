import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

from datasets import Dataset
from tqdm import tqdm
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

from src.data_loaders import load_audio_gen
from src.vad import process_vad

logger = logging.getLogger(__name__)


def recognize(audio: Path, vad_json: Path, model: str, batch_size: int = 16, db_floor: float = -45) -> Dict[
    str, Dict]:
    audio_list = [audio]

    logger.info('Loading data...')

    audio = Dataset.from_generator(load_audio_gen, gen_kwargs={'files': audio_list}).with_format('np')
    audio_data = audio[0]['input_values']
    audio_fs = audio[0]['samp_freq']

    logger.info('Loaded audio files!')

    def seg_audio():
        with open(vad_json, 'rb') as f:
            for l in f:
                seg = json.loads(l.strip())
                beg = int(seg['start'] * audio_fs)
                end = int(seg['end'] * audio_fs)
                seg['input_values'] = audio_data[beg:end]
                yield seg

    data = Dataset.from_generator(seg_audio).with_format('np')

    logger.info(f'Loaded {len(data)} segments!')

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
    parser.add_argument('audio', type=Path, help='Audio WAV file.')
    parser.add_argument('vad', type=Path, help='JSON with VAD segments.')
    parser.add_argument('model', type=str)
    parser.add_argument('out_json', type=Path)
    parser.add_argument('--batch-size', '-b', default=16, type=int)

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    logger.setLevel(logging.INFO)

    logger.info(f'Starting recognition of {args.audio} with segments in {args.vad} '
                f'using {args.model} and saving result to {args.out_json}...')

    data = recognize(args.audio, args.vad, args.model, batch_size=args.batch_size)

    logger.info('Saving output...')

    with open(args.out_json, 'w') as f:
        json.dump(data, f, indent=4)

    logger.info('Done!')
