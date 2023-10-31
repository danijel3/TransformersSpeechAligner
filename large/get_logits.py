import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from datasets import Dataset, disable_caching
from transformers import AutoProcessor, AutoModelForCTC
from transformers.pipelines.automatic_speech_recognition import chunk_iter

logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')
logger = logging.getLogger()


def get_logits(audio_file: Path, model_name: str, chunk_len: float = 10, left_stride: float = 4,
               right_stride: float = 2, device: str = 'cuda', batch_size: int = 8):
    wav, fs = torchaudio.load(audio_file)
    audio_dur = wav.shape[1] / fs
    data = Dataset.from_list([{'input_values': wav.flatten()}]).with_format('np')

    logger.info(f'Loading audio from {audio_file} ({audio_dur:.2f}s).')

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCTC.from_pretrained(model_name).to(device)

    align_to = getattr(model.config, "inputs_to_logits_ratio", 1)
    chunk_len = int(round(chunk_len * processor.feature_extractor.sampling_rate / align_to) * align_to)
    stride_left = int(round(left_stride * processor.feature_extractor.sampling_rate / align_to) * align_to)
    stride_right = int(round(right_stride * processor.feature_extractor.sampling_rate / align_to) * align_to)

    def chunking():
        for sample in data:  # iterate single file
            for seq, chunk in enumerate(
                    chunk_iter(sample['input_values'], processor.feature_extractor, chunk_len, stride_left,
                               stride_right)):
                yield {'chunk_seq': seq, 'stride': chunk['stride'],
                       'input_values': chunk['input_values'][0],
                       'attention_mask': chunk['attention_mask'][0]}

    chunks = Dataset.from_generator(chunking)

    data.cleanup_cache_files()

    logger.info(f'Split file into {data.num_rows} chunks.')

    def process_logits(inputs):
        processed = processor.pad({'input_values': inputs['input_values'], 'attention_mask': inputs['attention_mask']},
                                  padding=True, return_tensors="pt")
        out = model(input_values=torch.Tensor(processed['input_values']).to(device),
                    attention_mask=torch.Tensor(processed['attention_mask']).to(device))
        return {'logits': out.logits.cpu()}

    logger.info(f'Computing logits with batch size {batch_size}...')
    s = time.perf_counter()

    with torch.no_grad():
        logits = chunks.map(process_logits, batched=True, batch_size=batch_size)

    chunks.cleanup_cache_files()

    dur = time.perf_counter() - s
    logger.info(f'Took {dur:.2f}s which is {dur / audio_dur:.2f}x realtime.')

    logger.info('Merging chunks together...')

    ratio = 1 / model.config.inputs_to_logits_ratio
    logits = logits.sort('chunk_seq')
    strides = logits['stride']
    full = []
    for i, lit in enumerate(logits['logits']):
        stride = strides[i]
        token_n = int(round(stride[0] * ratio))
        left = int(round(stride[1] / stride[0] * token_n))
        right = int(round(stride[2] / stride[0] * token_n))
        right_n = token_n - right
        full.append(np.array(lit)[left:right_n, :])

    logits.cleanup_cache_files()

    return np.vstack(full)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_file', type=Path)
    parser.add_argument('model_name', type=str)
    parser.add_argument('output_file', type=Path)
    parser.add_argument('--chunk_len', type=float, default=10)
    parser.add_argument('--left_stride', type=float, default=4)
    parser.add_argument('--right_stride', type=float, default=2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()

    disable_caching()

    logits = get_logits(args.audio_file, args.model_name, args.chunk_len, args.left_stride, args.right_stride,
                        args.device, args.batch_size)

    np.savez_compressed(args.output_file, logits=logits)
