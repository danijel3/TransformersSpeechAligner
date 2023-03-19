import argparse
import json
import shutil
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory
from typing import Optional, Tuple

from huggingface_hub import Repository
from pyctcdecode import build_ctcdecoder
from transformers import AutoProcessor, Wav2Vec2ProcessorWithLM


def make_ngram(text_file: Path, lm_dir: Path, bin_path: Optional[Path] = None) -> Tuple[Path, Path]:
    opt_bin = lambda x: bin_path / x if bin_path else x
    run([opt_bin('lmplz'), '-o', '3', '--text', str(text_file), '--arpa', str(lm_dir / 'lm.arpa')])
    run([opt_bin('build_binary'), str(lm_dir / 'lm.arpa'), str(lm_dir / 'lm.binary')])
    with open(lm_dir / 'lm.arpa') as f, open(lm_dir / 'unigrams.txt', 'w') as g:
        header = True
        for l in f:
            if header:
                if l.strip() == '\\1-grams:':
                    header = False
            else:
                if len(l.strip()) == 0:
                    break
                word = l.strip().split()[1]
                if word[0] != '<':
                    g.write(word + '\n')
    return lm_dir / 'lm.binary', lm_dir / 'unigrams.txt'


def update_model(model_in: str, lm: Path, unigrams: Path, model_out: Path):
    Repository(local_dir=str(model_out), clone_from=model_in)

    if (model_out / 'language_model').exists():
        shutil.copy(lm, model_out / 'language_model' / 'lm.binary')
        shutil.copy(unigrams, model_out / 'language_model' / 'unigrams.txt')
    else:
        with open(model_out / 'vocab.json') as f:
            vocab_dict = json.load(f)

        with open(unigrams) as f:
            unigram_list = [x.strip() for x in f]

        sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
        labels = list(sorted_vocab_dict.keys())

        processor = AutoProcessor.from_pretrained(model_out)

        decoder = build_ctcdecoder(labels=labels, kenlm_model_path=str(lm), unigrams=unigram_list)

        processor_with_lm = Wav2Vec2ProcessorWithLM(feature_extractor=processor.feature_extractor,
                                                    tokenizer=processor.tokenizer, decoder=decoder)

        processor_with_lm.save_pretrained(model_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=Path)
    parser.add_argument('model_in', type=str)
    parser.add_argument('model_out', type=Path)
    parser.add_argument('--kenlm-bin', type=Path)

    args = parser.parse_args()

    with TemporaryDirectory() as fp:
        temp_dir = Path(fp)
        lm, unigrams = make_ngram(args.text, temp_dir, bin_path=args.kenlm_bin)
        shutil.copy(temp_dir / 'lm.arpa', Path('.') / 'lm.arpa')
        shutil.copy(lm, Path('.') / 'lm.binary')
        shutil.copy(unigrams, Path('.') / 'unigrams.txt')
        update_model(args.model_in, lm, unigrams, args.model_out)
