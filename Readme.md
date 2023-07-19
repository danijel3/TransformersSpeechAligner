# Transformers Speech Aligner

This repo contains a solution to perform alignment of very long audio files to very long transcripts of text using the
Wav2Vec2 models from the Huggingface Transformers library.

## Procedure

The procedure for processing a file consists of the following steps:

1. Perform ASR on required audio files

```bash
python -m src.recognize [audio file(s)] [model name or path] [reco.json]
```

This will perform VAD to split the file into reasonable chunks and then perform ASR on each chunk. Alternatively, you
can perform VAD first (as it works on CPU and is kinda slow) and save the results to a json file, then run the ASR on
the generated VAD chunks:

```bash
python -m src.vad [vad.json] [audio file(s) ...]
python -m src.recognize_vad [audio file(s)] [vad.json] [model name or path] [reco.json]
```

2. Match the ASR output to a transcript:

```bash
python -m src.matching [reco.json] [reco ID] [transcript.txt] [model name or path] [output.json]
```

This will perform a fuzzy match of the ASR output to the transcript then re-align the matched text to the audio using
the provided model and output its results to a JSON file. The second argument is the ID of the file in the JSON file
given in the first argument, because that JSON file can contain multiple recordings and this script is used to process
files one at a time.

## Details

To learn more about the internals of the procedure, please refer to the [Procedure.ipynb](Procedure.ipynb) notebook.

To learn about the process of text to speech alignment using Wav2Vec2 models please refer to
the [Alignment.ipynb](Alignment.ipynb) notebook.