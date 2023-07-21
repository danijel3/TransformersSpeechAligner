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
python -m src.matching [reco.json] [reco ID] [audio file] [transcript.txt] [model name or path] [output.json]
```

This will perform a fuzzy match of the ASR output to the transcript then re-align the matched text to the audio using
the provided model and output its results to a JSON file. The second argument is the ID of the file in the JSON file
given in the first argument, because that JSON file can contain multiple recordings and this script is used to process
files one at a time.

## Parlamint corpus procedure

The difference here is that the text in the corpus is not normalized and each line is prefixed by an ID. We want to both
normalize the text, but also keep the alignment with the original text and group words according to the original lines
and their IDs.

1. Recognize the audio (like above):

```bash
python -m src.recognize [audio file(s)] [model name or path] [reco.json]
```

2. Normalize the text to a JSON format:

```bash
python -m src.normalize [transcript.txt] [transcript.json]
```

The file will have the following format:

```json
{
  "id": "ID of utterance",
  "text": "Original text as present in corpus.",
  "norm": "normalized lowercased text without punctuation",
  "normoff": [[0, 5], [6, 10], ...]
}
```
The `normoff` field contains for each normalized word a list of character offsets within the original text.

3. Match the ASR output to the transcript (again same as before):

```bash
python -m src.matching_parlamint [reco.json] [reco ID] [audio file] [transcript.json] [model name or path] [output.json]
```

### Output file

The final output JSON will contain the following fields:

```json
{
  "id": "ID of utterance",
  "text": "Original text as present in corpus.",
  "norm": "normalized lowercased text without punctuation",
  "reco": "ASR output",
  "start": 0.0,
  "end": 5.37,
  "words": [
    {
      "time_s": 0.0,
      "time_e": 0.5,
      "char_s": 0,
      "char_e": 5
    },
    {
      "time_s": 0.6,
      "time_e": 1.2,
      "char_s": 6,
      "char_e": 10
    },
    ...
  ],
  "reco_words": [
    {
      "time_s": 0.0,
      "time_e": 0.45
    },
    {
      "time_s": 0.52,
      "time_e": 1.1
    },
    ...
  ],
  "errors": {
    "delete": 0,
    "insert": 1,
    "replace": 2,
    "num": 5,
    "corr": 2,
    "wer": 0.6,
    "cer": 0.43
  }
}
```

## Details

To learn more about the internals of the procedure, please refer to the [Procedure.ipynb](Procedure.ipynb) notebook.

To learn about the process of text to speech alignment using Wav2Vec2 models please refer to
the [Alignment.ipynb](Alignment.ipynb) notebook.