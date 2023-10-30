# Voice Activity Detection

Due to incompatibility of the pyannote library with the rest of the Transformers platform,
it is recommended to create a separate virtual environment for the VAD module or to run the module
using docker or some other virtualization solution.

You will also need to acquire the models, which cannot be downloaded from the hub without logging
in and setting up the [authentication token](https://huggingface.co/settings/tokens).
You can then provide the token to any `from_pretrained` method using the `use_auth_token` argument.

Before downloading any pyannote model, you will need to visit their [hub page](https://huggingface.co/pyannote/segmentation-3.0)
and accept their terms of use while being logged in.

Alternatively, you can simply download the `pytorch_model.bin` file from the hub
and provide the path to the file as the `--model` argument. Please don't share the
model files if you download them this way, but this will make your life easier if you need
to deploy the VAD module to a separate (or multiple) GPU machines.