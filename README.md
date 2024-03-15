# CallSentra

CallSentra - Inno Challenge 2024

## Dev Setup

Make sure poetry is installed.

### Initial Setup

```terminal
poetry install --with dev
poetry run pre-commit install
```

### HuggingFace API Key

The whisperx library uses pyannote on HuggingFace to perform diarization. So we need to get a HuggingFace API key, then create a `.env` file (try running `touch .env`). **After** creating the API key, we should unlock the models that are being used:

[The diarization model](https://huggingface.co/pyannote/speaker-diarization-3.1)

[The segmentation model](https://huggingface.co/pyannote/segmentation-3.0)


### Torch version

Because we use poetry, we have to download `torch` from a url to use the GPU. If you are not on a Windows machine using CUDA 12.1+, follow [this issue](https://github.com/python-poetry/poetry/issues/6409) to find out how to install for your specific machine.

For the published version, we can revert to just using the `poetry add torch` version, or the CPU version.