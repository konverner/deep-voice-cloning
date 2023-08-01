# Few-Shot Voice Cloning

This repository is an implementation of the pipeline for few-short voice cloning based on SpeechT5 architecture introduced in [ SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing](https://arxiv.org/abs/2110.07205).
It is able to clone a voice from 15-30 seconds of audio recording in English (another languages are planned).

# Getting Started

Clone repository 
```angular2html
git clone https://github.com/konverner/deep-voice-cloning.git
```

Install the modules
```angular2html
pip install .
```

Run traning specifying arguments using config file `training_config.json` or the console command, for example
```angular2html
python scripts/train.py --audio_path scripts/input/hank.mp3 --output_dir /content/deep-voice-cloning/models
```
Resulting model will be saved in `output_dir` directory. It will be used in the next step.

Run inference specifying arguments using config file `inference_config.json` or the console command, for example
```angular2html
python scripts/cloning_inference.py --model_path "/content/deep-voice-cloning/models/microsoft_speecht5_tts_hank"\
--input_text 'do the things, not because they are easy, but because they are hard'\
--output_path "scripts/output/do_the_things.wav"
```

Resulting audio file will be saved as `output_path` file.

# Docker

To build docker image:

```
docker build -t deep-voice-cloning .
```

To pull docker image from Hub:

```angular2html
docker pull konverner/deep-voice-cloning:latest
```

To run image in a container:

```
docker run -it --entrypoint=/bin/bash konverner/deep-voice-cloning
```

To run training in a container for example:

```
python scripts/train.py --audio_path scripts/input/hank.mp3 --output_dir models
```

To run inference in a container for example:

```
python scripts/cloning_inference.py --model_path models/microsoft_speecht5_tts_hank --input_text "do the things, not because they are easy, but because they are hard" --output_path scripts/output/do_the_things.wav
```


# Notebook Examples

Example of using CLI for training and inference can be found in [notebook](https://github.com/konverner/deep-voice-cloning/blob/main/notebooks/CLI_Example.ipynb)

