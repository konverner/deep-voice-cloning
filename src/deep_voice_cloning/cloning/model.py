import os
import json
from typing import Dict
from pathlib import Path

import numpy as np
import torch
from speechbrain.pretrained import EncoderClassifier
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan


class CloningModel:
    def __init__(self, config: Dict[str, Dict[str, str]] = None, lang: str = 'en'):
        super(CloningModel, self).__init__()
        if config is None:
            self.speaker_embedding = None
            with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
                self.config = json.load(f)[lang]
        else:
            self.config = config
            self.speaker_embedding = torch.load(Path(self.config['model_path']) / "speaker_embedding.pt")[0]
        self.processor = SpeechT5Processor.from_pretrained(self.config['model_path'])
        self.model = SpeechT5ForTextToSpeech.from_pretrained(self.config['model_path'])
        self.vocoder = SpeechT5HifiGan.from_pretrained(self.config['vocoder_name'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.speaker_model = EncoderClassifier.from_hparams(source=self.config['speaker_model_name'])
        self.to(self.device)



    def to(self, device: torch.device):
        self.model = self.model.to(device)
        self.vocoder = self.vocoder.to(device)

    def save_pretrained(self, save_directory: str):
        self.model.save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory)
        torch.save(self.speaker_embedding, Path(save_directory) / "speaker_embedding.pt")

    def forward(self, text: str) -> np.array:
        # tokenize text
        inputs = self.processor(text=text, return_tensors="pt")
        # generate spectrogram using backbone model
        spectrogram = self.model.generate_speech(inputs["input_ids"].to(self.device),
                                                 self.speaker_embedding.to(self.device))
        # decode spectrogram into waveform using vocoder
        with torch.no_grad():
            waveform_array = self.vocoder(spectrogram).detach().cpu().numpy()
        return waveform_array

    def create_speaker_embedding(self, waveform: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            speaker_embeddings = self.speaker_model.encode_batch(waveform)
            speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
            self.speaker_embedding = speaker_embeddings
            speaker_embeddings = speaker_embeddings.squeeze()
        return speaker_embeddings
