import os
import json

import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class TranscriberModel:
    def __init__(self, lang: str = 'en'):
        with open(os.path.join(os.path.dirname(__file__), 'config.json')) as f:
            config = json.load(f)
        self.processor = Wav2Vec2Processor.from_pretrained(config['language_model_names'][lang])
        self.model = Wav2Vec2ForCTC.from_pretrained(config['language_model_names'][lang])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, speech_array: np.array, sampling_rate: int = 16000) -> str:
        model_input = self.processor(speech_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(model_input.input_values, attention_mask=model_input.attention_mask).logits
            predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(predicted_ids)
