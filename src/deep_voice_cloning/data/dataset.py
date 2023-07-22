from typing import Dict, Any

import torch
import librosa
import numpy as np
from datasets import Dataset

from ..cloning.model import CloningModel
from ..transcriber.model import TranscriberModel


def prepare_dataset(example: Dict[str, Any], model: CloningModel) -> Dict[str, Any]:
    """
    Prepare a single example for training
    """
    # feature extraction and tokenization
    processed_example = model.processor(
        text=example["normalized_text"],
        audio_target=example["audio"]["array"],
        sampling_rate=16000,
        return_attention_mask=False,
    )

    # strip off the batch dimension
    if len(torch.tensor(processed_example['input_ids']).shape) > 1:
        processed_example['input_ids'] = processed_example['input_ids'][0]

    processed_example["labels"] = processed_example["labels"][0]

    # use SpeechBrain to obtain x-vector
    processed_example["speaker_embeddings"] = model.create_speaker_embedding(
        torch.tensor(example["audio"]["array"])
    ).numpy()

    return processed_example


def get_cloning_dataset(input_audio_path: str,
                        transcriber_model: TranscriberModel,
                        cloning_model: CloningModel,
                        sampling_rate: int = 16000,
                        window_size_secs: int = 5) -> Dataset:
    """
    Create dataset by transcribing an audio file using a pretrained Wav2Vec2 model.
    """
    speech_array, _ = librosa.load(input_audio_path, sr=sampling_rate)

    # split a waveform into splits of 5 secs each
    speech_arrays = np.split(speech_array, range(0, len(speech_array), window_size_secs * sampling_rate))[1:]
    texts = [transcriber_model.forward(speech_array, sampling_rate=sampling_rate)
             for speech_array in speech_arrays]

    dataset = Dataset.from_list([
        {'audio': {'array': speech_arrays[i]}, 'normalized_text': texts[i]}
        for i in range(len(speech_arrays))]
    )

    dataset = dataset.map(
        prepare_dataset, fn_kwargs={'model': cloning_model},
        remove_columns=dataset.column_names,
    )

    return dataset
