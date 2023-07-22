import argparse
import json
import os

import soundfile as sf

from deep_voice_cloning.cloning.model import CloningModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to model directory")
    parser.add_argument("--input_text", type=str, default=None, help="Text to be synthesized")
    parser.add_argument("--output_path", type=str, default=None, help="Path to output audio file")
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), "inference_config.json")) as f:
        config = json.load(f)

    if args.model_path is not None:
        config['model_path'] = args.model_path
    if args.input_text is not None:
        config['input_text'] = args.input_text
    if args.output_path is not None:
        config['output_path'] = args.output_path

    cloning_model = CloningModel(config)
    waveform_array = cloning_model.forward(config["input_text"])

    sf.write(config['output_path'], waveform_array, samplerate=16000)
