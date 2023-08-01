import os
from pathlib import Path

import gradio as gr


def greet(text, audio_file_path):
    text = "%s" % text
    audio_file_path = "%s" % audio_file_path
    out_path = Path("scripts/output/audio.wav")
    os.system(f'python scripts/train.py --audio_path {audio_file_path}\
     --output_dir "models"')
    os.system(f'python scripts/cloning_inference.py --model_path "models/microsoft_speecht5_tts_{Path(audio_file_path).stem}"\
     --input_text "{text}" --output_path "{str(out_path)}"')
    return out_path


demo = gr.Interface(
    fn=greet,
    inputs=[gr.Textbox(label='What would you like the voice to say? (max. 2000 characters per request)'),
            gr.Audio(type="filepath", source="upload", label='Upload a voice to clone (max. 50mb)')],
    outputs="audio",
    title="Deep Voice Cloning Tool"
    )
demo.launch()
