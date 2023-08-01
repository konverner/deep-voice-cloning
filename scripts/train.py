import argparse
import json
import os
from pathlib import Path

import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

from deep_voice_cloning.cloning.model import CloningModel
from deep_voice_cloning.transcriber.model import TranscriberModel
from deep_voice_cloning.data.collator import TTSDataCollatorWithPadding
from deep_voice_cloning.data.dataset import get_cloning_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default=None, help="Language of speech samples")
    parser.add_argument("--audio_path", type=str, default=None, help="Path to training audio file")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to output directory for trained model")
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), "training_config.json")) as f:
        training_config = json.load(f)

    if args.lang is not None:
        training_config['lang'] = args.lang
    if args.audio_path is not None:
        training_config['audio_path'] = Path(args.audio_path)
    if args.output_dir is not None:
        training_config['output_dir'] = Path(args.output_dir)

    transcriber_model = TranscriberModel(lang=training_config['lang'])
    cloning_model = CloningModel(lang=training_config['lang'])

    dataset = get_cloning_dataset(training_config['audio_path'], transcriber_model, cloning_model)
    data_collator = TTSDataCollatorWithPadding(processor=cloning_model.processor, model=cloning_model.model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=training_config["output_dir"],
        per_device_train_batch_size=training_config['batch_size'],
        gradient_accumulation_steps=2,
        overwrite_output_dir=True,
        learning_rate=training_config['learning_rate'],
        warmup_steps=training_config['warmup_steps'],
        max_steps=training_config['max_steps'],
        gradient_checkpointing=True,
        fp16=transcriber_model.device == torch.device("cuda"),
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        save_strategy="no",
        eval_steps=100,
        logging_steps=20,
        load_best_model_at_end=False,
        greater_is_better=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=cloning_model.model,
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=data_collator,
        tokenizer=cloning_model.processor.tokenizer,
    )

    trainer.train()
    cloning_model.save_pretrained(Path(training_config["output_dir"]) /
                                  Path(cloning_model.config['model_path'].replace('/', '_')
                                       + '_' + Path(training_config['audio_path']).stem)
                                  )
