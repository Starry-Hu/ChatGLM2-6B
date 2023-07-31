#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import copy
import gc
import logging
import os
import sys
from pathlib import Path

from datasets import load_dataset

import torch

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed, DefaultDataCollator,
)

# BASE_DIR = Path(__file__).resolve().parents[2]
# print(f"base_dir: {BASE_DIR}")

sys.path.append("../")

from ptuning_fed.param_efficient import use_lora

from configs.arguments import ModelArguments, DataTrainingArguments
from configs.federated import FederatedTrainingArguments
from configs.distil import DistilArguments
from ptuning_fed.trainer.server_trainer import ServerDistilHandler
from ptuning_fed.utils import deserialize_model_trainable, serialize_model_trainable, to_student, to_teacher, \
    get_layers, uniform_choose_layers, set_layers, load_student, pickle_write

from trainer.client_trainer import FedTrainer
from fedlab.utils.aggregator import Aggregators

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments,
                               FederatedTrainingArguments, DistilArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, fed_args, distil_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, fed_args, distil_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # wandb
    os.environ["WANDB_MODE"] = "disabled"

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    if model_args.ptuning_checkpoint is not None:
        # Evaluation
        # Loading extra state dict of prefix encoder

        if 'chatglm' in model_args.model_name_or_path.lower():
            model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config,
                                              trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config,
                                                         trust_remote_code=True)
        # model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    else:
        if 'chatglm' in model_args.model_name_or_path.lower():
            model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config,
                                              trust_remote_code=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config,
                                                         trust_remote_code=True)

        # model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)

    if model_args.pre_seq_len is not None:
        # P-tuning v2
        model = model.half()
        model.transformer.prefix_encoder.float()
    else:
        # Finetune
        model = model.float()

    # Data collator
    if 'gpt2' in model_args.model_name_or_path.lower():
        tokenizer.pad_token = tokenizer.eos_token
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer,
    #     model=model,
    #     label_pad_token_id=label_pad_token_id,
    #     pad_to_multiple_of=None,
    #     padding=False
    # )

    if distil_args.do_distil:
        server_handler = ServerDistilHandler(model, model_args, training_args, data_args, distil_args)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        if distil_args.distil_resume_from_checkpoint:
            server_handler.load_distilled_checkpoint()
            logger.info("Finished saving distilled checkpoint. You can use the distilled student model for client local training")
    else:
        logger.warning("There is nothing to do! Please check parameter --do_distil for this file running.")

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
