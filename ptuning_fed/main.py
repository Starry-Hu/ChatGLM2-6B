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
    set_seed,
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

    # Load dataset
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

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

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    # Data collator
    if 'gpt2' in model_args.model_name_or_path.lower():
        tokenizer.pad_token = tokenizer.eos_token
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )

    # copy from chatglm tokenizer py
    def build_prompt(query, history=None):
        if history is None:
            history = []
        prompt = ""
        for i, (old_query, response) in enumerate(history):
            prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
        prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
        return prompt

    def preprocess_function_eval(examples):
        inputs, targets = [], []
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query = examples[prompt_column][i]
                history = examples[history_column][i] if history_column is not None else None
                prompt = tokenizer.build_prompt(query, history) if 'chatglm' in model_args.model_name_or_path.lower() \
                    else build_prompt(query, history)  # fix for non-chatglm
                inputs.append(prompt)
                targets.append(examples[response_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, truncation=True, padding=True)
        labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

        if data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def preprocess_function_train(examples):
        max_seq_length = data_args.max_source_length + data_args.max_target_length + 1

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query, answer = examples[prompt_column][i], examples[response_column][i]

                history = examples[history_column][i] if history_column is not None else None
                prompt = tokenizer.build_prompt(query, history) if 'chatglm' in model_args.model_name_or_path.lower() \
                    else build_prompt(query, history)  # fix for non-chatglm

                prompt = prefix + prompt
                a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
                                         max_length=data_args.max_source_length)
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False, truncation=True,
                                         max_length=data_args.max_target_length)

                context_length = len(a_ids)
                input_ids = a_ids + b_ids + [tokenizer.eos_token_id]
                labels = [tokenizer.pad_token_id] * context_length + b_ids + [tokenizer.eos_token_id]

                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
                if data_args.ignore_pad_token_for_loss:
                    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)

        return model_inputs

    def print_dataset_example(example):  # bug
        print("input_ids", example["input_ids"])
        print("inputs", tokenizer.decode(example["input_ids"]))
        print("label_ids", example["labels"])
        print("pad_token_id", tokenizer.pad_token_id)
        print("labels", tokenizer.decode(example["labels"]), skip_special_tokens=True)

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        # print_dataset_example(train_dataset[0])

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        # print_dataset_example(eval_dataset[0])

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        # print_dataset_example(predict_dataset[0])


    if distil_args.do_distil:
        server_handler = ServerDistilHandler(model, model_args, training_args, data_args, distil_args)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        server_handler.distillation(train_dataset, eval_dataset, tokenizer, data_collator)
        logger.info("Finished distillation. You can use the distilled student model for client local training")
        return  # only for distillation

    # federated
    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    client_datasets = {0: {
        'train': train_dataset if training_args.do_train else None,
        'validation': eval_dataset if training_args.do_eval else None,
        'test': predict_dataset if training_args.do_predict else None
    }, 1: {
        'train': train_dataset if training_args.do_train else None,
        'validation': eval_dataset if training_args.do_eval else None,
        'test': predict_dataset if training_args.do_predict else None
    }}

    # generate sLLM
    if distil_args.num_student_layers is not None:
        layers = get_layers(model)
        layers = uniform_choose_layers(layers, distil_args.num_student_layers)  # student
        set_layers(model, layers)
    if distil_args.load_student:
        student_state_dict = torch.load(distil_args.load_student, map_location='cpu')
        model = load_student(model, student_state_dict, distil_args)
        model.trainable_module = get_layers(model)
        use_lora(model.trainable_module, distil_args.lora_rank, distil_args.lora_alpha)

    fed_trainer = FedTrainer(model, model_args, training_args, data_args, tokenizer, client_datasets,
                             len(client_datasets))
    client_ids = [i for i in range(len(client_datasets))]
    aggregator = Aggregators()
    for rd in range(fed_args.rounds):
        cur_params = serialize_model_trainable(model)
        logger.warning(f"cur_params: {cur_params}")
        trainable_parameter_list = fed_trainer.local_process(client_ids, [cur_params])
        agg_params = aggregator.fedavg_aggregate(trainable_parameter_list)
        deserialize_model_trainable(model, agg_params)
        # deserialize_model_trainable(server_handler.model.teacher, agg_params)  # plugin

    # save the latest trainable model params.
    state_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in model.named_parameters():
        if v.requires_grad:
            filtered_state_dict[k] = state_dict[k]
    output_dir = os.path.join(training_args.output_dir, 'fed_sLLM')
    model.save_pretrained(output_dir, state_dict=filtered_state_dict)

    pickle_write(agg_params, os.path.join(output_dir, 'fed_sLLM.pkl'))
    logger.info("Finished saving federated student lora parameters")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
