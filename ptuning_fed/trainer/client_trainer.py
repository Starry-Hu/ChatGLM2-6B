# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
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
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""
import gc
import os
from abc import ABC
from typing import Optional, List

from ptuning_fed.utils import serialize_model_trainable, deserialize_model_trainable
from transformers import Trainer, Seq2SeqTrainer

import torch
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.utils import logging
from transformers import DataCollatorForSeq2Seq
from fedlab.core.client import ClientTrainer
from fedlab.core.client.trainer import SerialClientTrainer
import numpy as np
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json


logger = logging.get_logger(__name__)

WEIGHTS_NAME = "pytorch_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"


class FedTrainer(SerialClientTrainer, ABC):
    def __init__(self, model, model_args, training_args, data_args, tokenizer, dataset,
                 num_clients, cuda=False, device=None, personal=False):
        super().__init__(model, num_clients, cuda, device, personal)
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.dataset = dataset  # id->{'train': List, 'valid': List, 'test': List}

        self.before_training()

    def before_training(self):
        assert len(self.dataset) == self.num_clients

        # Data collator
        if 'gpt2' in self.model_args.model_name_or_path.lower():
            self.tokenizer.pad_token = self.tokenizer.eos_token

        label_pad_token_id = -100 if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=None,
            padding=False
        )

    @property
    def uplink_package(self):
        return self.param_list

    @property
    def model_parameters(self) -> torch.Tensor:
        """Return serialized model parameters."""
        return serialize_model_trainable(self._model)

    def local_process(self, id_list: List, payload: List):
        """local process for Federated Learning"""
        model_parameters = payload[0]
        self.param_list = []
        for idx in id_list:
            logger.critical(f"===== training client {idx}=====")
            model_params = self.train(
                idx=idx,
                model_parameters=model_parameters,
            )
            del self.hf_trainer
            gc.collect()
            torch.cuda.empty_cache()

            self.param_list.append(model_params)
            logger.critical(f"===== Finish training client {idx}=====")
        return self.param_list

    def train(self, model_parameters, idx):
        # local data for client idx
        train_dataset = self.dataset[idx]['train']
        eval_dataset = self.dataset[idx]['validation'] if 'validation' in self.dataset[idx] else None

        # load parameters
        origin_trainable = serialize_model_trainable(self.model)
        deserialize_model_trainable(self.model, model_parameters)
        new_trainable = serialize_model_trainable(self.model)
        logger.critical(f"check content change: {torch.equal(origin_trainable, new_trainable)}")

        # Initialize our Trainer
        self.hf_trainer = Seq2SeqTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset if self.training_args.do_train else None,
            eval_dataset=eval_dataset if self.training_args.do_eval else None,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics if self.training_args.predict_with_generate else None,
            # save_changed=self.model_args.pre_seq_len is not None
        )

        if self.training_args.do_train:
            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint.format(idx)  # for each client
            # elif last_checkpoint is not None:
            #     checkpoint = last_checkpoint
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()

            logger.warning(f"device of hf_trainer model: {self.hf_trainer.model.device}")
            train_result = self.hf_trainer.train(resume_from_checkpoint=checkpoint)
            # trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics
            max_train_samples = (
                self.data_args.max_train_samples if self.data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            self.hf_trainer.log_metrics("train", metrics)
            self.hf_trainer.save_metrics("train", metrics)
            self.hf_trainer.save_state()

        return serialize_model_trainable(self.hf_trainer.model)

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
            result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    def evaluate(self):
        results = {}
        max_seq_length = self.data_args.max_source_length + self.data_args.max_target_length + 1
        if self.training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = self.hf_trainer.evaluate(metric_key_prefix="eval", do_sample=True, top_p=0.7,
                                               max_length=max_seq_length,
                                               temperature=0.95)
            max_eval_samples = self.data_args.max_eval_samples if self.data_args.max_eval_samples is not None else len(
                self.eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(self.eval_dataset))

            self.hf_trainer.log_metrics("eval", metrics)
            self.hf_trainer.save_metrics("eval", metrics)

        if self.training_args.do_predict:
            logger.info("*** Predict ***")
            predict_results = self.hf_trainer.predict(self.predict_dataset, metric_key_prefix="predict",
                                                      max_length=max_seq_length,
                                                      do_sample=True, top_p=0.7, temperature=0.95)
            metrics = predict_results.metrics
            max_predict_samples = (
                self.data_args.max_predict_samples if self.data_args.max_predict_samples is not None else len(
                    self.predict_dataset)
            )
            metrics["predict_samples"] = min(max_predict_samples, len(self.predict_dataset))

            self.hf_trainer.log_metrics("predict", metrics)
            self.hf_trainer.save_metrics("predict", metrics)

            if self.hf_trainer.is_world_process_zero():
                if self.training_args.predict_with_generate:
                    predictions = self.tokenizer.batch_decode(
                        predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    predictions = [pred.strip() for pred in predictions]
                    labels = self.tokenizer.batch_decode(
                        predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    labels = [label.strip() for label in labels]
                    output_prediction_file = os.path.join(self.training_args.output_dir, "generated_predictions.txt")
                    with open(output_prediction_file, "w", encoding="utf-8") as writer:
                        for p, l in zip(predictions, labels):
                            res = json.dumps({"labels": l, "predict": p}, ensure_ascii=False)
                            writer.write(f"{res}\n")
        return results
