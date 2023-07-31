import copy
import os
import gc
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain

import torch
from datasets import load_from_disk
from fedlab.core.server.handler import ServerHandler
from fedlab.core.standalone import ServerHandler
from transformers import Trainer, AutoConfig, DefaultDataCollator
from transformers.data.data_collator import DataCollatorMixin
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, AutoModelForCausalLM

from ptuning_fed.param_efficient import use_lora
from ptuning_fed.utils import get_layers, uniform_choose_layers, to_student, distillation_loss, add_prologue, \
    add_epilogue, get_block_size
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ServerDistilHandler(ServerHandler, ABC):
    def __init__(self, model, model_args, training_args, data_args, distil_args, cuda=False):
        super().__init__(model, cuda)
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args
        self.distil_args = distil_args

    def load_distil_data(self, tokenizer, lm_type='clm'):
        # Load tokenized dataset
        if self.distil_args.train_tokenized_dataset and self.distil_args.val_tokenized_dataset:
            tokenized_datasets = load_from_disk(self.distil_args.train_tokenized_dataset)
            val_dataset = load_from_disk(self.distil_args.val_tokenized_dataset)
            if 'validation' in val_dataset:
                tokenized_datasets["validation"] = val_dataset['validation']
            else:
                tokenized_datasets["validation"] = val_dataset['train']

        # group dataset
        block_size = get_block_size(self.distil_args.block_size, tokenizer)

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + block_size]
                    for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            if lm_type == 'clm':
                result["labels"] = result["input_ids"].copy()
            return result

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

        return lm_datasets

    def setup_teacher_student(self, model, args):
        for param in model.parameters():
            param.requires_grad = False
        model_type = type(model).__name__
        print(f"model is an instance of {model_type}")
        layers = get_layers(model)

        l, r = args.student_l_pad, len(layers) - args.student_r_pad
        if args.load_student:  # continue to train student
            student_state_dict = torch.load(os.path.join(
                args.load_student, 'student.pt'), map_location='cpu')
            student_layers_len = len(
                set([k.split('.')[0] for k in student_state_dict.keys()]))
            logger.info(
                f"Loading student module from {args.load_student} with {student_layers_len} layers.")
            student = deepcopy(layers[:student_layers_len])  # 与load_student的区别，deepcopy
            student.load_state_dict(student_state_dict)
        else:
            student = deepcopy(layers[l:r])  # 小模型

        if args.student_layer_selection_strategy == 'uniform':
            student = uniform_choose_layers(student, args.num_student_layers)
        else:
            raise NotImplementedError

        for param in student.parameters():
            param.data = param.data.float()
            param.requires_grad = True

        model.student = student
        model.teacher = layers[l:r].half()

        add_prologue(model.student[0], None)
        add_epilogue(model.student[-1], None)
        model.student_l = model.student[0]
        model.student_r = model.student[-1]

        num_student_layers = len(model.student)
        logger.info(f"Number of student layers: {num_student_layers}")

        gc.collect()
        torch.cuda.empty_cache()

    def distillation(self, tokenizer):
        lm_dataset = self.load_distil_data(tokenizer)

        logger.info(
            f"======== Distillation with student_l_pad {self.distil_args.student_l_pad}, "
            f"student_r_pad {self.distil_args.student_r_pad} ==========")
        self.setup_teacher_student(self.model, self.distil_args)

        to_student(self.model, self.distil_args.student_l_pad,
                   self.distil_args.student_r_pad)  # bind student with top and bottom layers to train

        trainer = DistillationTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=lm_dataset['train'],
            eval_dataset=lm_dataset['validation'],
            data_collator=DefaultDataCollator(),
            # compute_metrics=self.compute_metrics if self.training_args.predict_with_generate else None,
            distil_args=self.distil_args,
        )

        checkpoint = None
        if self.distil_args.distil_resume_from_checkpoint is not None:
            checkpoint = self.distil_args.distil_resume_from_checkpoint

        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(self.training_args.output_dir)  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            self.data_args.max_train_samples if self.data_args.max_train_samples is not None else len(lm_dataset['train'])
        )
        metrics["train_samples"] = min(max_train_samples, len(lm_dataset['train']))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if self.training_args.output_dir:
            # default path for distilled student
            output_path = os.path.join(self.training_args.output_dir, 'distilled_student')
            trainer.save_student(output_path, 'student.pt')
            logger.info(f"Saving distilled model in {os.path.join(output_path,'student.pt')}")

        model_type = type(self.model).__name__
        print(f"self.model is an instance of {model_type}, idx: {id(self.model)}")

        self._model = trainer.model

    def load_distilled_checkpoint(self):
        self.setup_teacher_student(self.model, self.distil_args)

        to_student(self.model, self.distil_args.student_l_pad,
                   self.distil_args.student_r_pad)  # bind student with top and bottom layers to train

        logger.info(f"Loading model from checkpoint {self.distil_args.distil_resume_from_checkpoint}")
        model_dict = torch.load(os.path.join(self.distil_args.distil_resume_from_checkpoint, 'pytorch_model.bin'))
        self.model.load_state_dict(model_dict)

        print(self.model)
        # save the distilled student model
        state_dict = self.model.student.state_dict()
        for k in state_dict:
            state_dict[k] = state_dict[k].to(torch.float16).cpu()
        torch.save(state_dict, os.path.join(self.distil_args.distil_resume_from_checkpoint, 'student.pt'))
        logger.info(f"Finish loading student module in {os.path.join(self.distil_args.distil_resume_from_checkpoint, 'student.pt')}")
        return self.model

    def set_plugin_params(self):
        logger.warning(f"self.model idx in set_plugin_params(): {id(self.model)}")
        if self.distil_args.use_lora:
            logger.info("====Setup Lora parameters===")
            # student lora
            use_lora(self.model.student, self.distil_args.lora_rank, self.distil_args.lora_alpha)

            stride = (len(self.model.teacher) - 1) / (len(self.model.student) - 1)
            assigned_layers_ids = [round(i * stride) for i in range(len(self.model.student))]
            # teacher lora
            use_lora(self.model.teacher, self.distil_args.lora_rank, self.distil_args.lora_alpha,
                     assigned_layers_ids=assigned_layers_ids)
        else:
            raise NotImplementedError


class DistillationTrainer(Trainer):
    def __init__(self, model=None,
                 args=None,
                 data_collator=None,
                 train_dataset=None,
                 eval_dataset=None,
                 tokenizer=None,
                 model_init=None,
                 compute_metrics=None,
                 callbacks=None,
                 optimizers=(None, None),
                 preprocess_logits_for_metrics=None,
                 distil_args=None):
        super(DistillationTrainer, self).__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer,
                                                  model_init, compute_metrics,
                                                  callbacks, optimizers, preprocess_logits_for_metrics)
        self.distil_args = distil_args

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs, output_hidden_states=True)
        # student_hidden_state = outputs.hidden_states[0]
        student_hidden_state = outputs.hidden_states  # outputs[2]
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # kd Loss
        kd_loss = self.get_kd_loss(model.module, distil_args=self.distil_args,
                                   student_hidden_states=student_hidden_state)
        # loss_dl, kd_loss, ce_loss = distillation_loss(outputs.logits, )

        loss = self.distil_args.lm_weight * loss + self.distil_args.kd_weight * kd_loss

        return (loss, outputs) if return_outputs else loss

    def get_kd_loss(self, model, distil_args, student_hidden_states=None):
        kwargs = model.student_l.input_kwargs
        args = model.student_l.input_args
        # kwargs = model.student[0].input_kwargs
        # args = model.student[0].input_args
        output_teacher = args[0].to(torch.float16)

        # calculate aligned uniformed layer
        _, align_layer_idx = uniform_choose_layers(model.teacher, len(model.student), only_get_idx=True)
        # print(f"origin_layer_idx: {origin_layer_idx}")
        teacher_hidden_states = ()

        args = list(args[1:])
        for i, arg in enumerate(args):
            if torch.is_tensor(arg) and arg.dtype == torch.float32:
                args[i] = arg.to(torch.float16)
        args = tuple(args)

        for k, v in kwargs.items():
            if torch.is_tensor(v) and v.dtype == torch.float32:
                kwargs[k] = v.to(torch.float16)

        with torch.no_grad():
            model.teacher.eval()
            for i, teacher_layer in enumerate(model.teacher):
                output_teacher = teacher_layer(output_teacher, *args, **kwargs)
                if isinstance(output_teacher, tuple):
                    output_teacher = output_teacher[0]
                    if i in align_layer_idx and student_hidden_states is not None:
                        teacher_hidden_states = teacher_hidden_states + (output_teacher,)

        output_student = model.student[-1].cached_output.float()
        output_teacher = output_teacher.float()

        std = output_teacher.pow(2).mean().sqrt()
        kd_loss = (output_teacher - output_student).div(std).pow(2).mean()

        # logger.critical(f"teacher_hidden_states: {len(teacher_hidden_states)}, {teacher_hidden_states[-1].shape}")
        # logger.critical(
        #     f"student_hidden_states: origin: {len(student_hidden_states)}, now: {len(student_hidden_states[distil_args.student_l_pad: -distil_args.student_r_pad-1])}, {student_hidden_states[-1].shape}")
        # logger.info(f"len student hidden state: {len(student_hidden_states[distil_args.student_l_pad: -distil_args.student_r_pad-1])}; pad: {distil_args.student_l_pad}, {-distil_args.student_r_pad-1}")
        hidden_kd_loss = 0
        for t_hidden, s_hidden in zip(teacher_hidden_states,
                                      student_hidden_states[distil_args.student_l_pad: -distil_args.student_r_pad-1]):  # remove the repeated last layer output
            hidden_kd_loss += (t_hidden - s_hidden).div(std).pow(2).mean()

        return kd_loss + hidden_kd_loss

    def save_student(self, output_dir, filename):
        # self._save(output_file, state_dict)

        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        state_dict = self.model.student.state_dict()
        for k in state_dict:
            state_dict[k] = state_dict[k].to(torch.float16).cpu()
        torch.save(state_dict, os.path.join(output_dir, filename))
