import gc
import os
import pickle
from copy import deepcopy
import torch
from torch import nn
import logging
from transformers import (
    SchedulerType,
    MODEL_MAPPING,
    OPTForCausalLM,
    GPT2LMHeadModel,
    BloomForCausalLM,
    ViTForImageClassification,
)
from offsite_tuning.models.clip_vit import CLIPViTForImageClassification
from offsite_tuning.models.eva_vit import EVAViTForImageClassification
from accelerate import Accelerator, DistributedType
import argparse

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, activation=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


def add_prologue(module, prologue):
    module.old_forward = module.forward
    module.prologue = prologue

    def new_forward(self):
        def lambda_forward(*args, **kwargs):
            self.input_args = args
            self.input_kwargs = kwargs
            if self.prologue is not None:
                x = self.prologue(args[0])
            else:
                x = args[0]
            args = (x,) + args[1:]
            return self.old_forward(*args, **kwargs)

        return lambda_forward

    module.forward = new_forward(module)
    return module


def add_epilogue(module, epilogue):
    module.old_forward = module.forward
    module.epilogue = epilogue

    def new_forward(self):
        def lambda_forward(*args, **kwargs):
            output = self.old_forward(*args, **kwargs)
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output

            if self.epilogue is not None:
                x = self.epilogue(x)

            if isinstance(output, tuple):
                output = (x,) + output[1:]
            else:
                output = x

            self.cached_output = x
            return output

        return lambda_forward

    module.forward = new_forward(module)
    return module


def uniform_choose_layers(layers: nn.ModuleList, num_student_layers=None, only_get_idx=False):
    if num_student_layers is None:
        num_student_layers = len(layers)

    student = nn.ModuleList()
    stride = (len(layers) - 1) / (num_student_layers - 1)
    origin_idx, align_idx = [], []  # align_idx records the layer index for distillation alignment
    for i in range(num_student_layers):
        idx = round(i * stride)
        if only_get_idx:
            origin_idx.append(idx)
            align_item = round((i + 1) * stride) - 1 if round((i + 1) * stride) - 1 < len(layers) else len(layers) - 1
            align_idx.append(align_item)
            continue
        logger.info(f"Adding layer {idx} to student")
        student.append(layers[idx])

    if only_get_idx:
        return origin_idx, align_idx
    return student


@torch.no_grad()
def magnitude_prune(model, ratio):
    for param in model.parameters():
        if param.dim() == 1:
            continue
        num_prune = int(param.numel() * ratio)
        threshold = param.abs().view(-1).kthvalue(num_prune).values.item()
        mask = (param.abs() >= threshold).to(param.dtype)
        param.mul_(mask)


@torch.no_grad()
def quantize(model, bits):
    for param in model.parameters():
        if param.dim() == 1:
            continue
        min, max = param.min(), param.max()
        zp = (max + min) / 2
        scale = (max - min) / (2 ** bits - 1)
        param.sub_(zp).div_(scale).round_().mul_(scale).add_(zp)


def get_layers(model):
    if isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, GPT2LMHeadModel):
        layers = model.transformer.h
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif isinstance(model, ViTForImageClassification):
        layers = model.vit.encoder.layer
    elif isinstance(model, CLIPViTForImageClassification):
        layers = model.vit.encoder.layers
    elif isinstance(model, EVAViTForImageClassification):
        layers = model.blocks
    else:
        raise NotImplementedError
    return layers


def set_layers(model, layers):
    if isinstance(model, OPTForCausalLM):
        model.model.decoder.layers = layers
    elif isinstance(model, GPT2LMHeadModel):
        model.transformer.h = layers
    elif isinstance(model, BloomForCausalLM):
        model.transformer.h = layers
    elif isinstance(model, ViTForImageClassification):
        model.vit.encoder.layer = layers
    elif isinstance(model, CLIPViTForImageClassification):
        model.vit.encoder.layers = layers
    elif isinstance(model, EVAViTForImageClassification):
        model.blocks = layers
    else:
        raise NotImplementedError


def to_teacher(model, student_l_pad, student_r_pad):
    l = student_l_pad
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - student_r_pad
        model.model.decoder.layers = model.model.decoder.layers[
            :l] + model.teacher + model.model.decoder.layers[r:]
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            model.teacher + model.transformer.h[r:]
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
            model.teacher + model.transformer.h[r:]
    elif isinstance(model, ViTForImageClassification):
        r = len(model.vit.encoder.layer) - student_r_pad
        model.vit.encoder.layer = model.vit.encoder.layer[:l] + \
            model.teacher + model.vit.encoder.layer[r:]
    elif isinstance(model, CLIPViTForImageClassification):
        r = len(model.vit.encoder.layers) - student_r_pad
        model.vit.encoder.layers = model.vit.encoder.layers[:l] + \
            model.teacher + model.vit.encoder.layers[r:]
    elif isinstance(model, EVAViTForImageClassification):
        r = len(model.blocks) - student_r_pad
        model.blocks = model.blocks[:l] + \
            model.teacher + model.blocks[r:]
    else:
        raise NotImplementedError


def to_student(model, student_l_pad, student_r_pad):
    l = student_l_pad
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - student_r_pad
        model.model.decoder.layers = model.model.decoder.layers[
            :l] + model.student + model.model.decoder.layers[r:]
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
                              model.student + model.transformer.h[r:]
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - student_r_pad
        model.transformer.h = model.transformer.h[:l] + \
                              model.student + model.transformer.h[r:]
    elif isinstance(model, ViTForImageClassification):
        r = len(model.vit.encoder.layer) - student_r_pad
        model.vit.encoder.layer = model.vit.encoder.layer[:l] + \
                                  model.student + model.vit.encoder.layer[r:]
    elif isinstance(model, CLIPViTForImageClassification):
        r = len(model.vit.encoder.layers) - student_r_pad
        model.vit.encoder.layers = model.vit.encoder.layers[:l] + \
                                   model.student + model.vit.encoder.layers[r:]
    elif isinstance(model, EVAViTForImageClassification):
        r = len(model.blocks) - student_r_pad
        model.blocks = model.blocks[:l] + \
                       model.student + model.blocks[r:]
    else:
        raise NotImplementedError

# for sLLM model load
def load_student(model, student_state_dict, args):
    l = args.student_l_pad

    student_layers_len = len(
        set([k.split('.')[0] for k in student_state_dict.keys()]))
    logger.info(f"Loading student module from with {student_layers_len} layers.")
    if isinstance(model, OPTForCausalLM):
        r = len(model.model.decoder.layers) - args.student_r_pad
        student_layers = model.model.decoder.layers[l:l + student_layers_len]
        student_layers.load_state_dict(student_state_dict)
        model.model.decoder.layers = model.model.decoder.layers[:l] + \
                                     student_layers + model.model.decoder.layers[r:]
    elif isinstance(model, GPT2LMHeadModel):
        r = len(model.transformer.h) - args.student_r_pad
        student_layers = model.transformer.h[l:l + student_layers_len]
        student_layers.load_state_dict(student_state_dict)
        model.transformer.h = model.transformer.h[:l] + \
                              student_layers + model.transformer.h[r:]
    elif isinstance(model, BloomForCausalLM):
        r = len(model.transformer.h) - args.student_r_pad
        student_layers = model.transformer.h[l:l + student_layers_len]
        student_layers.load_state_dict(student_state_dict)
        model.transformer.h = model.transformer.h[:l] + \
                              student_layers + model.transformer.h[r:]
    else:
        raise NotImplementedError

    return model


def setup_trainable_classification_head(model):
    # Setup trainable classification heads
    if isinstance(model, ViTForImageClassification):
        for param in model.classifier.parameters():
            param.requires_grad = True
            param.data = param.data.float()
    elif isinstance(model, CLIPViTForImageClassification):
        for param in model.classifier.parameters():
            param.requires_grad = True
            param.data = param.data.float()
    elif isinstance(model, EVAViTForImageClassification):
        for param in model.classifier.parameters():
            param.requires_grad = True
            param.data = param.data.float()
    else:
        raise NotImplementedError


def save_state_dict(state_dict, output_dir, filename):
    for k in state_dict:
        state_dict[k] = state_dict[k].to(torch.float16).cpu()
    torch.save(state_dict, os.path.join(output_dir, filename))


def pickle_read(path, read_format="rb"):
    with open(path, read_format) as file:
        obj = pickle.load(file)
    return obj


def pickle_write(obj, path, write_format="wb"):
    with open(path, write_format) as file:
        pickle.dump(obj, file)


def get_block_size(block_size, tokenizer):
    if block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(block_size, tokenizer.model_max_length)
    return block_size


def serialize_model_trainable(model: torch.nn.Module) -> torch.Tensor:
    """Unfold model parameters

    Unfold every layer of model, concate all of tensors into one.
    Return a `torch.Tensor` with shape (size, ).

    Args:
        model (torch.nn.Module): model to serialize.
    """
    parameters = [param.data.reshape(-1) for param in model.parameters() if param.requires_grad]
    m_parameters = torch.cat(parameters).cpu()
    return m_parameters


def deserialize_model_trainable(model: torch.nn.Module,
                                serialized_parameters: torch.Tensor,
                                mode="copy"):
    """Assigns serialized parameters to model.parameters.
    This is done by iterating through ``model.parameters()`` and assigning the relevant params in ``grad_update``.
    NOTE: this function manipulates ``model.parameters``.

    Args:
        model (torch.nn.Module): model to deserialize.
        serialized_parameters (torch.Tensor): serialized model parameters.
        mode (str): deserialize mode. "copy" or "add".
    """
    current_index = 0  # keep track of where to read from grad_update
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        numel = parameter.data.numel()
        size = parameter.data.size()
        if mode == "copy":
            parameter.data.copy_(
                serialized_parameters[current_index:current_index +
                                                    numel].view(size))
        elif mode == "add":
            parameter.data.add_(
                serialized_parameters[current_index:current_index +
                                                    numel].view(size))
        else:
            raise ValueError(
                "Invalid deserialize mode {}, require \"copy\" or \"add\" "
                    .format(mode))
        current_index += numel



import torch
import math
from torch import nn
import torch.nn.functional as F

# patient loss code, need teacher scores
def distillation_loss(y, labels, teacher_scores, T, alpha, reduction_kd='mean', reduction_nll='mean'):
    #if teacher_scores is not None and y.dtype != teacher_scores.dtype:
    #    teacher_scores = teacher_scores.half()

    if teacher_scores is not None:
        d_loss = nn.KLDivLoss(reduction=reduction_kd)(F.log_softmax(y / T, dim=1),
                                                      F.softmax(teacher_scores / T, dim=1)) * T * T
    else:
        assert alpha == 0, 'alpha cannot be {} when teacher scores are not provided'.format(alpha)
        d_loss = 0.0
    nll_loss = F.cross_entropy(y, labels, reduction=reduction_nll)
    # print(d_loss.shape, d_loss)
    # print('\n', nll_loss.shape, nll_loss)
    tol_loss = alpha * d_loss + (1.0 - alpha) * nll_loss
    # print('in func:', d_loss.item(), nll_loss.item(), alpha, tol_loss.item())
    return tol_loss, d_loss, nll_loss


def patience_loss(teacher_patience, student_patience, normalized_patience=False):
    # n_batch = teacher_patience.shape[0]
    if normalized_patience:
        teacher_patience = F.normalize(teacher_patience, p=2, dim=2)
        student_patience = F.normalize(student_patience, p=2, dim=2)
    return F.mse_loss(teacher_patience.float(), student_patience.float()).half()

    # diff = (teacher_patience - student_patience).pow(2).sum()
    # const = math.sqrt(teacher_patience.numel())
    # return diff / const / const