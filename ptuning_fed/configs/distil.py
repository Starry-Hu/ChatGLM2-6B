"""federated configs for FNLP"""

from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DistilArguments:
    do_distil: bool = field(default=False, metadata={"help": "Whether to run distillation."})

    train_tokenized_dataset: Optional[str] = field(
        default=None, metadata={"help": "The input tokenized training data file used for distillation"}
    )

    val_tokenized_dataset: Optional[str] = field(
        default=None, metadata={"help": "The input tokenized validation data file used for distillation"}
    )

    # group datasets
    block_size: int = field(default=None, metadata={"help": "Number of input block in model."})

    num_student_layers: int = field(default=None, metadata={"help": "Number of layers in the student model."})

    student_layer_selection_strategy: str = field(default='uniform', metadata={"help": "Layer selection strategy"})

    load_student: Optional[str] = field(default=None, metadata={"help": "The path to the student model"})

    student_l_pad: int = field(default=0, metadata={"help": "The padding layer number from left"})

    student_r_pad: int = field(default=0, metadata={"help": "The padding layer number from right"})

    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA."})

    lora_rank: int = field(default=4, metadata={"help": "Rank of the LoRA matrix"})

    lora_alpha: float = field(default=32, metadata={"help": "Alpha of the LoRA matrix"})

    lm_weight: float = field(default=1.0, metadata={"help": "Weight of the LM loss."})

    kd_weight: float = field(default=0.0, metadata={"help": "Weight of the knowledge distillation loss."})

    distil_resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your distilled model."},
    )

    def __post_init__(self):
        if self.do_distil and self.train_tokenized_dataset is None and self.val_tokenized_dataset is None:
            raise ValueError("Need a training and a validation file simultaneously when distilling.")