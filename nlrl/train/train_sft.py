#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import os
import torch
import torch.distributed as dist
import transformers
from transformers import Trainer, TrainingArguments

from dataset import TokenizedDataset
from model_config import ModelConfig

IGNORE_INDEX = -100


@dataclass
class SftScriptArguments:
    dataset_name: str = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "the dataset name"},
    )
    dataset_text_field: str = field(
        default="text", metadata={"help": "the text field of the dataset"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "The maximum sequence length for SFT Trainer"}
    )
    packing: bool = field(
        default=False,
        metadata={"help": "Whether to apply data packing or not during training"},
    )
    config: str = field(
        default=None, metadata={"help": "Path to the optional config file"}
    )
    gradient_checkpointing_use_reentrant: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply `use_reentrant` for gradient_checkpointing"
        },
    )
    train_dataset_path: str = field(
        default=None, metadata={"help": "Path to the optional config file"}
    )
    eval_dataset_path: str | None = field(
        default=None, metadata={"help": "Path to the optional config file"}
    )
    checkpoint_path: str | None = field(
        default=None, metadata={"help": "Path to the optional config file"}
    )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def train():
    parser = transformers.HfArgumentParser(
        (SftScriptArguments, TrainingArguments, ModelConfig)
    )
    args, training_args, model_config = parser.parse_args_into_dataclasses()

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, **model_kwargs
    )

    model = model.to(torch.bfloat16)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        model_max_length=args.max_seq_length,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    train_dataset = TokenizedDataset(
        args.train_dataset_path, tokenizer, args.max_seq_length
    )
    eval_dataset = None
    if not args.eval_dataset_path == "None":
        eval_dataset = TokenizedDataset(
            args.eval_dataset_path, tokenizer, args.max_seq_length
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    # trainer.load_optimizer(model_config.model_name_or_path)
    if args.checkpoint_path == "None":
        args.checkpoint_path = None
    trainer.train(resume_from_checkpoint=args.checkpoint_path)
    # # model.config.to_json_file(os.path.join(training_args.output_dir, "config.json"))
    # trainer.save_model(output_dir=training_args.output_dir)

    # local_rank = int(os.environ['LOCAL_RANK'])
    # if local_rank == 0:
    #     trainer.save_optimizer(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
