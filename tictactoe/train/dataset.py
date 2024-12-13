from torch.utils.data import Dataset
import transformers
import torch
from typing import Dict
from loguru import logger
import json

IGNORE_INDEX = -100


def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            entry = json.loads(line.strip())
            data.append(entry)
    return data


class TokenizedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        max_seq_len: int,
    ):
        super().__init__()
        logger.info("Loading data...")
        self._data_dict_list = read_jsonl(data_path)
        logger.info("Finish loading {} samples".format(len(self._data_dict_list)))
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len

    def __len__(self):
        return len(self._data_dict_list)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        text = self._tokenizer.apply_chat_template(
            self._data_dict_list[i], tokenize=False
        )
        # text = self._data_dict_list[i]["text"] + self._tokenizer.eos_token
        text = self._tokenizer.apply_chat_template(
            self._data_dict_list[i], tokenize=False
        )
        input_ids = self._tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding=False,
            max_length=self._max_seq_len,
            return_length=False,
        )["input_ids"]
        # hacky implementation, only work for one-round conversation
        query_text = self._tokenizer.apply_chat_template(
            self._data_dict_list[i][:-1], add_generation_prompt=True, tokenize=False
        )
        query_input_ids = self._tokenizer(
            query_text, add_special_tokens=True, return_length=False
        )["input_ids"]
        assert len(input_ids) > len(query_input_ids)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(input_ids, dtype=torch.long)
        labels[: len(query_input_ids)] = IGNORE_INDEX

        return dict(
            input_ids=input_ids,
            labels=labels,
        )
