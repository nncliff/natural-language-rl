from nlrl.utils import read_jsonl, write_jsonl
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers import pipeline
import torch
import json
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path

boards = read_jsonl(Path(__file__).parent / "../../../board_value_prompt.jsonl")
value_dict = {}
df = pd.DataFrame.from_dict(boards)


# Constants
BASE_MODEL = "/home/zhiyu/code/llm/results"
OUTPUT_DIR = "./results/"
SAVE_DIR = "./trained_model/"
sft_save_dir = SAVE_DIR + "sft"
batch_size = 64


prompt_dataset_batches = [df[i : i + batch_size] for i in range(0, len(df), batch_size)]


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
attn_implementation = "flash_attention_2"
torch_dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    attn_implementation=attn_implementation,
    torch_dtype=torch_dtype,
    device_map="auto",
)
correct = 0
full_idx = 0
generator = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, batch_size=batch_size
)


for batch in tqdm(prompt_dataset_batches):
    inputs = tokenizer(
        batch["prompt"].to_list(), return_tensors="pt", padding=True, truncation=True
    ).to(model.device)
    outputs = model.generate(
        inputs["input_ids"], num_return_sequences=1, do_sample=False, max_length=1024
    )
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    items = []
    for idx, response in enumerate(responses):
        items.append(
            {
                "state": batch.iloc[idx]["state"],
                "prompt": batch.iloc[idx]["prompt"],
                "response": response.split(batch.iloc[idx]["prompt"])[-1],
            }
        )
    with open(f"board_value.jsonl", "a") as f:
        for item in items:
            try:
                f.write(
                    json.dumps(
                        {
                            "state": item["state"],
                            "prompt": item["prompt"],
                            "response": item["response"],
                        }
                    )
                    + "\n"
                )
            except:
                print(item)
