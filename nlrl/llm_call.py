import signal
from typing import Optional
from openai import OpenAI
import openai
import requests
import os
import time
from typing import Optional, Any
from nlrl.config import LLMSamplingParams
import functools

from transformers import AutoTokenizer
import gc
import torch

from vllm import LLM, SamplingParams

import subprocess
import requests
import json
import signal


def llama3_instruct_format(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

class openai_model:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        sample_config: LLMSamplingParams = None,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.sample_config = sample_config
    
    def generate(self, message, return_text=False):
        # OPENAI API cannot batch generate
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message,
                    temperature=self.sample_config.temperature,
                    max_tokens=self.sample_config.max_tokens,
                    top_p=self.sample_config.top_p,
                    response_format={"type": "json_object"}
                )
            except openai.OpenAIError as e:
                print(e)
                time.sleep(1)
                continue
            break
        text = response.choices[0].message.content
        if return_text:
            return text
        else:
            message.append({"role": "assistant", "content": text})
        return message

class vllm_model:
    def __init__(
        self,
        model_path: str,
        sample_config: LLMSamplingParams,
        prompt_format: Any = llama3_instruct_format,
        tensor_parallel_size: int = None,
    ):
        self.sampling_params = SamplingParams(
            n=sample_config.n,
            temperature=sample_config.temperature,
            top_p=sample_config.top_p,
            top_k=sample_config.top_k,
            max_tokens=sample_config.max_tokens,
            prompt_logprobs=0,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size or torch.cuda.device_count(),
            disable_custom_all_reduce=True,
        )
        self.prompt_format = functools.partial(prompt_format, tokenizer=self.tokenizer)

    def generate(self, messages):
        # Batched generation
        prompts = [self.prompt_format(messages=d) for d in messages]
        outputs = self.llm.generate(prompts, self.sampling_params)
        for output, message in zip(outputs, messages):
            generated_text = output.outputs[0].text
            message.append({"role": "assistant", "content": generated_text})
        return messages, outputs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class RemoteVLLMModel:
    def __init__(
        self,
        model_path: str,
        sample_config: LLMSamplingParams,
        prompt_format: Any = llama3_instruct_format,
        tensor_parallel_size: int = None,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        self.prompt_format = prompt_format
        self.sample_config = sample_config
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size or torch.cuda.device_count()
        self.port = port
        self.host = host
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.prompt_format = functools.partial(prompt_format, tokenizer=self.tokenizer)

        self._build_remote_server()

    def _build_remote_server(self):
        args = [
            "python",
            "nlrl/vllm_api_server.py",
            "--model",
            self.model_path,
            "--tensor-parallel-size",
            "4",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        print("Start vLLM server by command: {}".format(args))
        self._vllm_process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        print("Try connecting to vLLM Server ...")
        # try connect 300 seconds
        max_retry = 60
        i = 0
        while True:
            if self._test_request() is None:
                print(f"Failed will sleep for 5 seconds: {i+1}/{max_retry}", end="\r")
                i += 1
            else:
                print("Connected!!!")
                break
            if i >= max_retry:
                raise RuntimeError("vLLM Server Launch Failed")
            time.sleep(5)

    def _test_request(self):
        api_url = f"http://{self.host}:{self.port}/generate"
        headers = {"Content-Type": "application/json"}
        payload = {"prompt": "Tell me why"}
        try:
            response = requests.post(api_url, headers=headers, json=payload)
            return response.json()
        except requests.exceptions.ConnectionError as e:
            return None
        # return response.json()

    def destroy(self):
        if self._vllm_process is None:
            print("No VLLM process to terminate.")
            return

        self._vllm_process.send_signal(signal.SIGINT)
        self._vllm_process.wait()
        self._vllm_process = None

    def generate(self, messages):
        prompts = [self.prompt_format(messages=d) for d in messages]

        api_url = f"http://{self.host}:{self.port}/generate"
        headers = {"Content-Type": "application/json"}
        params = {
            "prompt": prompts,
            "n": self.sample_config.n,
            "temperature": self.sample_config.temperature,
            "top_k": self.sample_config.top_k,
            "top_p": self.sample_config.top_p,
            "max_tokens": self.sample_config.max_tokens,
        }
        # outputs = requests.post(api_url, headers=headers, json=params).json()["text"]
        # for generated_text, message in zip(outputs, messages):
        #     message.append({"role": "assistant", "content": generated_text})

        response = requests.post(api_url, headers=headers, json=params, stream=True)
        generated_texts = []

        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    # Each chunk is a complete JSON object, decode it
                    decoded_chunk = json.loads(chunk.decode("utf-8"))
                    generated_texts.extend(decoded_chunk["text"])

        # Append the generated texts to messages
        for generated_text, message in zip(generated_texts, messages):
            message.append({"role": "assistant", "content": generated_text})
        return messages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()


if __name__ == "__main__":
    with RemoteVLLMModel(
        "meta-llama/Meta-Llama-3-8B-Instruct", sample_config=SamplingParams(n=1)
    ) as llm:
        res = llm.generate([[{"role": "system", "content": "Who are you?"}] * 1024])
        print(res)
