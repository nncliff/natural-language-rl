from dataclasses import dataclass
from typing import Optional


@dataclass(kw_only=True)
class ValueDataConfig:
    data_path: str = "./llm_outputs.jsonl"
    dataloader_params: dict = None
    rollout_per_state: int = 1
    target_method: str = "mc"
    save_dir: str = "nlrl/envs/tictactoe"


@dataclass(kw_only=True)
class EnvConfig:
    env_name: str = "TicTacToeEnv"
    batch_sample: bool = False
    batch_sample_size: int = 64


@dataclass(kw_only=True)
class LLMSamplingParams:
    n: int = 3
    temperature: float = 0.1
    top_k: int = 50
    top_p: float = 0.95
    max_tokens: int = 256
    prompt_logprobs: int = 0


@dataclass(kw_only=True)
class PolicyConfig:
    policy_name: str = "TicTacToe_minmax"
    model_path: str = None
    env_config: EnvConfig = None
    llm_config: LLMSamplingParams = None
    epsilon_greedy: Optional[float] = None


@dataclass(kw_only=True)
class DataConfig:
    data_path: str
    batch_size: int = 1
    shuffle: bool = False
    drop_reminder: bool = True
    value_data_config: ValueDataConfig = None
    env_config: str = "TicTacToeEnv"
    policy_config: PolicyConfig = None
    replay_buffer_path: str = ""
    prompt_dir: str = ""


if __name__ == "__main__":
    data_config = DataConfig(
        data_path="./sampling/best_td_agent.dat",
        batch_size=1,
        shuffle=False,
        drop_reminder=True,
    )
    print(data_config)
