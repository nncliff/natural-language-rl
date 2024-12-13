from typing import Optional
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from nlrl.envs import get_env
from nlrl.envs.tictactoe.prompt import (
    POLICY_prompt,
)
from nlrl.config import EnvConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import copy
from torch.utils.checkpoint import checkpoint


class CheckpointedSequential(nn.Sequential):
    def forward(self, x):
        for module in self:
            x = checkpoint(module, x)
        return x


class PPOLLMAgent(nn.Module):
    def __init__(
        self,
        model_path,
        env_config,
        device="cuda",
        lr_actor=1e-5,
        lr_critic=1e-4,
        n_iterations=4,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        epsilon_greedy: Optional[float] = None,
        load_dir: Optional[str] = None,
        temperature: Optional[float] = 0.1,
    ):

        super(PPOLLMAgent, self).__init__()
        self.temperature = temperature
        self.model_path = model_path
        self.env_config = env_config
        self.device = device
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.n_iterations = n_iterations
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epsilon_greedy = epsilon_greedy
        self.load_dir = load_dir
        self.env = get_env(env_config)
        self.model_path = model_path

        print(f"\nInitializing model from {model_path}")
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Initial tokenizer vocabulary size: {len(self.tokenizer)}")

        print("Loading actor model...")
        self.actor = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        print(f"Actor loaded and moved to device: {self.device}")
        print(
            f"Initial actor model size: {sum(p.numel() for p in self.actor.parameters()) / 1e6:.2f}M parameters"
        )

        # Add special tokens for moves 1-9
        print("\nAdding special tokens for moves 1-9...")
        move_tokens = [f"{i}" for i in range(1, 10)]
        special_tokens = {"additional_special_tokens": move_tokens}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        print(f"Added {num_added} special tokens")
        print(f"New tokenizer vocabulary size: {len(self.tokenizer)}")

        print("Resizing token embeddings...")
        original_embedding_size = self.actor.get_input_embeddings().weight.shape[0]
        self.actor.resize_token_embeddings(len(self.tokenizer))
        new_embedding_size = self.actor.get_input_embeddings().weight.shape[0]
        print(
            f"Token embeddings resized from {original_embedding_size} to {new_embedding_size}"
        )

        hidden_size = self.actor.config.hidden_size
        self.critic = CheckpointedSequential(
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                # 添加Tanh限制value范围
                nn.Tanh(),  # 可以帮助稳定训练
            )
        )
        # self.add_module("actor", self.actor)
        # self.add_module("critic", self.critic)

        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_iterations = n_iterations
        self._eps_greedy = epsilon_greedy

        self.select_action_prompter = POLICY_prompt()
        print("Initialization completed.")

    def save_pretrained(self, save_directory: str):
        print(f"\nSaving model to {save_directory}")

        os.makedirs(f"{save_directory}/actor", exist_ok=True)

        self.actor.save_pretrained(
            f"{save_directory}/actor",
            safe_serialization=True
        )

        self.tokenizer.save_pretrained(
            f"{save_directory}/actor"
        )

        torch.save(
            {
                "critic": self.critic.state_dict(),
                # "cpu_optimizer": self.cpu_optimizer.state_dict(),
            },
            f"{save_directory}/auxiliary_models.pt",
        )

        print("Model saved successfully")

    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs):
        """从保存的目录加载模型"""
        print(f"\nLoading saved model from {pretrained_path}")

        instance = cls(
            model_path=f"{pretrained_path}/actor",
            # device="cpu",
            **kwargs
        )

        aux_path = f"{pretrained_path}/auxiliary_models.pt"
        if os.path.exists(aux_path):
            print("Loading auxiliary models and optimizers...")
            aux_dict = torch.load(aux_path, map_location="cpu")

            instance.critic.load_state_dict(aux_dict['critic'])
            print("Auxiliary models loaded successfully")
            del aux_dict
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print("No auxiliary models found, using initialized weights")

        return instance

    def get_prompt(self, state):
        """使用与Agent相同的prompt格式"""
        text_prompt = self.select_action_prompter(state, response_type="LLM")['prompt']
        full_prompt = self.tokenizer.apply_chat_template(
            text_prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
        return full_prompt.strip()

    def _get_action_mask(self, state, device):
        board, _ = state
        valid_moves = torch.zeros(9, dtype=torch.bool, device=device if device is not None else self.device)
        for i in range(9):
            if board[i] == 0:
                valid_moves[i] = True
        return valid_moves

    def _get_move_token_ids(self, device=None):
        """获取1-9动作的token IDs"""
        move_token_ids = []
        for i in range(1, 10):  # 1-9
            token = self.tokenizer.encode(str(i), add_special_tokens=False)[0]
            move_token_ids.append(token)
        return torch.tensor(move_token_ids, device=device if device is not None else self.device)

    def _get_action_probs(self, state, device=None):
        """获取动作概率分布"""
        device = device if device is not None else self.device
        self.actor.config.output_hidden_states = True
        # 准备输入
        prompt = self.get_prompt(state)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True,
        ).to(device)

        # 确保模型在正确的设备上
        self.actor.to(device)

        # Forward pass
        outputs = self.actor(**inputs)

        # 获取hidden states和logits
        hidden_states = (
            outputs.hidden_states[-1]
            if hasattr(outputs, "hidden_states")
            else outputs.last_hidden_state
        )
        last_hidden = hidden_states[:, -1, :]
        logits = outputs.logits[:, -1, :]

        # 检查维度
        assert hidden_states.dim() == 3, f"Expected hidden_states to be 3D, got {hidden_states.dim()}D"
        assert last_hidden.dim() == 2, f"Expected last_hidden to be 2D, got {last_hidden.dim()}D"
        assert logits.dim() == 2, f"Expected logits to be 2D, got {logits.dim()}D"
        assert last_hidden.size(0) == 1, f"Expected batch size 1, got {last_hidden.size(0)}"

        # 检查梯度
        assert logits.requires_grad, "Logits must require gradients"
        assert last_hidden.requires_grad, "Hidden states must require gradients"

        # 获取和处理move token ids
        move_token_ids = self._get_move_token_ids(device=device)
        move_logits = logits[:, move_token_ids]
        assert move_logits.size() == (1, 9), f"Expected move_logits size (1, 9), got {move_logits.size()}"

        # 应用action mask
        valid_moves = self._get_action_mask(state, device)
        assert valid_moves.size() == (9,), f"Expected valid_moves size (9,), got {valid_moves.size()}"

        masked_logits = torch.where(
            valid_moves, move_logits, torch.tensor(float("-inf"), device=device)
        )

        assert masked_logits.requires_grad, "Masked logits must require gradients"
        assert masked_logits.size() == (1, 9), f"Expected masked_logits size (1, 9), got {masked_logits.size()}"

        return masked_logits, last_hidden  # 返回logits而不是probs

    def get_action(self, state, device=None):
        """选择动作"""
        if self._eps_greedy and np.random.random() < self._eps_greedy:
            valid_actions = [i for i in range(9) if state[0][i] == 0]
            action = np.random.choice(valid_actions)
            return action, None, None

        logits, _ = self._get_action_probs(state, device=device)
        if self.temperature == 0:
            action_idx = torch.argmax(logits)
            return action_idx.item(), None, None
        else:
            # 只在这里应用一次temperature和softmax
            scaled_logits = logits / self.temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            dist = Categorical(probs)
            action_idx = dist.sample()

            log_prob = dist.log_prob(action_idx)
            entropy = dist.entropy()

            return action_idx.item(), log_prob, entropy

    def evaluate_actions(self, states, actions, device=None):
        """评估给定的状态-动作对"""
        batch_values = []
        batch_log_probs = []
        batch_entropy = []

        # 检查输入
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions)
        if actions.device != device:
            actions = actions.to(device)

        for state, action in zip(states, actions):
            # 获取logits和critic的hidden state
            logits, hidden = self._get_action_probs(state, device)
            assert logits.size() == (1, 9), f"Expected logits size (1, 9), got {logits.size()}"
            assert hidden.dim() == 2, f"Expected hidden to be 2D, got {hidden.dim()}D"
            # 应用temperature并计算概率
            scaled_logits = logits / self.temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            assert probs.sum().item() - 1.0 < 1e-6, "Probabilities must sum to 1"

            dist = Categorical(probs)
            assert 0 <= action < 9, f"Action must be in [0, 8], got {action}"

            # 计算log_prob
            log_prob = dist.log_prob(action.to(device))
            entropy = dist.entropy()

            # 计算value
            self.critic.to(device)
            value = self.critic(hidden)

            assert log_prob.dim() == 1, f"Expected log_prob to be 1D, got {log_prob.dim()}D"
            assert entropy.dim() == 1, f"Expected entropy to be 1D, got {entropy.dim()}D"
            assert value.dim() == 2, f"Expected value to be 2D, got {value.dim()}D"

            assert log_prob.requires_grad, "Log probabilities must require gradients"
            assert entropy.requires_grad, "Entropy must require gradients"
            assert value.requires_grad, "Values must require gradients"

            batch_values.append(value)
            batch_log_probs.append(log_prob)
            batch_entropy.append(entropy)

        values = torch.stack(batch_values).squeeze(-1)
        log_probs = torch.stack(batch_log_probs).squeeze(-1)
        entropies = torch.stack(batch_entropy).squeeze(-1)

        assert values.size(1) == 1, f"Expected values to be [batch_size, 1], got {values.size()}"
        assert log_probs.dim() == 1, f"Expected log_probs to be 1D [batch_size], got {log_probs.dim()}D"
        assert entropies.dim() == 1, f"Expected entropies to be 1D [batch_size], got {entropies.dim()}D"

        return values, log_probs, entropies

    def get_value(self, state, device):
        """获取状态价值"""
        _, hidden = self._get_action_probs(state, device)
        assert hidden.dim() == 2, f"Expected hidden to be 2D, got {hidden.dim()}D"
        assert hidden.size(0) == 1, f"Expected batch size 1, got {hidden.size(0)}"

        self.critic.to(device)
        value = self.critic(hidden)
        assert value.requires_grad, "Value must require gradients"

        if value.dim() > 1:
            value = value.squeeze(-1)
        assert value.dim() == 1, f"Expected value to be 1D after squeeze, got {value.dim()}D"

        return value

    def get_batch_action(self, state_list):
        """处理批量状态"""
        actions = []
        for state in state_list:
            action, _, _ = self.get_action(state)
            actions.append(action)
        return actions

    def __call__(self, state):
        """保持与Agent相同的接口"""
        if isinstance(state, list):
            return self.get_batch_action(state)
        action, _, _ = self.get_action(state)
        return action

    # def evaluate_actions_with_cpu_offload(self, states, actions, device=None):
    #     """分段执行forward pass以减少内存使用"""
    #     batch_values = []
    #     batch_log_probs = []
    #     batch_entropy = []

    #     for state, action in zip(states, actions):
    #         # 获取action概率分布和critic的hidden state
    #         probs, hidden = self._get_action_probs_with_offload(state, device)

    #         # 计算action的log probability
    #         dist = Categorical(probs)
    #         log_prob = dist.log_prob(action.to(device))  # 确保action在正确设备上
    #         entropy = dist.entropy()

    #         # 计算value
    #         value = self.critic(hidden.to(device))

    #         # 验证梯度
    #         assert log_prob.requires_grad, "Log probabilities must require gradients"
    #         assert entropy.requires_grad, "Entropy must require gradients"
    #         assert value.requires_grad, "Values must require gradients"

    #         batch_values.append(value)
    #         batch_log_probs.append(log_prob)
    #         batch_entropy.append(entropy)

    #         # 清理内存
    #         del probs, hidden
    #         torch.cuda.empty_cache()

    #     # Stack并移到目标设备
    #     values = torch.stack(batch_values).squeeze()
    #     log_probs = torch.stack(batch_log_probs)
    #     entropies = torch.stack(batch_entropy)

    #     return values, log_probs, entropies

    # def _get_action_probs_with_offload(self, state, device=None):
    #     """带CPU offloading的action概率计算"""
    #     # device = next(self.actor.parameters()).device

    #     prompt = self.get_prompt(state)
    #     inputs = self.tokenizer(
    #         prompt, return_tensors="pt", max_length=512, truncation=True, padding=True
    #     )
    #     inputs = {k: v.to(device) for k, v in inputs.items()}

    #     # 分段处理以减少内存使用
    #     # inputs = {k: v.to(self.device) for k, v in inputs.items()}

    #     # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    #     self.actor.to(device if device is not None else self.device)

    #     outputs = self.actor(**inputs)

    #     if hasattr(outputs, "hidden_states"):
    #         hidden_states = outputs.hidden_states[-1]
    #     else:
    #         hidden_states = outputs.last_hidden_state

    #     last_hidden = hidden_states[:, -1, :]
    #     logits = outputs.logits[:, -1, :]
    #     assert logits.requires_grad, "Actor logits must require gradients"
    #     assert last_hidden.requires_grad, "Actor hidden states must require gradients"
    #     # 处理action logits
    #     move_token_ids = self._get_move_token_ids(device=device)
    #     move_logits = logits[:, move_token_ids]
    #     assert move_logits.requires_grad, "Move logits must require gradients"

    #     # 应用action mask
    #     valid_moves = self._get_action_mask(state, device)
    #     masked_logits = torch.where(
    #         valid_moves, move_logits, torch.tensor(float("-inf"), device=device)
    #     )
    #     assert masked_logits.requires_grad, "Masked logits must require gradients"

    #     probs = torch.softmax(masked_logits, dim=-1)
    #     assert probs.requires_grad, "Final probabilities must require gradients"

    #     return probs, last_hidden


if __name__ == "__main__":
    import json
    from nlrl.config import EnvConfig

    def test_prompts():
        print("\nTesting prompt generation...")
        policy = PPOLLMAgent(
            "/mnt/ssd/benjamin/assets/models/meta-llama/Meta-Llama-3.1-8B-Instruct",
            EnvConfig(env_name="TicTacToeEnv"),
        )

        # 使用真实数据格式测试
        test_states = [
            [[0, 0, 0, 0, 0, 0, 0, 0, 0], "O"],  # 空棋盘
            [[0, 0, 0, 0, 1, 0, 0, 0, 0], "X"],  # 一步棋
            [[2, 0, 0, 0, 1, 0, 0, 0, 1], "O"],  # 多步棋
            [[2, 1, 2, 1, 1, 2, 1, 2, 1], "X"],  # 终盘
        ]

        for state in test_states:
            prompt = policy.get_prompt(state)
            print(f"\nBoard state: {state[0]}")
            print(f"Current player: {state[1]}")
            print("Generated prompt:")
            print(prompt)

            # 检查提示中的关键部分
            assert isinstance(prompt, str)
            assert "board" in prompt.lower()
            assert "best_move" in prompt
            assert state[1] in prompt  # 确保玩家标识在提示中
            assert "available positions" in prompt.lower()

    def test_action_mask():
        print("\nTesting action masking...")
        policy = PPOLLMAgent(
            "/mnt/ssd/benjamin/assets/models/meta-llama/Meta-Llama-3.1-8B-Instruct",
            EnvConfig(env_name="TicTacToeEnv"),
        )

        # 使用真实数据格式测试
        test_states = [
            # 从真实数据选取的状态
            [[0, 0, 0, 0, 0, 0, 0, 0, 0], "O"],  # 空棋盘
            [[0, 0, 0, 0, 1, 0, 0, 0, 0], "X"],  # 一步棋
            [[2, 0, 0, 0, 1, 0, 0, 0, 1], "O"],  # 多步棋
            [[2, 1, 2, 1, 1, 2, 1, 2, 1], "X"],  # 终盘
        ]

        for state in test_states:
            mask = policy._get_action_mask(state)
            print(f"\nBoard state: {state[0]}")
            print(f"Player: {state[1]}")
            print("Action mask:", mask)

            # 检查掩码的正确性
            assert len(mask) == 9, "Mask should have length 9"
            empty_positions = sum(1 for pos in state[0] if pos == 0)
            assert (
                mask.sum() == empty_positions
            ), f"Should have {empty_positions} valid actions"

            # 验证掩码与棋盘一致
            for i, pos in enumerate(state[0]):
                if pos == 0:
                    assert mask[i], f"Position {i} should be valid"
                else:
                    assert not mask[i], f"Position {i} should be invalid"

    def test_action_probs():
        print("\nTesting action probability generation...")
        policy = PPOLLMAgent(
            "/mnt/ssd/benjamin/assets/models/meta-llama/Meta-Llama-3.1-8B-Instruct",
            EnvConfig(env_name="TicTacToeEnv"),
        )

        test_states = [
            [[0, 0, 0, 0, 0, 0, 0, 0, 0], "O"],  # 空棋盘
            [[0, 0, 0, 0, 1, 0, 0, 2, 0], "X"],  # 进行中的棋局
        ]

        for state in test_states:
            probs, hidden = policy._get_action_probs(state)
            print(f"\nBoard state: {state[0]}")
            print(f"Current player: {state[1]}")
            print("Action probabilities:", probs[0].cpu().detach().numpy())
            print("Hidden state shape:", hidden.shape)

            # 检查概率分布
            assert probs.shape[-1] == 9, "Should have probabilities for all 9 positions"
            assert torch.isclose(probs.sum(), torch.tensor(1.0, device=probs.device))
            assert (probs >= 0).all(), "All probabilities should be non-negative"

            # 验证无效位置的概率为0
            for i, pos in enumerate(state[0]):
                if pos != 0:
                    assert (
                        probs[0][i].item() == 0
                    ), f"Probability for occupied position {i} should be 0"

    def test_value_estimation():
        print("\nTesting value estimation...")
        policy = PPOLLMAgent(
            "/mnt/ssd/benjamin/assets/models/meta-llama/Meta-Llama-3.1-8B-Instruct",
            EnvConfig(env_name="TicTacToeEnv"),
        )

        test_states = [
            [[0, 0, 0, 0, 0, 0, 0, 0, 0], "O"],  # 初始状态
            [[0, 0, 0, 0, 1, 0, 0, 2, 0], "X"],  # 中期状态
            [[2, 1, 2, 1, 1, 2, 0, 2, 1], "O"],  # 接近胜利的状态
        ]

        for state in test_states:
            value = policy.get_value(state)
            print(f"\nBoard state: {state[0]}")
            print(f"Player: {state[1]}")
            print("Estimated value:", value.item())

            assert value.shape == (1, 1), "Value should be a single scalar"
            assert not torch.isnan(value).any(), "Value should not be NaN"
            assert -1 <= value.item() <= 1, "Value should be in [-1, 1] range"

    def test_action_selection():
        print("\nTesting action selection...")
        policy = PPOLLMAgent(
            "/mnt/ssd/benjamin/assets/models/meta-llama/Meta-Llama-3.1-8B-Instruct",
            EnvConfig(env_name="TicTacToeEnv"),
            epsilon_greedy=0.1,
        )

        # 使用真实数据格式的测试状态
        test_states = [
            [[0, 0, 0, 0, 0, 0, 0, 0, 0], "O"],  # 空棋盘
            [[0, 0, 0, 0, 1, 0, 0, 0, 0], "X"],  # X刚下了4
        ]

        print("\nTesting single action selection:")
        for state in test_states:
            action, log_prob, entropy = policy.get_action(state)
            print(f"\nBoard state: {state[0]}")
            print(f"Current player: {state[1]}")
            print(
                f"Selected action (raw 0-8 format): {action}"
            )
            print(
                f"Log probability: {log_prob.item() if log_prob is not None else None}"
            )
            print(f"Entropy: {entropy.item() if entropy is not None else None}")

            assert 0 <= action <= 8, "Raw action should be in range 0-8"
            assert (
                state[0][action] == 0
            ), "Selected action should point to empty position"

        print("\nTesting batch action selection:")
        batch_actions = policy.get_batch_action(test_states)
        print(
            "Batch actions (1-9 format):", batch_actions
        )
        assert len(batch_actions) == len(test_states)
        assert all(
            1 <= a <= 9 for a in batch_actions
        ), "API actions should be in range 1-9"
        for state, action in zip(test_states, batch_actions):
            assert state[0][action - 1] == 0, "All selected positions should be empty"

    def test_action_evaluation():
        print("\nTesting action evaluation...")
        policy = PPOLLMAgent(
            "/mnt/ssd/benjamin/assets/models/meta-llama/Meta-Llama-3.1-8B-Instruct",
            EnvConfig(env_name="TicTacToeEnv"),
        )

        test_cases = [
            {
                "states": [
                    [[0, 0, 0, 0, 0, 0, 0, 0, 0], "O"],
                    [[0, 0, 0, 0, 1, 0, 0, 0, 0], "X"],
                ],
                "actions": [5, 8],
            },
            {
                "states": [
                    [[0, 0, 0, 0, 1, 0, 0, 2, 0], "O"],
                ],
                "actions": [3],
            },
        ]

        for i, test_case in enumerate(test_cases):
            print(f"\nTest case {i+1}:")
            values, log_probs, entropies = policy.evaluate_actions(
                test_case["states"], test_case["actions"]
            )

            values_np = values.cpu().detach().numpy()
            log_probs_np = log_probs.cpu().detach().numpy()
            entropies_np = entropies.cpu().detach().numpy()

            values_np = np.atleast_1d(values_np)
            log_probs_np = np.atleast_1d(log_probs_np)
            entropies_np = np.atleast_1d(entropies_np)

            print(f"States: {[s[0] for s in test_case['states']]}")
            print(f"Players: {[s[1] for s in test_case['states']]}")
            print(f"Actions (1-9 format): {test_case['actions']}")
            print("Values:", values_np)
            print("Log probs:", log_probs_np)
            print("Entropies:", entropies_np)

            assert values_np.size == len(test_case["states"])
            assert log_probs_np.size == len(test_case["actions"])
            assert entropies_np.size == len(test_case["states"])

            assert not np.isnan(values_np).any()
            assert not np.isnan(log_probs_np).any()
            assert not np.isnan(entropies_np).any()
            assert (entropies_np >= 0).all()
            assert (log_probs_np <= 0).all()
            assert (-20 < log_probs_np).all()

            for state, action in zip(test_case["states"], test_case["actions"]):
                assert (
                    state[0][action - 1] == 0
                ), f"Position {action} should be empty in state {state[0]}"

            print(f"Test case {i+1} passed all checks!")

    def run_all_tests():
        try:
            print("Starting PPO Policy tests...")
            print("Model loading may take a few moments...")
            test_prompts()
            test_action_mask()
            test_action_probs()
            test_value_estimation()
            test_action_selection()
            test_action_evaluation()
            print("\nAll tests passed successfully!")
        except Exception as e:
            print(f"\nTest failed: {str(e)}")
            import traceback

            traceback.print_exc()
            raise

    # Run all tests
    run_all_tests()
