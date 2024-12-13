# pipeline/train_ppo.py


import argparse
import os
import random
import sys
import time
from json import load
import re
import torch


from nlrl.config import EnvConfig
from nlrl.utils import read_jsonl, write_jsonl
from nlrl.policy import PPOLLMAgent

import numpy as np
from tqdm import tqdm
from torch.distributions import Categorical
from torch.utils.data import Dataset
from torch.optim import Adam
from torch import nn
from torch.nn import functional as F
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import copy
import math
import time
import torch.cuda.amp as amp
from torch.distributed.utils import _alloc_storage, _free_storage
from torch.utils.checkpoint import checkpoint
from functools import partial
import wandb


def process_trajectories(replay_buffer):
    """Improved trajectory processing"""
    trajectories = []

    for episode in replay_buffer:
        we_are_O = episode["state"][0][1] == "O" and episode["turn"][0]

        traj_data = {
            "states": [],
            "next_states": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "returns": [],
            "advantages": [],
            "dones": [],
        }

        episode_length = len(episode["action"])
        last_value = None

        for i in range(episode_length):
            current_player = episode["state"][i][1]
            is_our_turn = (current_player == "O" and we_are_O) or (
                current_player == "X" and not we_are_O
            )

            if is_our_turn:
                board_state = episode["state"][i]
                next_board_state = episode["state"][i + 1]
                action = episode["action"][i]

                # 处理奖励
                reward = episode["reward"][i]
                if not we_are_O:
                    reward = -reward

                done = (abs(reward) > 0) or (i == len(episode["action"]) - 1)
                # print(board_state)
                # print(next_board_state)
                traj_data["states"].append(board_state)
                traj_data["next_states"].append(next_board_state)
                traj_data["actions"].append(action)
                traj_data["rewards"].append(reward)
                traj_data["dones"].append(done)

        if len(traj_data["states"]) > 0:
            trajectories.append(traj_data)
    print(f"Processed {len(trajectories)} trajectories")
    return trajectories


def compute_gae(rewards, values, next_values, dones, gamma=0.99, lambda_=0.95):
    assert (
        len(rewards) == len(values) == len(next_values) == len(dones)
    ), "All inputs must have same length"
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_value = 0
        else:
            next_value = next_values[t]

        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lambda_ * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def normalize_advantages(advantages):
    advantages = torch.tensor(advantages)
    # 减均值除标准差
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 1e-8防止除零


def collate_ppo_batch(batch):
    """自定义的collate函数，确保正确的设备放置"""
    states = [item[0] for item in batch]  # 保持list格式
    # 确保所有tensor都在CPU上
    actions = torch.stack([item[1].cpu() for item in batch])
    advantages = torch.stack([item[2].cpu() for item in batch])
    returns = torch.stack([item[3].cpu() for item in batch])
    old_log_probs = torch.stack([item[4].cpu() for item in batch])

    return states, actions, advantages, returns, old_log_probs


# class PPODataset(Dataset):
#     def __init__(self, states, actions, advantages, returns, old_log_probs):
#         self.states = states  # 保持states为list
#         # 确保所有tensor都在CPU上
#         self.actions = actions.cpu()
#         self.advantages = advantages.cpu()
#         self.returns = returns.cpu()
#         self.old_log_probs = old_log_probs.cpu()


class PPODataset(Dataset):
    def __init__(self, states, actions, advantages, returns, old_log_probs):
        assert (
            len(states)
            == len(actions)
            == len(advantages)
            == len(returns)
            == len(old_log_probs)
        ), "All inputs must have same length"
        if len(states) == 0:
            raise ValueError("Empty trajectory data")

        self.states = states
        self.actions = torch.as_tensor(actions, dtype=torch.long)
        self.advantages = torch.as_tensor(advantages)
        self.returns = torch.as_tensor(returns)
        self.old_log_probs = torch.as_tensor(old_log_probs)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.advantages[idx],
            self.returns[idx],
            self.old_log_probs[idx],
        )


def update_policy(args, policy, trajectories, device, batch_size=64, rank=0, world_size=1, optimizer=None, gradient_accumulation_steps=4, scheduler=None):

    if not trajectories:
        raise ValueError("Empty trajectories provided")

    # Collect and validate data
    all_states = []
    all_actions = []
    all_advantages = []
    all_returns = []
    all_old_log_probs = []

    for traj in trajectories:
        # Validate trajectory data
        required_keys = ["states", "actions", "advantages", "returns", "log_probs"]
        if not all(key in traj for key in required_keys):
            raise KeyError(
                f"Missing required keys in trajectory. Required: {required_keys}"
            )

        all_states.extend(traj["states"])
        all_actions.extend(traj["actions"])
        all_advantages.extend(traj["advantages"])
        all_returns.extend(traj["returns"])
        all_old_log_probs.extend(traj["log_probs"])

    # Convert to tensors and normalize advantages
    all_actions = torch.tensor(all_actions, device="cpu")
    all_advantages = normalize_advantages(all_advantages).cpu()
    all_returns = torch.tensor(all_returns, device="cpu")
    all_old_log_probs = torch.tensor(all_old_log_probs, device="cpu")

    # 创建dataset和sampler
    dataset = PPODataset(
        all_states,
        all_actions,
        all_advantages,
        all_returns,
        all_old_log_probs,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_ppo_batch,
        pin_memory=True,  # 现在可以安全地使用pin_memory
        num_workers=0,
        drop_last=True,  # 丢弃不完整的最后一个batch
    )

    total_batches = len(data_loader)
    total_steps = (
        total_batches * args.num_iterations
    )  # Total steps across all iterations

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=total_steps,  # Now considers all iterations
    )

    # 训练循环
    total_loss = 0
    total_actor_loss = 0
    total_value_loss = 0
    total_entropy = 0
    n_batches = 0

    # Early stopping 参数
    best_loss = float("inf")
    patience = 1
    no_improve_steps = 0
    min_improvement = 1e-5
    early_stop = False

    # 记录每个step的loss
    losses_history = []

    optimizer.zero_grad()

    for batch_idx, (
        batch_states,
        batch_actions,
        batch_advantages,
        batch_returns,
        batch_old_log_probs,
    ) in enumerate(data_loader):
        # print(f"Batch {batch_idx} - Memory before forward pass:")
        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
        # batch_states已经是list格式，不需要转换
        # 确保其他数据在正确的设备上
        # 手动将tensors移到GPU
        values = log_probs = entropy = total_loss = None
        surr1 = surr2 = ratio = actor_loss = value_loss = entropy_loss = None
        current_actions = current_advantages = current_returns = (
            current_old_log_probs
        ) = None
        try:
            batch_actions = batch_actions.to(device)
            batch_advantages = batch_advantages.to(device)
            batch_returns = batch_returns.to(device)
            batch_old_log_probs = batch_old_log_probs.to(device)

            values, log_probs, entropy = (
                policy.evaluate_actions(
                    batch_states, batch_actions, device
                )
            )

            # 再次确认关键tensor有梯度
            assert log_probs.requires_grad, "log_probs in update must require gradients"
            assert values.requires_grad, "values in update must require gradients"
            assert entropy.requires_grad, "entropy in update must require gradients"

            batch_old_log_probs = batch_old_log_probs.detach()
            assert (
                not batch_old_log_probs.requires_grad
            ), "old_log_probs should not require gradients"

            # 计算policy loss
            ratio = torch.exp(log_probs - batch_old_log_probs)
            assert ratio.requires_grad, "policy ratio must require gradients"

            # 再次检查ratio是否有梯度
            # if rank == 0:
            #     print("ratio requires grad:", ratio.requires_grad)

            surr1 = ratio * batch_advantages
            surr2 = (
                torch.clamp(
                    ratio,
                    1.0 - policy.clip_epsilon,
                    1.0 + policy.clip_epsilon,
                )
                * batch_advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()

            # 计算value loss
            value_loss = 0.5 * (batch_returns - values).pow(2).mean()
            entropy_loss = -entropy.mean()

            assert actor_loss.requires_grad, "actor_loss must require gradients"
            assert value_loss.requires_grad, "value_loss must require gradients"
            assert entropy_loss.requires_grad, "entropy_loss must require gradients"

            # 总loss
            total_loss = (
                actor_loss
                + policy.value_coef * value_loss
                + policy.entropy_coef * entropy_loss
            ) / gradient_accumulation_steps
            assert total_loss.requires_grad, "total_loss must require gradients"
            total_loss.backward()

            # print(total_loss)
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad()
                # Early stopping 检查
                current_loss = value_loss.item() * gradient_accumulation_steps
                losses_history.append(current_loss)

                current_lr_actor = scheduler.get_last_lr()[0]
                current_lr_critic = scheduler.get_last_lr()[1]
                wandb.log(
                    {
                        "lr_actor": current_lr_actor,
                        "lr_critic": current_lr_critic,
                        "batch_actor_loss": actor_loss.item(),
                        "batch_value_loss": value_loss.item(),
                        "batch_entropy": entropy.mean().item(),
                        "batch_total_loss": total_loss.item()
                        * gradient_accumulation_steps,
                        "policy_ratio_mean": ratio.mean().item(),
                        "policy_ratio_std": ratio.std().item(),
                        "advantages_mean": batch_advantages.mean().item(),
                        "advantages_std": batch_advantages.std().item(),
                    }
                )

                # 计算最近几个batch的平均loss来平滑波动
                # window_size = 5
                # if len(losses_history) >= window_size:
                #     avg_loss = sum(losses_history[-window_size:]) / window_size

                #     if avg_loss < best_loss - min_improvement:
                #         best_loss = avg_loss
                #         no_improve_steps = 0
                #         # 可以在这里保存最佳模型状态
                #         if rank == 0:  # 只在主进程打印
                #             print(f"\nLoss improved to {best_loss:.6f}")
                #     else:
                #         no_improve_steps += 1
                #         if rank == 0:  # 只在主进程打印
                #             print(
                #                 f"\nNo improvement for {no_improve_steps} steps. Best loss: {best_loss:.6f}, Current loss: {avg_loss:.6f}"
                #             )

                #         if no_improve_steps >= patience:
                #             if rank == 0:
                #                 print(
                #                     f"\nEarly stopping triggered after {batch_idx + 1} batches"
                #                 )
                #             early_stop = True
                # break

            if batch_idx > 0:
                print(f"Batch {batch_idx}")
                print(f"Actor loss: {total_actor_loss / n_batches}")
                print(f"Value loss: {total_value_loss / n_batches}")
                print(f"Entropy: {total_entropy / n_batches}")
                print(
                    f"Total loss: {(total_actor_loss + policy.value_coef * total_value_loss + policy.entropy_coef * total_entropy) / n_batches}"
                )
                wandb.log(
                    {
                        "running_actor_loss": total_actor_loss / n_batches,
                        "running_value_loss": total_value_loss / n_batches,
                        "running_entropy": total_entropy / n_batches,
                        "running_total_loss": (
                            total_actor_loss
                            + policy.value_coef * total_value_loss
                            + policy.entropy_coef * total_entropy
                        )
                        / n_batches,
                    }
                )

            torch.cuda.empty_cache()
            total_loss += (
                actor_loss.item()
                + policy.value_coef * value_loss.item()
                + policy.entropy_coef * entropy_loss.mean().item()
            )
            total_actor_loss += actor_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            n_batches += 1
        except Exception as e:
            print(f"Error in update_policy: {str(e)}")
            raise

        finally:

            del values, log_probs, entropy, total_loss
            del surr1, surr2, ratio, actor_loss, value_loss, entropy_loss
            if "current_actions" in locals():
                del current_actions
            if "current_advantages" in locals():
                del current_advantages
            if "current_returns" in locals():
                del current_returns
            if "current_old_log_probs" in locals():
                del current_old_log_probs
            torch.cuda.empty_cache()

        # if early_stop:
        #     break
        # if scheduler is not None:
        #     scheduler.step()

    # 同步多GPU的损失
    if world_size > 1:
        metrics = torch.tensor(
            [total_actor_loss, total_value_loss, total_entropy, n_batches],
            device=device,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_actor_loss, total_value_loss, total_entropy, n_batches = metrics.tolist()

    return {
        "actor_loss": total_actor_loss / n_batches,
        "value_loss": total_value_loss / n_batches,
        "entropy": total_entropy / n_batches,
        "total_loss": (total_actor_loss + policy.value_coef * total_value_loss + policy.entropy_coef * total_entropy)
        / n_batches,
        "early_stopped": early_stop,
        "best_loss": best_loss,
    }


def enable_checkpointing(sequential_model):
    original_forward_funcs = {}
    for name, module in sequential_model.named_children():
        original_forward_funcs[name] = module.forward
        module.forward = partial(checkpoint, module.forward)
    return original_forward_funcs


def train_ppo(args):
    """Non-distributed training function"""
    device = torch.device("cuda:0")

    # Initialize policy
    env_config = EnvConfig(env_name="TicTacToeEnv")
    if args.load_dir is not None:
        policy = PPOLLMAgent.from_pretrained(
            pretrained_path=args.load_dir,
            env_config=env_config,
            device=device,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            clip_epsilon=args.clip_epsilon,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            n_iterations=args.num_iterations,
            load_dir=args.load_dir,
        )
    else:
        policy = PPOLLMAgent(
            args.model_path,
            env_config,
            device=device,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            clip_epsilon=args.clip_epsilon,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            n_iterations=args.num_iterations,
        )

    # Enable training optimizations
    policy.actor.gradient_checkpointing_enable()
    enable_checkpointing(policy.critic)
    # policy.critic.gradient_checkpointing_enable()
    if hasattr(policy.actor.config, "use_cache"):
        policy.actor.config.use_cache = False

    policy.to(dtype=torch.bfloat16)
    policy.to(device)
    policy.train()

    # Load data
    print("Loading replay buffer...")
    replay_buffer = read_jsonl(args.replay_buffer_path)
    # replay_buffer = replay_buffer[:32]  # Small subset for testing
    trajectories = process_trajectories(replay_buffer)  # Use non-distributed version
    print(f"Loaded {len(trajectories)} valid trajectories")

    best_reward = float("-inf")
    # if hasattr(torch, "compile"):
    #     policy = torch.compile(policy)
    gpu_optimizer = torch.optim.AdamW(
        [
            {"params": policy.actor.parameters(), "lr": policy.lr_actor},
            {"params": policy.critic.parameters(), "lr": policy.lr_critic},
        ],
        eps=1e-8,
        weight_decay=0.01,
        foreach=False,  # Disable vectorized operations
        # fused=False,  # Disable fused operations
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     gpu_optimizer, T_max=args.n_epochs * args.num_iterations
    # )
    # scheduler = torch.optim.lr_scheduler.LinearLR(
    #     gpu_optimizer,
    #     start_factor=1.0,  # Start at 50% of base lr
    #     end_factor=0.5,  # End at 10%
    #     total_iters=args.num_iterations,
    # )
    for epoch in range(args.n_epochs):
        print(f"\nEpoch {epoch + 1}/{args.n_epochs}")
        print("Computing values and advantages...")
        total_reward = 0
        for traj in tqdm(trajectories):
            # 将states和next_states合并为一次inference
            states_list = traj["states"] + [traj["next_states"][-1]]  # 加入最后一个next_state
            # print(states_list)
            # 一次性计算所有states的values
            all_values = []
            # with torch.no_grad():
            for state in states_list:
                value = policy.get_value(state, device)
                all_values.append(value.item())

            # 分离values和next_values
            values = all_values[:-1]  # 除了最后一个
            next_values = all_values[1:]  # 除了第一个

            traj["values"] = values
            traj["next_values"] = next_values
            total_reward += sum(traj["rewards"])

            # 计算实际执行action的log_probs
            log_probs = []
            # with torch.no_grad():
            for state, action in zip(traj["states"], traj["actions"]):
                _, log_prob, _ = policy.evaluate_actions(
                    [state],
                    torch.tensor(
                        [action], device=device
                    ),  # 确保action是2D的 [1, 1]
                )
                log_probs.append(log_prob.item())
            traj["log_probs"] = log_probs

            # 使用新的GAE计算
            advantages, returns = compute_gae(
                traj["rewards"],
                values,
                next_values,
                traj["dones"]
            )
            traj["advantages"] = advantages
            traj["returns"] = returns

            torch.cuda.empty_cache()

        avg_reward = total_reward / len(trajectories)
        print(f"Average reward: {avg_reward:.4f}")

        if avg_reward > best_reward:
            best_reward = avg_reward
        wandb.log(
            {
                "epoch": epoch,
                "epoch_avg_reward": avg_reward,
                "epoch_best_reward": best_reward,
                "epoch_total_reward": total_reward,
            }
        )
        for _ in tqdm(range(args.num_iterations)):
            metrics = update_policy(
                args,
                policy,
                trajectories,
                device,
                batch_size=args.batch_size,
                rank=0,
                world_size=1,
                optimizer=gpu_optimizer,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                scheduler=None,
            )

            torch.cuda.empty_cache()

            # if metrics["early_stopped"]:
            #     print(f"Training stopped early in iteration {_+1} with best loss {metrics['best_loss']:.6f}")
            #     break

        # if metrics["early_stopped"]:
        #     print(f"Training stopped early in epoch {epoch+1}")
        #     break

        print(
            f"Losses - Actor: {metrics['actor_loss']:.4f}, "
            f"Value: {metrics['value_loss']:.4f}, "
            f"Entropy: {metrics['entropy']:.4f}, "
            f"Total: {metrics['total_loss']:.4f}"
        )

    # Save model
    print("\nSaving models...")
    policy.save_pretrained(args.save_dir)
    print(f"Training completed. Best average reward: {best_reward:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--replay_buffer_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--lr_actor", type=float, default=1e-5)
    parser.add_argument("--lr_critic", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--wandb_project", type=str, default="nlrl-tictactoe-ppo")
    parser.add_argument(
        "--wandb_entity", type=str, default="benjamin-eecs"
    )  # Your wandb username
    args = parser.parse_args()

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
    )

    train_ppo(args)
    wandb.finish()
