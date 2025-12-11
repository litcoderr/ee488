import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist

from rl_trader.agents import (
    DQNAgent,
    DQNConfig,
    PPOAgent,
    PPOConfig,
    flatten_observation,
    get_rank,
    to_tensor,
)
from rl_trader.env import EnvConfig, TradingEnv


def init_distributed():
    if dist.is_available() and not dist.is_initialized() and "RANK" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")


def device_from_env() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


@dataclass
class TrainConfig:
    algorithm: str = "dqn"  # "dqn" or "ppo"
    episodes: int = 50
    max_steps_per_episode: int = 500
    eval_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    rollout_length: int = 512  # used by PPO
    save_final: bool = True


class Trainer:
    def __init__(self, env_cfg: EnvConfig, train_cfg: TrainConfig, test_env_cfg: Optional[EnvConfig] = None):
        init_distributed()
        self.rank = get_rank()
        self.device = device_from_env()
        self.env_cfg = env_cfg
        self.test_env_cfg = test_env_cfg or env_cfg
        self.train_cfg = train_cfg
        self.env = TradingEnv(env_cfg)
        obs_shape = flatten_observation(self.env.reset()[0]).shape[0]
        if train_cfg.algorithm.lower() == "dqn":
            self.agent = DQNAgent(obs_shape, self.env.action_space.n, DQNConfig(), self.device)
            self.algo = "dqn"
        elif train_cfg.algorithm.lower() == "ppo":
            discrete = hasattr(self.env.action_space, "n")
            self.agent = PPOAgent(obs_shape, self.env.action_space, PPOConfig(rollout_length=train_cfg.rollout_length), self.device, discrete)
            self.algo = "ppo"
        else:
            raise ValueError("Unsupported algorithm")
        self.checkpoint_dir = Path(train_cfg.checkpoint_dir) / self.algo
        if self.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _checkpoint_path(self, name: str) -> Path:
        return self.checkpoint_dir / f"{name}.pt"

    def save(self, name: str = "latest"):
        if self.rank != 0:
            return
        ckpt = {"env_cfg": self.env_cfg.__dict__, "algo": self.algo, "state": self.agent.state_dict()}
        torch.save(ckpt, self._checkpoint_path(name))

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if ckpt.get("algo") != self.algo:
            raise ValueError(f"Checkpoint algorithm {ckpt.get('algo')} does not match trainer algorithm {self.algo}")
        self.agent.load_state_dict(ckpt["state"])

    def train(self):
        if self.algo == "dqn":
            self._train_dqn()
        else:
            self._train_ppo()
        if self.train_cfg.save_final:
            self.save("final")

    def _train_dqn(self):
        for ep in range(self.train_cfg.episodes):
            obs, _ = self.env.reset(seed=self.env_cfg.seed)
            total_reward = 0.0
            for step in range(self.train_cfg.max_steps_per_episode):
                action = self.agent.act(obs, explore=True)
                next_obs, reward, done, truncated, info = self.env.step(action)
                self.agent.add(obs, action, reward, next_obs, done or truncated)
                metrics = self.agent.update()
                total_reward += reward
                obs = next_obs
                if done or truncated:
                    break
            if self.rank == 0 and ep % max(1, self.train_cfg.eval_interval) == 0:
                eval_value = self.evaluate()
                print(f"[DQN] Ep {ep} | reward {total_reward:.2f} | eval value {eval_value:.2f} | steps {step}")

    def _train_ppo(self):
        for ep in range(self.train_cfg.episodes):
            obs, _ = self.env.reset(seed=self.env_cfg.seed)
            ep_reward = 0.0
            for _ in range(self.train_cfg.max_steps_per_episode):
                action, logprob, value = self.agent.act(obs, explore=True)
                next_obs, reward, done, truncated, info = self.env.step(action)
                self.agent.store(obs, action, reward, done or truncated, logprob, value)
                ep_reward += reward
                obs = next_obs
                if self.agent.ready():
                    with torch.no_grad():
                        _, next_value = self.agent.ac(to_tensor(flatten_observation(obs), self.device).unsqueeze(0))
                    self.agent.update(next_value.item())
                if done or truncated:
                    break
            if self.agent.buffer["obs"]:
                with torch.no_grad():
                    _, next_value = self.agent.ac(to_tensor(flatten_observation(obs), self.device).unsqueeze(0))
                self.agent.update(next_value.item())
            if self.rank == 0 and ep % max(1, self.train_cfg.eval_interval) == 0:
                eval_value = self.evaluate()
                print(f"[PPO] Ep {ep} | reward {ep_reward:.2f} | eval value {eval_value:.2f}")

    def evaluate(self, episodes: int = 3) -> float:
        test_env = TradingEnv(self.test_env_cfg)
        final_values = []
        for _ in range(episodes):
            obs, _ = test_env.reset(seed=self.env_cfg.seed)
            done = False
            value = 0.0
            while not done:
                action = self.agent.act(obs, explore=False)
                obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated
                value = info.get("portfolio_value", value)
            final_values.append(value)
        return float(sum(final_values) / len(final_values))
