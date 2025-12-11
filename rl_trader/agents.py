from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim


def flatten_observation(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        parts = []
        for key in sorted(obs.keys()):
            val = np.asarray(obs[key], dtype=np.float32).reshape(-1)
            parts.append(val)
        return np.concatenate(parts, axis=0)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def to_tensor(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256)):
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_obs=self.next_obs_buf[idxs],
            dones=self.dones[idxs],
        )
        return batch

    def ready(self, warmup: int) -> bool:
        return self.size >= warmup


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 100_000
    warmup: int = 1_000
    update_freq: int = 1
    target_update_interval: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 10_000
    hidden_sizes: Tuple[int, ...] = (256, 256)


class DQNAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: DQNConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.action_dim = action_dim
        self.q_net = MLP(obs_dim, action_dim, cfg.hidden_sizes).to(device)
        self.target_net = MLP(obs_dim, action_dim, cfg.hidden_sizes).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        if dist.is_available() and dist.is_initialized() and device.type == "cuda":
            self.q_net = nn.parallel.DistributedDataParallel(self.q_net, device_ids=[device])
        self.optim = optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.buffer_size, obs_dim)
        self.steps = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.steps / float(self.cfg.epsilon_decay))
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def act(self, obs: Any, explore: bool = True) -> int:
        obs_vec = flatten_observation(obs)
        if explore and np.random.rand() < self.epsilon():
            action = np.random.randint(0, self.action_dim)
            return int(action)
        with torch.no_grad():
            obs_t = to_tensor(obs_vec, self.device).unsqueeze(0)
            q_vals = self.q_net(obs_t)
            action = int(torch.argmax(q_vals, dim=1).item())
        return action

    def add(self, obs, action, reward, next_obs, done):
        self.replay.add(flatten_observation(obs), action, reward, flatten_observation(next_obs), done)
        self.steps += 1

    def update(self):
        if not self.replay.ready(self.cfg.warmup):
            return {}
        if self.steps % self.cfg.update_freq != 0:
            return {}

        batch = self.replay.sample(self.cfg.batch_size)
        obs = to_tensor(batch["obs"], self.device)
        actions = to_tensor(batch["actions"], self.device).long()
        rewards = to_tensor(batch["rewards"], self.device)
        next_obs = to_tensor(batch["next_obs"], self.device)
        dones = to_tensor(batch["dones"], self.device)

        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_obs).max(dim=1)[0]
            target = rewards + self.cfg.gamma * (1 - dones) * next_q
        loss = nn.functional.mse_loss(q_values, target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.steps % self.cfg.target_update_interval == 0:
            self.target_net.load_state_dict(
                self.q_net.module.state_dict() if isinstance(self.q_net, nn.parallel.DistributedDataParallel) else self.q_net.state_dict()
            )
        return {"loss": loss.item()}

    def state_dict(self):
        net = self.q_net.module if isinstance(self.q_net, nn.parallel.DistributedDataParallel) else self.q_net
        return {
            "q_net": net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optim": self.optim.state_dict(),
            "steps": self.steps,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        net = self.q_net.module if isinstance(self.q_net, nn.parallel.DistributedDataParallel) else self.q_net
        net.load_state_dict(state["q_net"])
        self.target_net.load_state_dict(state["target_net"])
        self.optim.load_state_dict(state["optim"])
        self.steps = state.get("steps", 0)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lr: float = 3e-4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    rollout_length: int = 512
    ppo_epochs: int = 10
    batch_size: int = 64
    gae_lambda: float = 0.95
    hidden_sizes: Tuple[int, ...] = (256, 256)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, discrete: bool, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        self.discrete = discrete
        self.body = MLP(obs_dim, hidden_sizes[-1], hidden_sizes[:-1] or (256,))
        if discrete:
            self.policy_head = nn.Linear(hidden_sizes[-1], action_dim)
            self.log_std = None
        else:
            self.policy_head = nn.Linear(hidden_sizes[-1], action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value


class PPOAgent:
    def __init__(self, obs_dim: int, action_space, cfg: PPOConfig, device: torch.device, discrete: bool):
        self.cfg = cfg
        self.device = device
        self.discrete = discrete
        self.action_dim = action_space.n if discrete else action_space.shape[0]
        self.ac = ActorCritic(obs_dim, self.action_dim, discrete, cfg.hidden_sizes).to(device)
        if dist.is_available() and dist.is_initialized() and device.type == "cuda":
            self.ac = nn.parallel.DistributedDataParallel(self.ac, device_ids=[device])
        self.optim = optim.Adam(self.ac.parameters(), lr=cfg.lr)
        self.buffer: Dict[str, List[np.ndarray]] = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "logprobs": [],
            "values": [],
        }

    def act(self, obs: Any, explore: bool = True):
        obs_vec = flatten_observation(obs)
        obs_t = to_tensor(obs_vec, self.device).unsqueeze(0)
        logits, value = self.ac(obs_t)
        if self.discrete:
            probs = torch.distributions.Categorical(logits=logits)
            action = probs.sample()
            logprob = probs.log_prob(action)
            action_np = int(action.item())
        else:
            std = torch.exp(self.ac.module.log_std if isinstance(self.ac, nn.parallel.DistributedDataParallel) else self.ac.log_std)
            dist_norm = torch.distributions.Normal(logits, std)
            action = dist_norm.sample()
            logprob = dist_norm.log_prob(action).sum(-1)
            action_np = action.squeeze(0).cpu().numpy()
        if not explore:
            if self.discrete:
                action_np = int(torch.argmax(logits, dim=-1).item())
                probs = torch.distributions.Categorical(logits=logits)
                logprob = probs.log_prob(torch.tensor(action_np, device=self.device))
            else:
                action_np = logits.detach().squeeze(0).cpu().numpy()
                std = torch.exp(
                    self.ac.module.log_std if isinstance(self.ac, nn.parallel.DistributedDataParallel) else self.ac.log_std
                )
                dist_norm = torch.distributions.Normal(logits, std)
                logprob = dist_norm.log_prob(torch.as_tensor(action_np, device=self.device)).sum(-1)
        return action_np, logprob.detach().cpu().numpy(), value.detach().cpu().numpy()

    def store(self, obs, action, reward, done, logprob, value):
        self.buffer["obs"].append(flatten_observation(obs))
        self.buffer["actions"].append(action)
        self.buffer["rewards"].append(reward)
        self.buffer["dones"].append(done)
        self.buffer["logprobs"].append(logprob)
        self.buffer["values"].append(value)

    def _compute_advantages(self, next_value: float):
        rewards = np.array(self.buffer["rewards"], dtype=np.float32)
        dones = np.array(self.buffer["dones"], dtype=np.float32)
        values = np.array(self.buffer["values"], dtype=np.float32).squeeze(-1)
        adv = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.cfg.gamma * (1 - dones[t]) * next_val - values[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1 - dones[t]) * last_gae
            adv[t] = last_gae
        returns = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    def ready(self):
        return len(self.buffer["obs"]) >= self.cfg.rollout_length

    def update(self, next_value: float):
        adv, returns = self._compute_advantages(next_value)
        obs = torch.as_tensor(np.array(self.buffer["obs"], dtype=np.float32), device=self.device)
        actions = torch.as_tensor(np.array(self.buffer["actions"]), device=self.device)
        old_logprobs = torch.as_tensor(np.array(self.buffer["logprobs"], dtype=np.float32), device=self.device).squeeze()
        returns_t = torch.as_tensor(returns, device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device)

        dataset_size = len(obs)
        idxs = np.arange(dataset_size)
        metrics = {}
        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, self.cfg.batch_size):
                end = start + self.cfg.batch_size
                batch_idx = idxs[start:end]
                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_returns = returns_t[batch_idx]
                batch_adv = adv_t[batch_idx]
                batch_old_logp = old_logprobs[batch_idx]

                logits, values = self.ac(batch_obs)
                if self.discrete:
                    dist_policy = torch.distributions.Categorical(logits=logits)
                    logp = dist_policy.log_prob(batch_actions)
                    entropy = dist_policy.entropy().mean()
                else:
                    std = torch.exp(self.ac.module.log_std if isinstance(self.ac, nn.parallel.DistributedDataParallel) else self.ac.log_std)
                    dist_policy = torch.distributions.Normal(logits, std)
                    logp = dist_policy.log_prob(batch_actions).sum(-1)
                    entropy = dist_policy.entropy().sum(-1).mean()
                ratio = torch.exp(logp - batch_old_logp)
                clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * batch_adv
                policy_loss = -(torch.min(ratio * batch_adv, clipped)).mean()
                value_loss = nn.functional.mse_loss(values.squeeze(-1), batch_returns)
                loss = policy_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                metrics = {"policy_loss": float(policy_loss.item()), "value_loss": float(value_loss.item()), "entropy": float(entropy.item())}
        self.buffer = {k: [] for k in self.buffer}
        return metrics

    def state_dict(self):
        net = self.ac.module if isinstance(self.ac, nn.parallel.DistributedDataParallel) else self.ac
        return {
            "ac": net.state_dict(),
            "optim": self.optim.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]):
        net = self.ac.module if isinstance(self.ac, nn.parallel.DistributedDataParallel) else self.ac
        net.load_state_dict(state["ac"])
        self.optim.load_state_dict(state["optim"])
