"""Soft actor-critic whose env, replay buffer and updates all stay on one torch device, for ``BatchRacingEnv``.

No installed package fits: Stable-Baselines3, sb3-contrib and sbx all step envs and keep the
replay buffer in NumPy on the CPU. This follows SB3's SAC (tanh-squashed Gaussian actor, twin
critics with target networks, automatic entropy tuning) with what massively parallel envs need:
many envs per step, large batches, and LayerNorm in the critics for stability. It exposes what
SB3 callbacks use (``num_timesteps``, ``logger``, ``get_env``, ``save``) and an SB3-like
``predict`` that takes NumPy observations, for ``time_trial``.
"""

from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import configure_logger
from torch import nn

OBSERVATION_SIZE, ACTION_SIZE = 19, 2
LOG_STD_RANGE = (-5.0, 2.0)


def _mlp(sizes: list[int], layer_norm: bool) -> nn.Sequential:
    layers = []
    for size_in, size_out in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(size_in, size_out))
        if layer_norm:
            layers.append(nn.LayerNorm(size_out))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, net_arch: list[int]):
        super().__init__()
        self.body = _mlp([OBSERVATION_SIZE, *net_arch], layer_norm=False)
        self.mean = nn.Linear(net_arch[-1], ACTION_SIZE)
        self.log_std = nn.Linear(net_arch[-1], ACTION_SIZE)

    def forward(self, observation: torch.Tensor, deterministic: bool = False):
        features = self.body(observation)
        mean = self.mean(features)
        if deterministic:
            return torch.tanh(mean), None
        log_std = self.log_std(features).clamp(*LOG_STD_RANGE)
        noise = torch.randn_like(mean)
        raw = mean + log_std.exp() * noise
        action = torch.tanh(raw)
        # Gaussian log-density, corrected for the tanh squashing
        log_prob = (-0.5 * noise**2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        log_prob = log_prob - (2 * (math.log(2) - raw - nn.functional.softplus(-2 * raw))).sum(-1)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, net_arch: list[int]):
        super().__init__()
        self.q1 = nn.Sequential(_mlp([OBSERVATION_SIZE + ACTION_SIZE, *net_arch], layer_norm=True),
                                nn.Linear(net_arch[-1], 1))
        self.q2 = nn.Sequential(_mlp([OBSERVATION_SIZE + ACTION_SIZE, *net_arch], layer_norm=True),
                                nn.Linear(net_arch[-1], 1))

    def forward(self, observation: torch.Tensor, action: torch.Tensor):
        both = torch.cat([observation, action], -1)
        return self.q1(both)[..., 0], self.q2(both)[..., 0]


class ReplayBuffer:
    def __init__(self, capacity: int, device: torch.device):
        self.capacity, self.size, self.position = capacity, 0, 0
        zeros = lambda *shape: torch.zeros(capacity, *shape, device=device)
        self.observations, self.next_observations = zeros(OBSERVATION_SIZE), zeros(OBSERVATION_SIZE)
        self.actions, self.rewards, self.terminals = zeros(ACTION_SIZE), zeros(), zeros()

    def add(self, observation, action, reward, next_observation, terminal) -> None:
        n = observation.shape[0]
        index = (self.position + torch.arange(n, device=observation.device)) % self.capacity
        self.observations[index], self.actions[index], self.rewards[index] = observation, action, reward
        self.next_observations[index], self.terminals[index] = next_observation, terminal
        self.position = (self.position + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int):
        index = torch.randint(0, self.size, (batch_size,), device=self.observations.device)
        return (self.observations[index], self.actions[index], self.rewards[index],
                self.next_observations[index], self.terminals[index])


class SAC:
    def __init__(
        self,
        policy: str,
        env,
        learning_rate: float = 3e-4,
        buffer_size: int = 4_000_000,
        batch_size: int = 8192,
        gamma: float = 0.99,
        tau: float = 0.005,
        gradient_steps: int = 2,
        learning_starts: int = 0,
        ent_coef: str | float = "auto",
        ent_coef_init: float = 0.05,
        target_entropy: float = -2.0,
        policy_kwargs: dict | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: str = "cuda",
    ) -> None:
        if policy != "MlpPolicy":
            raise ValueError("only 'MlpPolicy' is supported")
        self.kwargs = {
            "learning_rate": learning_rate, "buffer_size": buffer_size, "batch_size": batch_size, "gamma": gamma,
            "tau": tau, "gradient_steps": gradient_steps, "learning_starts": learning_starts, "ent_coef": ent_coef,
            "ent_coef_init": ent_coef_init, "target_entropy": target_entropy, "policy_kwargs": policy_kwargs,
            "stats_window_size": stats_window_size, "tensorboard_log": tensorboard_log, "verbose": verbose,
        }
        for name, value in self.kwargs.items():
            setattr(self, name, value)
        if seed is not None:
            torch.manual_seed(seed)
        self.device = torch.device(device)
        self.env = env
        arch = dict((policy_kwargs or {}).get("net_arch", {"pi": [256, 256], "qf": [512, 512, 512]}))
        self.actor = Actor(arch["pi"]).to(self.device)
        self.critic = Critic(arch["qf"]).to(self.device)
        self.critic_target = Critic(arch["qf"]).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)
        # Plain Adam: the fused variant does not survive torch.compile of the update
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.learn_alpha = ent_coef == "auto"
        initial = ent_coef_init if self.learn_alpha else float(ent_coef)
        self.log_alpha = torch.tensor(math.log(initial), device=self.device, requires_grad=self.learn_alpha)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate) if self.learn_alpha else None
        self.buffer = None
        self.num_timesteps = 0
        self.ep_info_buffer: deque = deque(maxlen=stats_window_size)
        self._logger = None

    # The attributes SB3 callbacks rely on
    @property
    def logger(self):
        return self._logger

    @property
    def policy(self) -> nn.Module:
        return self.actor

    def get_env(self):
        return self.env

    def get_vec_normalize_env(self) -> None:
        return None

    def _update(self) -> dict:
        observation, action, reward, next_observation, terminal = self.buffer.sample(self.batch_size)
        alpha = self.log_alpha.detach().exp()
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_observation)
            q1_next, q2_next = self.critic_target(next_observation, next_action)
            target = reward + self.gamma * (1 - terminal) * (torch.minimum(q1_next, q2_next) - alpha * next_log_prob)
        q1, q2 = self.critic(observation, action)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_prob = self.actor(observation)
        q1_new, q2_new = self.critic(observation, new_action)
        actor_loss = (alpha * log_prob - torch.minimum(q1_new, q2_new)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learn_alpha:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_parameter.lerp_(parameter, self.tau)
        return {"critic_loss": critic_loss, "actor_loss": actor_loss, "log_prob": log_prob.mean(), "q": q1.mean()}

    def learn(self, total_timesteps: int, callback=None, log_interval_seconds: float = 30.0,
              tb_log_name: str = "SAC", reset_num_timesteps: bool = True) -> SAC:
        env, n = self.env, self.env.num_envs
        if reset_num_timesteps:
            self.num_timesteps = 0
        if self.buffer is None:
            self.buffer = ReplayBuffer(self.buffer_size, self.device)
        self._logger = configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)
        callback = CallbackList(callback) if isinstance(callback, list) else (callback or CallbackList([]))
        callback.init_callback(self)
        callback.on_training_start(locals(), globals())
        observation = env.reset().to(self.device)
        episode_return = torch.zeros(n, device=self.device)
        episode_length = torch.zeros(n, device=self.device)
        began, start_timesteps, next_log, updates = time.time(), self.num_timesteps, time.time() + log_interval_seconds, 0
        target = total_timesteps + (0 if reset_num_timesteps else self.num_timesteps)
        stats = None
        callback.on_rollout_start()
        while self.num_timesteps < target:
            with torch.no_grad():
                action, _ = self.actor(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            reward = reward.to(torch.float32)
            done = terminated | truncated
            # A time or lap limit is not a real ending: store where the car really was and keep bootstrapping
            real_next = torch.where(done[:, None], info["final_observation"], next_observation)
            self.buffer.add(observation, action, reward, real_next, terminated.to(torch.float32))
            episode_return += reward
            episode_length += 1
            if done.any():
                for r, length in zip(episode_return[done].tolist(), episode_length[done].tolist()):
                    self.ep_info_buffer.append({"r": r, "l": length})
                episode_return[done] = 0
                episode_length[done] = 0
            observation = next_observation
            self.num_timesteps += n
            callback.update_locals(locals())
            if not callback.on_step():
                break
            if self.buffer.size >= self.learning_starts:
                for _ in range(self.gradient_steps):
                    stats = self._update()
                    updates += 1
            if time.time() >= next_log and stats is not None:
                next_log = time.time() + log_interval_seconds
                elapsed = max(time.time() - began, 1e-9)
                if self.ep_info_buffer:
                    self.logger.record("rollout/ep_rew_mean", float(np.mean([e["r"] for e in self.ep_info_buffer])))
                    self.logger.record("rollout/ep_len_mean", float(np.mean([e["l"] for e in self.ep_info_buffer])))
                for name, value in stats.items():
                    self.logger.record(f"train/{name}", float(value.detach()))
                self.logger.record("train/ent_coef", float(self.log_alpha.exp()))
                self.logger.record("train/n_updates", updates)
                self.logger.record("time/fps", int((self.num_timesteps - start_timesteps) / elapsed))
                self.logger.record("time/total_timesteps", self.num_timesteps)
                self.logger.dump(step=self.num_timesteps)
        callback.on_rollout_end()
        callback.on_training_end()
        return self

    def predict(self, observation, state=None, episode_start=None, deterministic: bool = False):
        """SB3's ``predict``: NumPy observation(s) in, NumPy action(s) in [-1, 1] out."""
        single = np.ndim(observation) == 1
        tensor = torch.as_tensor(np.atleast_2d(observation), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _ = self.actor(tensor, deterministic=deterministic)
        action = action.cpu().numpy()
        return (action[0] if single else action), None

    def save(self, path: str | Path) -> None:
        """Saves the weights and settings; the replay buffer is not saved."""
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kwargs": self.kwargs, "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "critic_target": self.critic_target.state_dict(), "log_alpha": self.log_alpha.detach().cpu(),
                    "num_timesteps": self.num_timesteps}, path)

    @classmethod
    def load(cls, path: str | Path, env=None, device: str = "cpu") -> SAC:
        path = Path(path)
        if not path.exists() and not path.suffix:
            path = path.with_suffix(".pt")
        saved = torch.load(path, map_location=device, weights_only=False)
        model = cls("MlpPolicy", env, device=device, **saved["kwargs"])
        model.actor.load_state_dict(saved["actor"])
        model.critic.load_state_dict(saved["critic"])
        model.critic_target.load_state_dict(saved["critic_target"])
        model.log_alpha.data.copy_(saved["log_alpha"].to(device))
        model.num_timesteps = saved["num_timesteps"]
        return model
