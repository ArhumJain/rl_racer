"""PPO whose rollouts and updates stay on one torch device, for ``BatchRacingEnv``.

It follows Stable-Baselines3's PPO: the same constructor arguments, network
(separate policy and value MLPs, orthogonal init, a state-independent log std),
unsquashed Gaussian actions clipped only when sent to the env, GAE with
bootstrapping on truncation, per-minibatch advantage normalisation, the clipped
surrogate loss, Adam with eps 1e-5 and gradient-norm clipping. It also exposes what
SB3 callbacks use (``num_timesteps``, ``logger``, ``get_env``, ``save``), so
``CheckpointCallback`` and custom ``BaseCallback`` subclasses work unchanged, and a
``predict`` that takes NumPy observations like SB3's, for ``time_trial``.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import configure_logger
from torch import nn

from gpu_racer.env import BatchRacingEnv


def _mlp(sizes: list[int], activation: type[nn.Module]) -> list[nn.Module]:
    layers = []
    for size_in, size_out in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(size_in, size_out), activation()]
    return layers


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_size: int, net_arch, activation_fn, log_std_init: float):
        super().__init__()
        if isinstance(net_arch, list):
            net_arch = {"pi": net_arch, "vf": net_arch}
        self.policy_net = nn.Sequential(*_mlp([observation_size, *net_arch["pi"]], activation_fn))
        self.value_net = nn.Sequential(*_mlp([observation_size, *net_arch["vf"]], activation_fn))
        self.action_head = nn.Linear(([observation_size, *net_arch["pi"]])[-1], action_size)
        self.value_head = nn.Linear(([observation_size, *net_arch["vf"]])[-1], 1)
        self.log_std = nn.Parameter(torch.full((action_size,), float(log_std_init)))
        # SB3's orthogonal initialisation
        for module, gain in [(self.policy_net, math.sqrt(2)), (self.value_net, math.sqrt(2)),
                             (self.action_head, 0.01), (self.value_head, 1.0)]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=gain)
                    nn.init.zeros_(layer.bias)

    def mean(self, observation: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.policy_net(observation))

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.value_net(observation))[..., 0]

    def distribution(self, observation: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean(observation)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))


class PPO:
    def __init__(
        self,
        policy: str,
        env: BatchRacingEnv | None,
        learning_rate: float | Callable[[float], float] = 3e-4,
        n_steps: int = 32,
        batch_size: int = 16384,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: str = "cuda",
    ) -> None:
        if policy != "MlpPolicy":
            raise ValueError("only 'MlpPolicy' is supported")
        self.kwargs = {
            "learning_rate": learning_rate, "n_steps": n_steps, "batch_size": batch_size, "n_epochs": n_epochs,
            "gamma": gamma, "gae_lambda": gae_lambda, "clip_range": clip_range,
            "normalize_advantage": normalize_advantage, "ent_coef": ent_coef, "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm, "target_kl": target_kl, "stats_window_size": stats_window_size,
            "tensorboard_log": tensorboard_log, "policy_kwargs": policy_kwargs, "verbose": verbose,
        }
        for name, value in self.kwargs.items():
            setattr(self, name, value)
        if seed is not None:
            torch.manual_seed(seed)
        self.device = torch.device(device)
        self.env = env
        policy_kwargs = dict(policy_kwargs or {})
        self.policy = ActorCritic(
            observation_size=19,
            action_size=2,
            net_arch=policy_kwargs.get("net_arch", {"pi": [64, 64], "vf": [64, 64]}),
            activation_fn=policy_kwargs.get("activation_fn", nn.Tanh),
            log_std_init=policy_kwargs.get("log_std_init", 0.0),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate(1.0), eps=1e-5)
        self.num_timesteps = 0
        self.ep_info_buffer: deque = deque(maxlen=stats_window_size)
        self._logger = None

    @property
    def logger(self):
        return self._logger

    def get_env(self) -> BatchRacingEnv | None:
        return self.env

    def get_vec_normalize_env(self) -> None:
        return None

    def _learning_rate(self, progress_remaining: float) -> float:
        rate = self.learning_rate
        return rate(progress_remaining) if callable(rate) else rate

    def learn(self, total_timesteps: int, callback: BaseCallback | list | None = None,
              log_interval: int = 1, tb_log_name: str = "PPO", reset_num_timesteps: bool = True) -> PPO:
        env, n = self.env, self.env.num_envs
        if reset_num_timesteps:
            self.num_timesteps = 0
        self._logger = configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)
        if isinstance(callback, list):
            callback = CallbackList(callback)
        callback = callback or CallbackList([])
        callback.init_callback(self)
        callback.on_training_start(locals(), globals())

        shape = (self.n_steps, n)
        buffer = {name: torch.zeros(*shape, *extra, device=self.device) for name, extra in
                  [("observations", (19,)), ("actions", (2,)), ("log_probs", ()), ("rewards", ()),
                   ("values", ()), ("episode_starts", ())]}
        observation = env.reset().to(self.device)
        episode_start = torch.ones(n, device=self.device)
        episode_return = torch.zeros(n, device=self.device)
        episode_length = torch.zeros(n, device=self.device)
        began, start_timesteps, iteration = time.time(), self.num_timesteps, 0
        target = self.num_timesteps + total_timesteps if not reset_num_timesteps else total_timesteps

        while self.num_timesteps < target:
            callback.on_rollout_start()
            finished_returns, finished_lengths = [], []
            for k in range(self.n_steps):
                with torch.no_grad():
                    distribution = self.policy.distribution(observation)
                    action = distribution.sample()
                    log_prob = distribution.log_prob(action).sum(-1)
                    value = self.policy.value(observation)
                next_observation, reward, terminated, truncated, info = env.step(action.clamp(-1.0, 1.0))
                reward = reward.to(torch.float32)
                with torch.no_grad():
                    # A time limit is not a real ending: bootstrap from where the car was
                    bootstrap = self.policy.value(info["final_observation"].to(self.device))
                done = terminated | truncated
                episode_return += reward
                episode_length += 1
                if done.any():
                    finished_returns.append(episode_return[done].clone())
                    finished_lengths.append(episode_length[done].clone())
                    episode_return[done] = 0
                    episode_length[done] = 0
                for name, value_k in [("observations", observation), ("actions", action), ("log_probs", log_prob),
                                      ("rewards", reward + self.gamma * bootstrap * (truncated & ~terminated)),
                                      ("values", value), ("episode_starts", episode_start)]:
                    buffer[name][k] = value_k
                observation, episode_start = next_observation, done.to(torch.float32)
                self.num_timesteps += n
                callback.update_locals(locals())
                if not callback.on_step():
                    callback.on_training_end()
                    return self

            with torch.no_grad():
                last_value = self.policy.value(observation)
            advantages = self._gae(buffer, last_value, episode_start)
            returns = advantages + buffer["values"]
            if finished_returns:
                for r, length in zip(torch.cat(finished_returns).tolist(), torch.cat(finished_lengths).tolist()):
                    self.ep_info_buffer.append({"r": r, "l": length})
            callback.on_rollout_end()

            iteration += 1
            progress_remaining = 1.0 - (self.num_timesteps - start_timesteps) / max(total_timesteps, 1)
            self._train(buffer, advantages, returns, progress_remaining)
            if log_interval and iteration % log_interval == 0:
                elapsed = max(time.time() - began, 1e-9)
                if self.ep_info_buffer:
                    self.logger.record("rollout/ep_rew_mean", float(np.mean([e["r"] for e in self.ep_info_buffer])))
                    self.logger.record("rollout/ep_len_mean", float(np.mean([e["l"] for e in self.ep_info_buffer])))
                self.logger.record("time/iterations", iteration)
                self.logger.record("time/fps", int((self.num_timesteps - start_timesteps) / elapsed))
                self.logger.record("time/time_elapsed", int(elapsed))
                self.logger.record("time/total_timesteps", self.num_timesteps)
                self.logger.dump(step=self.num_timesteps)

        callback.on_training_end()
        return self

    def _gae(self, buffer: dict, last_value: torch.Tensor, last_episode_start: torch.Tensor) -> torch.Tensor:
        advantages = torch.zeros_like(buffer["rewards"])
        running = torch.zeros_like(last_value)
        for k in reversed(range(self.n_steps)):
            if k == self.n_steps - 1:
                not_done, next_value = 1.0 - last_episode_start, last_value
            else:
                not_done, next_value = 1.0 - buffer["episode_starts"][k + 1], buffer["values"][k + 1]
            delta = buffer["rewards"][k] + self.gamma * next_value * not_done - buffer["values"][k]
            running = delta + self.gamma * self.gae_lambda * not_done * running
            advantages[k] = running
        return advantages

    def _train(self, buffer: dict, advantages: torch.Tensor, returns: torch.Tensor, progress_remaining: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = self._learning_rate(progress_remaining)
        flat = lambda tensor: tensor.reshape(-1, *tensor.shape[2:])
        observations, actions = flat(buffer["observations"]), flat(buffer["actions"])
        old_log_probs, old_values = flat(buffer["log_probs"]), flat(buffer["values"])
        advantages, returns = flat(advantages), flat(returns)
        size = observations.shape[0]
        stats = {"policy_gradient_loss": [], "value_loss": [], "entropy_loss": [], "clip_fraction": [], "approx_kl": []}
        continue_training = True
        for _ in range(self.n_epochs):
            for indices in torch.randperm(size, device=self.device).split(self.batch_size):
                distribution = self.policy.distribution(observations[indices])
                log_prob = distribution.log_prob(actions[indices]).sum(-1)
                entropy = distribution.entropy().sum(-1)
                values = self.policy.value(observations[indices])
                advantage = advantages[indices]
                if self.normalize_advantage and len(indices) > 1:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                log_ratio = log_prob - old_log_probs[indices]
                ratio = log_ratio.exp()
                policy_loss = -torch.min(advantage * ratio,
                                         advantage * ratio.clamp(1 - self.clip_range, 1 + self.clip_range)).mean()
                value_loss = nn.functional.mse_loss(returns[indices], values)
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                stats["policy_gradient_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy_loss"].append(entropy_loss.item())
                stats["clip_fraction"].append(((ratio - 1).abs() > self.clip_range).float().mean().item())
                stats["approx_kl"].append(approx_kl)
                if self.target_kl is not None and approx_kl > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
            if not continue_training:
                break
        variance = returns.var()
        explained = 1 - (returns - old_values).var() / variance if variance > 0 else torch.tensor(float("nan"))
        for name, values in stats.items():
            self.logger.record(f"train/{name}", float(np.mean(values)))
        self.logger.record("train/explained_variance", float(explained))
        self.logger.record("train/std", self.policy.log_std.detach().exp().mean().item())
        self.logger.record("train/learning_rate", self._learning_rate(progress_remaining))

    def predict(self, observation, state=None, episode_start=None, deterministic: bool = False):
        """SB3's ``predict``: NumPy observation(s) in, NumPy action(s) in [-1, 1] out."""
        single = np.ndim(observation) == 1
        tensor = torch.as_tensor(np.atleast_2d(observation), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.policy.mean(tensor) if deterministic else self.policy.distribution(tensor).sample()
        action = action.clamp(-1.0, 1.0).cpu().numpy()
        return (action[0] if single else action), None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {k: v for k, v in self.kwargs.items() if not callable(v)}
        torch.save({"kwargs": kwargs, "policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "num_timesteps": self.num_timesteps}, path)

    @classmethod
    def load(cls, path: str | Path, env: BatchRacingEnv | None = None, device: str = "cpu") -> PPO:
        path = Path(path)
        if not path.exists() and not path.suffix:
            path = path.with_suffix(".pt")
        saved = torch.load(path, map_location=device, weights_only=False)
        model = cls("MlpPolicy", env, device=device, **saved["kwargs"])
        model.policy.load_state_dict(saved["policy"])
        model.optimizer.load_state_dict(saved["optimizer"])
        model.num_timesteps = saved["num_timesteps"]
        return model
