"""PPO on the GPU racing env, and the variants tried on top of it: POEM, EPO, and evolution strategies (ES) on a
trained policy's last layers.

    uv run python train_ppo.py --name base --minutes 30 [--option value ...]

Runs, promotion and kept logs work as training.py describes, with the algorithm name "ppo".
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import configure_logger

import gpu_racer.env as gpu_env
from gpu_racer import PPO
from gpu_racer.ppo import ActorCritic
from racer_rl import Track
from racer_rl.observation import CURVATURE_SCALE, LOOKAHEAD
from training import (DYNAMICS, TRACK, FastRacingEnv, RaceAndKeepBest, add_environment_arguments,
                      make_training_env, promote, race, reached_target, score, start_run)

ALGORITHM = "ppo"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add("--name", required=True)
    add("--mode", default="ppo", choices=["ppo", "es"], help="es: evolution strategies on a trained policy's last layers")
    add("--minutes", type=float, default=30.0)
    add("--steps", type=float, default=None, help="stop after this many env steps (overrides --minutes)")
    add("--seed", type=int, default=0)
    add("--init", default=None, help="model.pt to start from (weights only)")
    add("--keep-as", default=None, help="also copy the run's best model here before its folder is deleted")
    add("--n-envs", type=int, default=8192)
    add_environment_arguments(add)
    # PPO
    add("--n-steps", type=int, default=32)
    add("--batch", type=int, default=32768)
    add("--epochs", type=int, default=10)
    add("--gamma", type=float, default=0.99)
    add("--lam", type=float, default=0.95)
    add("--lr", type=float, default=3e-4)
    add("--lr-end", type=float, default=None, help="linear decay to this rate over the run")
    add("--clip", type=float, default=0.2)
    add("--ent", type=float, default=0.0)
    add("--target-kl", type=float, default=None)
    add("--net", default="256,256")
    add("--log-std", type=float, default=-1.0)
    add("--features", type=int, default=0, help="1: give the network track position and widths ahead")
    add("--reset-log-std", type=float, default=None, help="after --init, set the exploration log std to this")
    add("--log-std-floor", type=float, default=None, help="never let the exploration log std fall below this")
    add("--std-reset", default=None, help="pedal,steering exploration std to set after --init (e.g. 0.04,0.015)")
    add("--std-range-pedal", default=None, help="min,max exploration std kept for the pedal throughout training")
    add("--std-range-steer", default=None, help="min,max exploration std kept for the steering throughout training")
    add("--eval-seconds", type=float, default=60.0)
    # POEM (PPO with evolutionary mutations, arXiv 2601.14705); values the paper leaves unspecified are flagged
    add("--poem", type=int, default=0, help="1: train with POEM instead of plain PPO")
    add("--poem-delta", type=float, default=0.02, help="mutate when KL(policy || moving average) < this (unspecified)")
    add("--poem-beta", type=float, default=0.9, help="moving-average coefficient of the policy weights (unspecified)")
    add("--poem-sigma-min", type=float, default=1e-3, help="smallest mutation noise std, absolute (unspecified)")
    add("--poem-sigma-max", type=float, default=1e-2, help="largest mutation noise std, absolute (unspecified)")
    add("--poem-lambda", type=float, default=0.05, help="weight of the diversity term lambda_div (unspecified)")
    add("--poem-mutate", default="actor", choices=["actor", "all"], help="mutate the actor only, or every policy-object parameter")
    # EPO (Evolutionary Policy Optimization, arXiv 2503.19037); defaults follow its locomotion table (Appendix B.2)
    add("--epo", type=int, default=0, help="1: train with EPO instead of PPO")
    add("--epo-agents", type=int, default=64, help="population size K (1 = PPO with EPO's settings, the control)")
    add("--epo-latent", type=int, default=32, help="size of each agent's latent gene (official code default 32)")
    add("--epo-sigma", type=float, default=0.25, help="std of the mutation noise on a latent (unspecified)")
    add("--epo-ga-gamma", type=float, default=0.5, help="gamma in Eq. 3: GA runs when max - min > gamma * median fitness")
    add("--epo-off-lambda", type=float, default=1.0, help="lambda in Eq. 5 and 10: weight of the master's off-policy losses")
    add("--epo-kl-threshold", type=float, default=0.016, help="adaptive learning rate: KL threshold (0: fixed rate)")
    add("--epo-ga-interval", type=float, default=20e6, help="env steps between GA steps (official code: 20M)")
    add("--epo-ga-warmup", type=float, default=200e6, help="env steps before the first GA step (official code: 200M)")
    add("--epo-code-match", type=int, default=0, help="1: follow the authors' code where it differs from the paper "
        "(master can be replaced by the GA; plain PPO clip on the off-policy data; no mutation noise)")
    # evolution strategies (--mode es, needs --init)
    add("--es-pop", type=int, default=4096, help="perturbed policies raced at once (even)")
    add("--es-sigma", type=float, default=0.02, help="perturbation size, relative to the layer's RMS weight")
    add("--es-lr", type=float, default=0.25, help="Adam step size, relative to the absolute perturbation size")
    add("--es-real-every", type=int, default=5, help="generations between real-simulator checks")
    add("--es-layers", type=int, default=1, choices=[1, 2, 3], help="how many of the last layers to evolve (3 = all)")
    return parser.parse_args()


WIDTHS_AHEAD = (0.0, 25.0, 50.0, 100.0, 150.0, 200.0)
HARMONICS = 4


class TrackFeatures(torch.nn.Module):
    """
    Extra network inputs computed from the observation alone, so the policy still maps obs -> action.

    The 12 curvature readings ahead are a fingerprint of where the car is on the lap: the
    nearest entry of a table built from the track gives its position to within about 2 m. From
    that position come Fourier features of the lap position and the half-width at a few
    distances ahead, which the observation does not contain.
    """

    def __init__(self):
        super().__init__()
        track = Track.from_name(TRACK)
        self.length = track.length
        grid = np.arange(0.0, track.length, 0.25)
        prints = (CURVATURE_SCALE * track.curvature(grid[:, None] + np.asarray(LOOKAHEAD, float))).astype(np.float32)
        # One entry per run of identical consecutive fingerprints, located at the run's middle
        starts = np.flatnonzero(np.r_[True, np.any(prints[1:] != prints[:-1], axis=1)])
        ends = np.r_[starts[1:], len(grid)]
        centres = (grid[starts] + grid[ends - 1]) / 2
        self.register_buffer("fingerprints", torch.as_tensor(prints[starts]))
        self.register_buffer("centres", torch.as_tensor(centres, dtype=torch.float32))
        self.register_buffer("widths", torch.as_tensor(track.half_widths, dtype=torch.float32))
        self.register_buffer("ahead", torch.as_tensor(WIDTHS_AHEAD, dtype=torch.float32))
        self.size = 19 + len(WIDTHS_AHEAD) + 2 * HARMONICS

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # Squared distances by a float64 matrix multiply: fast, and exact enough to separate fingerprints
        # that differ by 1e-6 (float32 or TF32 could pick the wrong near-identical neighbour)
        prints = self.fingerprints.double()
        print_norms = (prints**2).sum(1)
        nearest = torch.cat([
            ((readings**2).sum(1, keepdim=True) - 2 * readings @ prints.T + print_norms).argmin(1)
            for readings in observation[:, 7:19].double().split(16384)  # bounds the distance matrix's memory
        ])
        s = self.centres[nearest]
        at = torch.remainder(s[:, None] + self.ahead, self.length)
        low = at.floor().long() % len(self.widths)
        fraction = at - at.floor()
        widths = (1 - fraction) * self.widths[low] + fraction * self.widths[(low + 1) % len(self.widths)]
        angle = 2 * np.pi * s[:, None] / self.length * torch.arange(1, HARMONICS + 1, device=s.device)
        return torch.cat([observation, widths / 10.0, torch.sin(angle), torch.cos(angle)], -1)


class PositionAwareActorCritic(ActorCritic):
    """gpu_racer's ActorCritic with TrackFeatures in front of both networks."""

    def __init__(self, net_arch, log_std_init: float):
        features = TrackFeatures()
        super().__init__(features.size, 2, net_arch, torch.nn.Tanh, log_std_init)
        self.features = features

    def mean(self, observation: torch.Tensor) -> torch.Tensor:
        return super().mean(self.features(observation))

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        return super().value(self.features(observation))


class POEM(PPO):
    """
    PPO with evolutionary mutations (Czworkowski, Hornish and Yasin, arXiv 2601.14705), as its Section 3 describes.

    theta is every parameter of the policy object (actor, critic and log std), as Stable-Baselines3's
    ``policy.parameters()``, which the paper modified. Once per round (Algorithm 1, line 8):
        theta_bar <- beta theta_bar + (1 - beta) theta.
    On each minibatch:
        L_total(theta) = L_PPO(theta) - lambda D_KL(pi_theta || pi_theta_bar) + alpha_vf L_VF(theta) - alpha_ent H(pi_theta)
    with D_KL = mean_i log pi_theta(a_i|s_i) / pi_theta_bar(a_i|s_i) over the minibatch (Algorithm 1, line 9). After the
    gradient step, d_post = D_KL(pi_theta_new || pi_theta_bar); if d_post < delta, one candidate
    theta' = theta_new + N(0, sigma^2 I), sigma = sigma_min + (sigma_max - sigma_min)(delta - d_post)/delta clamped to
    [sigma_min, sigma_max], replaces theta_new if and only if L_total(theta') < L_total(theta_new).
    The paper leaves beta, delta, sigma_min, sigma_max and lambda unspecified, and does not say how theta_bar starts;
    it starts here as the initial policy.
    """

    poem_args = None

    def _losses(self, observations, actions, old_log_probs, advantage, returns):
        """L_total and its parts on one minibatch, for the current parameters."""
        distribution = self.policy.distribution(observations)
        log_prob = distribution.log_prob(actions).sum(-1)
        with torch.no_grad():
            average_log_prob = self.average.distribution(observations).log_prob(actions).sum(-1)
        ratio = (log_prob - old_log_probs).exp()
        ppo = -torch.min(advantage * ratio, advantage * ratio.clamp(1 - self.clip_range, 1 + self.clip_range)).mean()
        divergence = (log_prob - average_log_prob).mean()
        value = torch.nn.functional.mse_loss(returns, self.policy.value(observations))
        entropy = distribution.entropy().sum(-1).mean()
        total = ppo - self.poem_args.poem_lambda * divergence + self.vf_coef * value - self.ent_coef * entropy
        return total, ppo, divergence, value, entropy, ratio, log_prob

    def _train(self, buffer: dict, advantages: torch.Tensor, returns: torch.Tensor, progress_remaining: float) -> None:
        args = self.poem_args
        if not hasattr(self, "average"):
            self.average = copy.deepcopy(self.policy).requires_grad_(False)
            self.checks = self.mutations = self.accepted = 0
        with torch.no_grad():
            for average, current in zip(self.average.parameters(), self.policy.parameters()):
                average.mul_(args.poem_beta).add_(current, alpha=1 - args.poem_beta)
        for group in self.optimizer.param_groups:
            group["lr"] = self._learning_rate(progress_remaining)
        flat = lambda tensor: tensor.reshape(-1, *tensor.shape[2:])
        observations, actions = flat(buffer["observations"]), flat(buffer["actions"])
        old_log_probs, old_values = flat(buffer["log_probs"]), flat(buffer["values"])
        advantages, returns = flat(advantages), flat(returns)
        size = observations.shape[0]
        stats = {"policy_gradient_loss": [], "value_loss": [], "entropy_loss": [], "clip_fraction": [], "approx_kl": [],
                 "kl_to_average": []}
        parameters = list(self.policy.parameters())
        # Mutating the critic too makes nearly every candidate lose on L_VF, which was just minimized on this
        # minibatch; "actor" mutates only the policy network, action head and exploration std
        mutated = parameters if args.poem_mutate == "all" else (
            list(self.policy.policy_net.parameters()) + list(self.policy.action_head.parameters()) + [self.policy.log_std])
        for _ in range(self.n_epochs):
            for indices in torch.randperm(size, device=self.device).split(self.batch_size):
                advantage = advantages[indices]
                if self.normalize_advantage and len(indices) > 1:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                batch = (observations[indices], actions[indices], old_log_probs[indices], advantage, returns[indices])
                total, ppo, divergence, value, entropy, ratio, log_prob = self._losses(*batch)
                self.optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    total_new, _, d_post, *_ = self._losses(*batch)
                    self.checks += 1
                    if d_post < args.poem_delta:
                        sigma = args.poem_sigma_min + (args.poem_sigma_max - args.poem_sigma_min) * (args.poem_delta - d_post) / args.poem_delta
                        sigma = float(sigma.clamp(args.poem_sigma_min, args.poem_sigma_max))
                        noise = [sigma * torch.randn_like(parameter) for parameter in mutated]
                        for parameter, n in zip(mutated, noise):
                            parameter.add_(n)
                        total_candidate = self._losses(*batch)[0]
                        self.mutations += 1
                        if total_candidate < total_new:
                            self.accepted += 1
                        else:
                            for parameter, n in zip(mutated, noise):
                                parameter.sub_(n)
                    log_ratio = log_prob - old_log_probs[indices]
                    stats["policy_gradient_loss"].append(ppo.item())
                    stats["value_loss"].append(value.item())
                    stats["entropy_loss"].append(-entropy.item())
                    stats["clip_fraction"].append(((ratio - 1).abs() > self.clip_range).float().mean().item())
                    stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                    stats["kl_to_average"].append(float(d_post))
        variance = returns.var()
        explained = 1 - (returns - old_values).var() / variance if variance > 0 else torch.tensor(float("nan"))
        for name, values in stats.items():
            self.logger.record(f"train/{name}", float(np.mean(values)))
        self.logger.record("train/explained_variance", float(explained))
        self.logger.record("train/std", self.policy.log_std.detach().exp().mean().item())
        self.logger.record("poem/mutations", self.mutations)
        self.logger.record("poem/accepted", self.accepted)
        self.rounds = getattr(self, "rounds", 0) + 1
        if self.rounds % 20 == 0:
            print(f"[poem] round {self.rounds}: d_post mean {np.mean(stats['kl_to_average']):.5f} "
                  f"(min {np.min(stats['kl_to_average']):.5f}), mutations {self.mutations} of {self.checks} steps, "
                  f"accepted {self.accepted}, std {self.policy.log_std.exp().tolist()}", flush=True)


class EPOActorCritic(torch.nn.Module):
    """One actor-critic shared by K agents; agent k sees the observation together with its latent gene phi_k."""

    def __init__(self, agents: int, latent_size: int, hidden=(512, 256, 128), log_std_init: float = -1.0):
        super().__init__()
        self.latents = torch.nn.Parameter(torch.randn(agents, latent_size))

        def mlp(outputs: int) -> torch.nn.Sequential:
            layers, size = [], 19 + latent_size
            for width in hidden:
                layers += [torch.nn.Linear(size, width), torch.nn.ELU()]
                size = width
            return torch.nn.Sequential(*layers, torch.nn.Linear(size, outputs))

        self.actor, self.critic = mlp(2), mlp(1)
        self.log_std = torch.nn.Parameter(torch.full((2,), float(log_std_init)))
        # rl_games' normalize_input: running mean and variance of the observation, clipped to +-5 after scaling
        self.register_buffer("obs_mean", torch.zeros(19))
        self.register_buffer("obs_var", torch.ones(19))
        self.register_buffer("obs_count", torch.tensor(1e-4))

    def update_observation_statistics(self, observations: torch.Tensor) -> None:
        batch_mean, batch_var, batch_count = observations.mean(0), observations.var(0), observations.shape[0]
        delta, total = batch_mean - self.obs_mean, self.obs_count + batch_count
        self.obs_mean += delta * batch_count / total
        m2 = self.obs_var * self.obs_count + batch_var * batch_count + delta**2 * self.obs_count * batch_count / total
        self.obs_var.copy_(m2 / total)
        self.obs_count.fill_(float(total))

    def _inputs(self, observation, agent):
        normalized = ((observation - self.obs_mean) / torch.sqrt(self.obs_var + 1e-5)).clamp(-5.0, 5.0)
        return torch.cat([normalized, self.latents[agent]], -1)

    def mean(self, observation, agent):
        return self.actor(self._inputs(observation, agent))

    def value(self, observation, agent):
        return self.critic(self._inputs(observation, agent))[..., 0]

    def distribution(self, observation, agent):
        mean = self.mean(observation, agent)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))


class EPO:
    """
    Evolutionary Policy Optimization (Wang, Su, Gupta and Pathak, arXiv 2503.19037), Algorithm 1 and Section 4.

    K agents share one actor-critic network and differ only by a latent gene; env i belongs to agent i mod K.
    Agent 0 is the master. Every round: all agents collect ``horizon`` steps on their own envs; when the
    non-master fitness spread satisfies Eq. 3, the K - 2 fittest non-master agents are kept as elites and the
    remaining one is replaced by the average of two random elites' genes plus Gaussian noise. All parameters,
    genes included, then take PPO steps: every agent's own data with the clipped surrogate and 3-step critic
    targets (Eq. 2, 6, 7), plus, for the master only, an equal-sized sample of the other agents' data with the
    importance-weighted clip of Eq. 4 and 1-step critic targets (Eq. 8, 9), weighted by lambda (Eq. 5, 10).
    Hyperparameters follow the paper's locomotion table; latent size, latent init and mutation noise are not
    given in the paper. From rl_games, which the paper's code builds on: GAE (lambda 0.95) advantages, and running
    normalization of value targets, which its critic coefficient of 4.0 assumes (without it the raw returns'
    gradients dominate the shared gradient-norm clip and the actor barely moves).
    """

    def __init__(self, env, agents=64, latent_size=16, horizon=16, minibatch_factor=4, epochs=2, gamma=0.99,
                 gae_lambda=0.95, learning_rate=1e-4, kl_threshold=0.016, clip=0.1, critic_coef=4.0, grad_norm=1.0,
                 bounds_coef=1e-5, ga_gamma=0.5, off_lambda=1.0, mutation_sigma=0.25, log_std_init=-1.0,
                 minibatch_size=None, ga_interval=20e6, ga_warmup=200e6, code_match=False, tensorboard_log=None, seed=0,
                 device="cuda"):
        self.kwargs = dict(agents=agents, latent_size=latent_size, horizon=horizon, minibatch_factor=minibatch_factor,
                           epochs=epochs, gamma=gamma, gae_lambda=gae_lambda, learning_rate=learning_rate,
                           kl_threshold=kl_threshold, clip=clip, critic_coef=critic_coef, grad_norm=grad_norm,
                           bounds_coef=bounds_coef, ga_gamma=ga_gamma, off_lambda=off_lambda,
                           mutation_sigma=mutation_sigma, log_std_init=log_std_init, minibatch_size=minibatch_size,
                           ga_interval=ga_interval, ga_warmup=ga_warmup, code_match=code_match,
                           tensorboard_log=tensorboard_log)
        for name, value in self.kwargs.items():
            setattr(self, name, value)
        torch.manual_seed(seed)
        self.env, self.device = env, torch.device(device)
        self.policy = EPOActorCritic(agents, latent_size, log_std_init=log_std_init).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        self.lr = learning_rate
        self.num_timesteps, self.generations, self.rounds, self.master_replaced, self._logger = 0, 0, 0, 0, None
        self.value_mean = torch.zeros((), device=self.device)
        self.value_var = torch.ones((), device=self.device)
        self.value_count = 1e-4

    def _value(self, observation, agent):
        """Critic output in return units (the network itself predicts normalized values)."""
        return self.policy.value(observation, agent) * self.value_var.sqrt() + self.value_mean

    def _normalize(self, values):
        return (values - self.value_mean) / self.value_var.sqrt()

    def _update_value_statistics(self, targets) -> None:
        """rl_games' running mean and variance of the value targets."""
        batch_mean, batch_var, batch_count = targets.mean(), targets.var(), targets.numel()
        delta, total = batch_mean - self.value_mean, self.value_count + batch_count
        self.value_mean = self.value_mean + delta * batch_count / total
        m2 = self.value_var * self.value_count + batch_var * batch_count + delta**2 * self.value_count * batch_count / total
        self.value_var, self.value_count = m2 / total, total

    logger = property(lambda self: self._logger)
    get_env = lambda self: self.env
    get_vec_normalize_env = lambda self: None

    def predict(self, observation, state=None, episode_start=None, deterministic: bool = True):
        """The master agent's action for NumPy observation(s), as SB3's ``predict``."""
        single = np.ndim(observation) == 1
        tensor = torch.as_tensor(np.atleast_2d(observation), dtype=torch.float32, device=self.device)
        master = torch.zeros(len(tensor), dtype=torch.long, device=self.device)
        with torch.no_grad():
            action = self.policy.mean(tensor, master) if deterministic else self.policy.distribution(tensor, master).sample()
        action = action.clamp(-1.0, 1.0).cpu().numpy()
        return (action[0] if single else action), None

    def save(self, path) -> None:
        path = Path(path)
        path = path.with_suffix(".pt") if not path.suffix else path
        torch.save({"epo": True, "kwargs": self.kwargs, "policy": self.policy.state_dict(),
                    "value_statistics": (self.value_mean.item(), self.value_var.item(), self.value_count),
                    "num_timesteps": self.num_timesteps}, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "EPO":
        saved = torch.load(path, map_location=device, weights_only=False)
        model = cls(None, device=device, **saved["kwargs"])
        model.policy.load_state_dict(saved["policy"])
        model.num_timesteps = saved["num_timesteps"]
        mean, var, count = saved.get("value_statistics", (0.0, 1.0, 1e-4))
        model.value_mean.fill_(mean)
        model.value_var.fill_(var)
        model.value_count = count
        return model

    def _nstep_targets(self, rewards, dones, values, last_values, n=3):
        """Eq. 6: sum of the next n rewards plus gamma^n V_old, cut at episode ends and at the rollout's end."""
        horizon = rewards.shape[0]
        targets = torch.empty_like(rewards)
        for t in range(horizon):
            total, discount = torch.zeros_like(rewards[0]), torch.ones_like(rewards[0])
            alive = torch.ones_like(rewards[0])
            for m in range(n):
                if t + m >= horizon:
                    break
                total = total + alive * discount * rewards[t + m]
                alive = alive * (1 - dones[t + m])
                discount = discount * self.gamma
            following = values[t + n] if t + n < horizon else last_values
            targets[t] = total + alive * discount * following
        return targets

    def _ga_step(self, last_return: torch.Tensor, finished: torch.Tensor, agent: torch.Tensor) -> bool:
        """
        The genetic step as the authors' code runs it (isaacgymenvs/utils/rlgames_utils.py, EPOObserver): every
        ``ga_interval`` env steps after ``ga_warmup``, once every env has finished an episode since the last step,
        rank the non-master agents by the mean over their envs of each env's latest episode return; replace the
        worst agent's gene with the average of the two best (network_builder.merge_extra_params), plus the paper's
        Gaussian mutation noise (Section 4.1; the code adds none).
        """
        if self.agents < 4 or self.num_timesteps < self.ga_warmup or self.num_timesteps < self.next_ga:
            return False
        if not bool(finished.all()):
            return False
        scores = torch.zeros(self.agents, device=self.device).index_add_(0, agent, last_return)
        scores = scores / torch.bincount(agent, minlength=self.agents)
        # The paper ranks and replaces only non-master agents; the authors' code ranks every block, the leader's too
        ranked = [k for k in torch.argsort(scores, descending=True).tolist() if self.code_match or k != 0]
        best, worst = ranked[:2], ranked[-1]
        with torch.no_grad():
            child = 0.5 * (self.policy.latents[best[0]] + self.policy.latents[best[1]])
            noise = 0.0 if self.code_match else self.mutation_sigma * torch.randn_like(child)
            self.policy.latents[worst] = child + noise
            self.master_replaced += int(worst == 0)
            state = self.optimizer.state.get(self.policy.latents)
            if state:
                state["exp_avg"][worst].zero_()
                state["exp_avg_sq"][worst].zero_()
        finished.zero_()
        self.next_ga = self.num_timesteps + self.ga_interval
        self.generations += 1
        self.last_scores = scores
        return True

    def learn(self, total_timesteps: int, callback=None) -> "EPO":
        env, n, H = self.env, self.env.num_envs, self.horizon
        self._logger = configure_logger(0, self.tensorboard_log, "EPO", True)
        callback = CallbackList(callback) if isinstance(callback, list) else (callback or CallbackList([]))
        callback.init_callback(self)
        callback.on_training_start(locals(), globals())
        agent = torch.arange(n, device=self.device) % self.agents
        fitness = {k: deque(maxlen=32) for k in range(self.agents)}
        last_return = torch.zeros(n, device=self.device)
        finished = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.next_ga = self.ga_warmup
        episode_return = torch.zeros(n, device=self.device)
        observation = env.reset()
        zeros = lambda *shape: torch.zeros(H, n, *shape, device=self.device)
        while self.num_timesteps < total_timesteps:
            callback.on_rollout_start()
            obs_buffer, next_buffer, action_buffer = zeros(19), zeros(19), zeros(2)
            log_prob_buffer, reward_buffer, raw_reward_buffer, done_buffer, terminal_buffer, value_buffer = (
                zeros(), zeros(), zeros(), zeros(), zeros(), zeros())
            for t in range(H):
                with torch.no_grad():
                    distribution = self.policy.distribution(observation, agent)
                    action = distribution.sample()
                    log_prob = distribution.log_prob(action).sum(-1)
                    value = self._value(observation, agent)
                next_observation, reward, terminated, truncated, info = env.step(action.clamp(-1.0, 1.0))
                reward = reward.float()
                done = terminated | truncated
                real_next = torch.where(done[:, None], info["final_observation"], next_observation)
                with torch.no_grad():
                    bootstrap = self._value(info["final_observation"], agent)
                obs_buffer[t], next_buffer[t], action_buffer[t] = observation, real_next, action
                log_prob_buffer[t], value_buffer[t] = log_prob, value
                raw_reward_buffer[t], terminal_buffer[t], done_buffer[t] = reward, terminated.float(), done.float()
                # A time or lap limit is not a real ending: fold the value of where the car was into the reward
                reward_buffer[t] = reward + self.gamma * bootstrap * (truncated & ~terminated)
                episode_return += reward
                if done.any():
                    for k, r in zip(agent[done].tolist(), episode_return[done].tolist()):
                        fitness[k].append(r)
                    last_return = torch.where(done, episode_return, last_return)
                    finished |= done
                    episode_return[done] = 0
                observation = next_observation
                self.num_timesteps += n
                callback.update_locals(locals())
                if not callback.on_step():
                    callback.on_training_end()
                    return self
            callback.on_rollout_end()
            with torch.no_grad():
                last_values = self._value(observation, agent)
            advantages = torch.zeros_like(reward_buffer)
            running = torch.zeros(n, device=self.device)
            for t in reversed(range(H)):
                following = last_values if t == H - 1 else value_buffer[t + 1]
                not_done = 1 - done_buffer[t]
                delta = reward_buffer[t] + self.gamma * following * not_done - value_buffer[t]
                running = delta + self.gamma * self.gae_lambda * not_done * running
                advantages[t] = running
            targets = self._nstep_targets(reward_buffer, done_buffer, value_buffer, last_values)
            with torch.no_grad():
                self.policy.update_observation_statistics(obs_buffer.reshape(-1, 19))
            evolved = self._ga_step(last_return, finished, agent)
            self._update(obs_buffer, next_buffer, action_buffer, log_prob_buffer, advantages, targets,
                         raw_reward_buffer, terminal_buffer, agent)
            means = [float(np.mean(fitness[k])) for k in range(self.agents) if fitness[k]]
            self.logger.record("epo/master_fitness", float(np.mean(fitness[0])) if fitness[0] else float("nan"))
            self.logger.record("epo/population_fitness_mean", float(np.mean(means)) if means else float("nan"))
            self.logger.record("epo/generations", self.generations)
            self.logger.record("epo/evolved_this_round", int(evolved))
            self.logger.record("train/learning_rate", self.lr)
            self.logger.record("time/total_timesteps", self.num_timesteps)
            self.logger.dump(step=self.num_timesteps)
            self.rounds += 1
            if self.rounds % 50 == 0 and means:
                print(f"[epo] round {self.rounds}: master fitness {np.mean(fitness[0]) if fitness[0] else float('nan'):.1f}, "
                      f"population {np.mean(means):.1f} (min {min(means):.1f}, max {max(means):.1f}), "
                      f"generations {self.generations}, master replaced {self.master_replaced}, lr {self.lr:.2e}, "
                      f"std {self.policy.log_std.exp().tolist()}", flush=True)
        callback.on_training_end()
        return self

    def _update(self, observations, next_observations, actions, old_log_probs, advantages, targets, raw_rewards,
                terminals, agent) -> None:
        flat = lambda tensor: tensor.reshape(-1, *tensor.shape[2:])
        H, n = observations.shape[:2]
        agents_flat = agent.repeat(H)
        observations, next_observations, actions = flat(observations), flat(next_observations), flat(actions)
        old_log_probs, advantages, targets = flat(old_log_probs), flat(advantages), flat(targets)
        raw_rewards, terminals = flat(raw_rewards), flat(terminals)
        size = observations.shape[0]
        self._update_value_statistics(targets)
        with torch.no_grad():
            old_means = torch.cat([self.policy.mean(o, a) for o, a in zip(observations.split(65536), agents_flat.split(65536))])
            old_std = self.policy.log_std.exp().detach().clone()
        # S'_1: the master gets one randomly chosen other agent's whole batch, the same size as its own, as the
        # authors' code does with use_others_experience 'lf' and off_policy_ratio 1 (a2c_common.augment_batch_for_mixed_expl)
        off = None
        if self.agents > 1:
            follower = int(torch.randint(1, self.agents, (1,)))
            off = torch.nonzero(agents_flat == follower)[:, 0]
        if off is not None:
            master = torch.zeros(len(off), dtype=torch.long, device=self.device)
            with torch.no_grad():
                master_old_log_prob = self.policy.distribution(observations[off], master).log_prob(actions[off]).sum(-1)
                v_now = self._value(observations[off], master)
                v_next = self._value(next_observations[off], master)
                off_targets = raw_rewards[off] + self.gamma * (1 - terminals[off]) * v_next          # Eq. 8
                off_advantages = off_targets - v_now
                off_advantages = (off_advantages - off_advantages.mean()) / (off_advantages.std() + 1e-8)
        batch = self.minibatch_size or self.minibatch_factor * n
        for _ in range(self.epochs):
            kls = []
            order = torch.randperm(size, device=self.device)
            pieces = order.split(batch)
            off_pieces = torch.randperm(len(off), device=self.device).chunk(len(pieces)) if off is not None else [None] * len(pieces)
            for indices, off_indices in zip(pieces, off_pieces):
                a = agents_flat[indices]
                distribution = self.policy.distribution(observations[indices], a)
                log_prob = distribution.log_prob(actions[indices]).sum(-1)
                ratio = (log_prob - old_log_probs[indices]).exp()
                advantage = advantages[indices]
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                policy_loss = -torch.min(advantage * ratio, advantage * ratio.clamp(1 - self.clip, 1 + self.clip)).mean()
                critic_loss = (self.policy.value(observations[indices], a) - self._normalize(targets[indices])).pow(2).mean()   # Eq. 7
                mean = distribution.mean
                bounds_loss = (torch.relu(mean - 1.1) ** 2 + torch.relu(-1.1 - mean) ** 2).sum(-1).mean()
                loss = policy_loss + 0.5 * self.critic_coef * critic_loss + self.bounds_coef * bounds_loss
                if off_indices is not None and len(off_indices):
                    sample = off[off_indices]
                    master = torch.zeros(len(sample), dtype=torch.long, device=self.device)
                    behaviour = old_log_probs[sample]
                    master_log_prob = self.policy.distribution(observations[sample], master).log_prob(actions[sample]).sum(-1)
                    r = (master_log_prob - behaviour).exp()                                   # pi_1 / pi_k
                    mu = (master_old_log_prob[off_indices] - behaviour).exp()                 # pi_1,old / pi_k
                    adv = off_advantages[off_indices]
                    if self.code_match:  # the authors' code: plain PPO clip against the follower's log-probability
                        off_policy = -torch.min(r * adv, r.clamp(1 - self.clip, 1 + self.clip) * adv).mean()
                    else:
                        off_policy = -torch.min(r * adv, torch.clamp(r, mu * (1 - self.clip), mu * (1 + self.clip)) * adv).mean()  # Eq. 4
                    off_critic = (self.policy.value(observations[sample], master) - self._normalize(off_targets[off_indices])).pow(2).mean()  # Eq. 9
                    loss = loss + self.off_lambda * (off_policy + 0.5 * self.critic_coef * off_critic)            # Eq. 5, 10
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    std = self.policy.log_std.exp()
                    old_mean = old_means[indices]
                    kl = (torch.log(std / old_std) + (old_std**2 + (old_mean - mean.detach()) ** 2) / (2 * std**2) - 0.5).sum(-1).mean()
                    kls.append(float(kl))
            # rl_games' adaptive learning rate, with the paper's KL threshold
            kl = float(np.mean(kls))
            if self.kl_threshold <= 0:
                continue  # fixed learning rate
            if kl > 2 * self.kl_threshold:
                self.lr = max(self.lr / 1.5, 1e-6)
            elif kl < 0.5 * self.kl_threshold:
                self.lr = min(self.lr * 1.5, 1e-2)
            for group in self.optimizer.param_groups:
                group["lr"] = self.lr
        self.logger.record("train/kl", kl)
        self.logger.record("train/value_std", float(self.value_var.sqrt()))
        self.logger.record("train/std", self.policy.log_std.detach().exp().mean().item())


def build(policy_kwargs: dict, features: bool, env=None, device: str = "cpu", poem_args=None, **kwargs) -> PPO:
    """A gpu_racer PPO (or POEM when ``poem_args`` is given), with the position-aware network when ``features`` is set."""
    model = (POEM if poem_args is not None else PPO)("MlpPolicy", env, device=device, policy_kwargs=policy_kwargs, **kwargs)
    model.poem_args = poem_args
    if features:
        model.policy = PositionAwareActorCritic(policy_kwargs["net_arch"], policy_kwargs["log_std_init"]).to(device)
        model.optimizer = torch.optim.Adam(model.policy.parameters(), lr=model._learning_rate(1.0), eps=1e-5)
    return model


def load_policy(path) -> PPO:
    """Loads a model saved by this script on the CPU, ready for ``predict`` (also for the notebook's policy cell)."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("epo"):
        return EPO.load(path, device="cpu")
    features = any(key.startswith("features.") for key in saved["policy"])
    kwargs = {k: v for k, v in saved["kwargs"].items() if k != "policy_kwargs"}
    model = build(saved["kwargs"]["policy_kwargs"], features, **kwargs)
    model.policy.load_state_dict(saved["policy"])
    return model


class ExplorationBounds(BaseCallback):
    """
    Keeps the exploration std inside --log-std-floor and the per-action --std-range-pedal / --std-range-steer
    before every rollout. Steering tolerates far less noise than the pedal: a policy at the grip limit crashes
    every 3-lap race with steering noise 0.05 and survives 0.01 to 0.02.
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.floor = args.log_std_floor
        self.ranges = [(index, [float(x) for x in bounds.split(",")])
                       for index, bounds in ((0, args.std_range_pedal), (1, args.std_range_steer)) if bounds]

    def _on_rollout_start(self) -> None:
        with torch.no_grad():
            if self.floor is not None:
                self.model.policy.log_std.clamp_(min=self.floor)
            for index, (low, high) in self.ranges:
                self.model.policy.log_std[index].clamp_(np.log(low), np.log(high))

    def _on_step(self) -> bool:
        return True


def race_population(env, act, track_length):
    """Races every row's deterministic policy over a standing-start 3-lap trial; returns laps 2 + 3 or a crash score."""
    n = env.num_envs
    observation = env.reset()
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    lap_times = torch.full((n, 3), float("nan"), dtype=torch.float64, device=env.device)
    reached = torch.zeros(n, dtype=torch.float64, device=env.device)
    for _ in range(int(300 * gpu_env.POLICY_FREQUENCY)):
        with torch.no_grad():
            action = act(observation).clamp(-1.0, 1.0)
        observation, _, terminated, _, info = env.step(action)
        completed = alive & ~torch.isnan(info["lap_time"])
        lap_index = (info["laps"] - 1).clamp(0, 2).long()
        lap_times[completed, lap_index[completed]] = info["lap_time"][completed].double()
        reached = torch.where(alive, info["distance"].double(), reached)
        alive &= ~terminated & (info["laps"] < 3)
        if not alive.any():
            break
    finished = ~torch.isnan(lap_times[:, 2])
    # Higher is better: finishers by -(lap 2 + lap 3), everyone else far below, ordered by distance reached
    return torch.where(finished, -(lap_times[:, 1] + lap_times[:, 2]), -1000.0 + reached / track_length), lap_times


def evolve(args: argparse.Namespace, folder: Path) -> None:
    """Evolution strategies on the last layer(s) of a trained policy, scored by the deterministic 3-lap race itself."""
    # Full float32 matrix multiplies: these races are decided by small differences in the policy's actions
    torch.set_float32_matmul_precision("highest")
    saved = torch.load(args.init, map_location="cpu", weights_only=False)
    features = any(key.startswith("features.") for key in saved["policy"])
    model = build(saved["kwargs"]["policy_kwargs"], features, device="cuda")
    model.policy.load_state_dict(saved["policy"])
    policy = model.policy
    inputs = (lambda o: policy.features(o)) if features else (lambda o: o)
    linears = [m for m in policy.policy_net if isinstance(m, torch.nn.Linear)] + [policy.action_head]
    evolved = linears[-args.es_layers:]
    frozen = policy.policy_net[: 2 * (len(linears) - args.es_layers)]
    shapes = [(layer.weight.shape, layer.bias.shape) for layer in evolved]
    blocks = [tensor.detach().flatten() for layer in evolved for tensor in (layer.weight, layer.bias)]
    theta = torch.cat(blocks).float()
    # Each weight matrix and bias vector is perturbed in proportion to its own RMS size
    scale = torch.cat([torch.full_like(b, float(b.pow(2).mean().sqrt()) + 1e-8) for b in blocks]) * args.es_sigma
    adam_m, adam_v, beta1, beta2 = torch.zeros_like(theta), torch.zeros_like(theta), 0.9, 0.999
    half = args.es_pop // 2
    env = FastRacingEnv(TRACK, None, dynamics=DYNAMICS, n_envs=2 * half + 1, device="cuda", seed=args.seed,
                        random_start=False, laps=3, duration=300.0, auto_reset=False)

    def unpack(candidates):
        layers, offset = [], 0
        for weight_shape, bias_shape in shapes:
            w, b = weight_shape.numel(), bias_shape.numel()
            layers.append((candidates[:, offset:offset + w].view(-1, *weight_shape), candidates[:, offset + w:offset + w + b]))
            offset += w + b
        return layers

    def make_act(candidates):
        layers = unpack(candidates)

        def act(observation):
            x = frozen(inputs(observation))
            for i, (w, b) in enumerate(layers):
                x = torch.einsum("pij,pj->pi", w, x) + b
                if i < len(layers) - 1:
                    x = torch.tanh(x)
            return x
        return act

    cpu_model = load_policy(args.init)
    cpu_linears = [m for m in cpu_model.policy.policy_net if isinstance(m, torch.nn.Linear)] + [cpu_model.policy.action_head]
    best = None
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    began, generation = time.time(), 0
    while time.time() - began < 60 * args.minutes:
        generation += 1
        noise = torch.randn(half, theta.numel(), device="cuda", generator=generator)
        candidates = torch.cat([theta[None], theta + scale * noise, theta - scale * noise])
        fitness, lap_times = race_population(env, make_act(candidates), env.track.length)
        del candidates
        ranks = torch.argsort(torch.argsort(fitness[1:])).float() / (2 * half - 1) - 0.5
        gradient = ((ranks[:half] - ranks[half:]) @ noise) / (2 * half) / scale
        adam_m = beta1 * adam_m + (1 - beta1) * gradient
        adam_v = beta2 * adam_v + (1 - beta2) * gradient**2
        step = adam_m / (1 - beta1**generation) / ((adam_v / (1 - beta2**generation)).sqrt() + 1e-12)
        centre_laps = [round(float(t), 3) for t in lap_times[0] if not torch.isnan(t)]
        log = {"generation": generation, "minutes": round((time.time() - began) / 60, 2),
               "centre_gpu_laps": centre_laps, "finishers": int((fitness[1:] > -500).sum()),
               "best_gpu": round(float(fitness[1:].max()), 3)}
        if generation % args.es_real_every == 1 or args.es_real_every == 1:
            for (w, b), layer in zip(unpack(theta[None]), cpu_linears[-args.es_layers:]):
                layer.weight.data.copy_(w[0].cpu())
                layer.bias.data.copy_(b[0].cpu())
            result = race(cpu_model)
            laps = [round(t, 3) for t in result.lap_times]
            log["real_laps"] = laps
            if best is None or score(result) > best[0]:
                best = (score(result), laps, dict(log))
                cpu_model.save(folder / "best.pt")
            if reached_target(laps):
                print(f"[{args.name}] target reached", flush=True)
                break
        with open(folder / "evals.jsonl", "a") as out:
            out.write(json.dumps(log) + "\n")
        print(f"[{args.name}] {log}", flush=True)
        theta = theta + args.es_lr * scale * step
    promote(folder, args, best, ALGORITHM, load_policy)


def main():
    args = parse_arguments()
    folder = start_run(ALGORITHM, args)
    if args.mode == "es":
        evolve(args, folder)
        return

    env = make_training_env(args)
    if args.epo:
        epo_kwargs = dict(agents=args.epo_agents, latent_size=args.epo_latent, horizon=args.n_steps, epochs=args.epochs,
                          gamma=args.gamma, gae_lambda=args.lam, learning_rate=args.lr, kl_threshold=args.epo_kl_threshold,
                          clip=args.clip, ga_gamma=args.epo_ga_gamma, off_lambda=args.epo_off_lambda,
                          mutation_sigma=args.epo_sigma, log_std_init=args.log_std, minibatch_size=args.batch,
                          ga_interval=args.epo_ga_interval, ga_warmup=args.epo_ga_warmup, code_match=bool(args.epo_code_match))
        model = EPO(env, tensorboard_log=str(folder / "tb"), seed=args.seed, device="cuda", **epo_kwargs)
        keeper = RaceAndKeepBest(folder, EPO(None, device="cpu", **epo_kwargs), args.name, args.minutes,
                                 args.eval_seconds, args.steps)
        try:
            model.learn(total_timesteps=10**13, callback=keeper)
        finally:
            promote(folder, args, keeper.best, ALGORITHM, load_policy)
        return
    width = [int(w) for w in args.net.split(",")]
    began = time.time()

    def learning_rate(_):
        """Constant, or decaying linearly over the run's wall-clock budget to --lr-end."""
        if args.lr_end is None:
            return args.lr
        left = max(0.0, 1.0 - (time.time() - began) / (60 * args.minutes))
        return args.lr_end + (args.lr - args.lr_end) * left

    policy_kwargs = dict(net_arch=dict(pi=width, vf=width), log_std_init=args.log_std)
    model = build(policy_kwargs, bool(args.features), env=env, device="cuda", poem_args=args if args.poem else None,
                  n_steps=args.n_steps, batch_size=args.batch, n_epochs=args.epochs, gamma=args.gamma,
                  gae_lambda=args.lam, learning_rate=learning_rate, clip_range=args.clip, ent_coef=args.ent,
                  target_kl=args.target_kl, verbose=0, tensorboard_log=str(folder / "tb"), seed=args.seed)
    if args.init:
        model.policy.load_state_dict(torch.load(args.init, map_location="cuda", weights_only=False)["policy"])
    if args.reset_log_std is not None:
        with torch.no_grad():
            model.policy.log_std.fill_(args.reset_log_std)
    if args.std_reset:
        with torch.no_grad():
            model.policy.log_std.copy_(torch.log(torch.tensor([float(x) for x in args.std_reset.split(",")])))
    keeper = RaceAndKeepBest(folder, build(policy_kwargs, bool(args.features)), args.name, args.minutes,
                             args.eval_seconds, args.steps)
    try:
        model.learn(total_timesteps=10**13, callback=[keeper, ExplorationBounds(args)])
    finally:
        promote(folder, args, keeper.best, ALGORITHM, load_policy)
    print(f"[{args.name}] finished; target reached: {keeper.target_reached}", flush=True)


if __name__ == "__main__":
    main()
