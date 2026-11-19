"""Check BatchRacingEnv against the real racer_rl env, step by step, and time it.

    uv run python -m gpu_racer.check --device cpu --dtype float64   # must match to rounding
    uv run python -m gpu_racer.check --device cuda --dtype float32  # reports the drift
    uv run python -m gpu_racer.check --device cuda --bench          # steps per second

Every row of the batch is paired with its own real env. Both start from the real
env's reset state and receive identical actions: a driver that follows the track
(the hand-written one from tests/test_racing.py, tuned) plus smooth random noise, so
the episodes cover braking, sliding, crashes, timeouts and lap completions.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from racer_rl import make_env
from racer_rl.observation import CURVATURE_SCALE, LOOKAHEAD

from gpu_racer.env import BatchRacingEnv


def driver(obs: np.ndarray, lateral_grip: float = 18.0, braking: float = 15.0) -> np.ndarray:
    corner = np.sqrt(lateral_grip * CURVATURE_SCALE / np.maximum(np.abs(obs[7:]), 1e-4))
    before = np.concatenate([[0.0], LOOKAHEAD[:-1]])
    speed = min(80.0, np.sqrt(corner**2 + 2 * braking * before).min())
    pedal = 0.5 * (speed - 80.0 * obs[0])
    steering = 0.24 * obs[8] - 0.4 * obs[1] - 2.5 * obs[2]
    return np.clip([pedal, steering], -1, 1)


def reward(step):
    """Uses every Step field that varies, so the comparison covers them all."""
    return (
        step.progress / 8.0
        - 10.0 * (step.on_track == 0)
        + 0.01 * (step.lateral + step.heading_error + step.sideslip + step.yaw_rate + step.curvature)
        + 0.001 * (step.speed + step.longitudinal_acceleration + step.lateral_acceleration + step.half_width)
        + 0.1 * (step.pedal - step.previous_pedal + step.steering - step.previous_steering + step.steering_actual)
        + 0.0001 * step.time
    )


def car_state(env) -> list[float]:
    car = env.unwrapped.vehicle
    return [*car.position, car.heading, car.speed, car.lateral_speed, car.yaw_rate, car.steering_angle]


def compare(device: str, dtype: torch.dtype, n_random: int, n_standing: int, steps: int, noise: float) -> None:
    n = n_standing + n_random
    real = [make_env("laguna_seca", reward=reward, dynamics="dynamic", random_start=False, laps=3, duration=300)
            for _ in range(n_standing)]
    real += [make_env("laguna_seca", reward=reward, dynamics="dynamic", random_start=True, duration=60)
             for _ in range(n_random)]
    observations = np.array([env.reset(seed=i)[0] for i, env in enumerate(real)])

    batch = BatchRacingEnv("laguna_seca", reward, n_envs=n, device=device, dtype=dtype, random_start=True,
                           standing_start_envs=n_standing, duration=60, standing_duration=300, standing_laps=3,
                           auto_reset=False)
    batch.reset()
    as_tensor = lambda values, dtype: torch.as_tensor(np.asarray(values), dtype=dtype, device=batch.device)
    batch.state = as_tensor([car_state(env) for env in real], dtype)
    batch.speed_before = batch.state[:, 3].clone()
    batch.segment = as_tensor([env.unwrapped.segment for env in real], torch.long)
    batch.progress = as_tensor([env.unwrapped.progress for env in real], batch.precise)
    batch.lateral = as_tensor([env.unwrapped.lateral for env in real], batch.precise)
    batch.half_width = as_tensor([env.unwrapped.half_width for env in real], batch.precise)
    batch.on_track = as_tensor([env.unwrapped.on_track for env in real], torch.bool)
    errors = {"observation": np.abs(batch.observe().cpu().numpy() - observations).max()}

    rng = np.random.default_rng(0)
    active = np.ones(n, dtype=bool)
    perturbation = np.zeros((n, 2))
    state_error = reward_error = 0.0
    mismatches = {"segment": 0, "terminated": 0, "truncated": 0}
    first_divergence = None
    lap_times = [[] for _ in range(n)]
    for k in range(steps):
        perturbation = 0.9 * perturbation + noise * rng.standard_normal((n, 2))
        perturbation[:n_standing] = 0.0
        actions = np.array([driver(o) for o in observations]) + perturbation
        batch_obs, batch_reward, batch_terminated, batch_truncated, info = batch.step(as_tensor(actions, dtype))
        batch_obs, batch_reward = batch_obs.cpu().numpy(), batch_reward.cpu().numpy()
        batch_state, batch_segment = batch.state.cpu().numpy(), batch.segment.cpu().numpy()
        batch_lap_time = info["lap_time"].cpu().numpy()
        for i in np.flatnonzero(active):
            obs, r, terminated, truncated, _ = real[i].step(actions[i])
            observations[i] = obs
            difference = np.abs(np.array(car_state(real[i])) - batch_state[i]).max()
            state_error = max(state_error, difference)
            if first_divergence is None and difference > 1e-3:
                first_divergence = (i, k, difference)
            errors["observation"] = max(errors["observation"], np.abs(obs - batch_obs[i]).max())
            reward_error = max(reward_error, abs(r - batch_reward[i]))
            mismatches["segment"] += int(real[i].unwrapped.segment != batch_segment[i])
            mismatches["terminated"] += int(terminated != bool(batch_terminated[i]))
            mismatches["truncated"] += int(truncated != bool(batch_truncated[i]))
            if not np.isnan(batch_lap_time[i]):
                lap_times[i].append(float(batch_lap_time[i]))
            if terminated or truncated:
                active[i] = False
                real_laps = real[i].unwrapped.lap_times
                if len(real_laps) != len(lap_times[i]) or not np.allclose(real_laps, lap_times[i], atol=1e-3):
                    mismatches.setdefault("lap times", 0)
                    mismatches["lap times"] += 1
        if not active.any():
            break

    print(f"{dtype} on {device}: {n} envs ({n_standing} standing 3-lap), {k + 1} steps")
    print(f"  max error: state {state_error:.2e}, observation {errors['observation']:.2e}, reward {reward_error:.2e}")
    print(f"  mismatched steps: {mismatches}")
    print(f"  first state divergence above 1 mm: {first_divergence}")
    for i in range(n_standing):
        print(f"  standing env {i}: real laps {np.round(real[i].unwrapped.lap_times, 3).tolist()}, "
              f"batch laps {np.round(lap_times[i], 3).tolist()}")


def bench(device: str, dtype: torch.dtype) -> None:
    for n in (1024, 4096, 16384, 65536):
        env = BatchRacingEnv("laguna_seca", reward, n_envs=n, device=device, dtype=dtype)
        env.reset()
        actions = torch.zeros(n, 2, device=env.device)
        for k in range(120):
            if k == 20:
                if env.device.type == "cuda":
                    torch.cuda.synchronize()
                began = time.perf_counter()
            actions = (0.9 * actions + 0.1 * torch.randn(n, 2, device=env.device)).clamp(-1, 1)
            actions[:, 0] = 0.6
            env.step(actions)
        if env.device.type == "cuda":
            torch.cuda.synchronize()
        print(f"{n:6d} envs: {100 * n / (time.perf_counter() - began):12,.0f} steps/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--random-envs", type=int, default=200)
    parser.add_argument("--standing-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--noise", type=float, default=0.03)
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)
    if args.bench:
        bench(args.device, dtype)
    else:
        compare(args.device, dtype, args.random_envs, args.standing_envs, args.steps, args.noise)
