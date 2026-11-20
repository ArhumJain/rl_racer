"""What train_ppo.py and train_sac.py share: the compiled GPU env, the reward options, real-sim evaluation, and
promotion of a run's best policy.

Each run trains in runs/<algorithm>_current/<name>/ and races the real 3-lap time trial on the CPU every
--eval-seconds, keeping the run's fastest policy. When the run ends, that policy replaces runs/<algorithm>_best/
(and its video in videos/<algorithm>_best/) if it beats the best so far; the run's TensorBoard logs and
evaluations are kept in experiments/curves/<name>/, and its folder is deleted.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

import gpu_racer.env as gpu_env
from gpu_racer import BatchRacingEnv
from gpu_racer.env import Car
from racer_rl import time_trial

TRACK = "laguna_seca"
DYNAMICS = "dynamic"
TARGET_LAP = 67.0
RUNS, VIDEOS, CURVES = Path("runs"), Path("videos"), Path("experiments/curves")


def add_environment_arguments(add) -> None:
    """The env and reward options of both runners; ``add`` is a parser's ``add_argument``."""
    add("--duration", type=float, default=60.0)
    add("--standing", type=float, default=0.05, help="fraction of envs on 3-lap standing starts")
    add("--drift-starts", type=float, default=0.0, help="fraction of random starts that begin mid-slide")
    add("--drift-speed", default="10,60", help="min,max speed of a mid-slide start [m/s]")
    add("--drift-angle", default="0,70", help="min,max slide angle of a mid-slide start, either way [deg]")
    add("--progress-scale", type=float, default=1 / 8)
    add("--crash", type=float, default=1.25)
    add("--slide", type=float, default=0.0, help="bonus per radian of sideslip, capped at 1.5 rad")
    add("--brake", type=float, default=0.0, help="penalty per unit of brake pedal")
    add("--smooth", type=float, default=0.0, help="penalty on squared steering change")
    add("--slack", type=float, default=0.0, help="penalty per step for unused grip, except at full throttle or top speed")
    add("--straight", type=float, default=0.0,
        help="penalty per step for driving straight (no slide), fading to 0 at --straight-angle")
    add("--straight-angle", type=float, default=10.0, help="slide angle at which the --straight penalty reaches 0 [deg]")
    add("--straight-where", default="corners", choices=["corners", "everywhere"],
        help="where --straight applies; corners: radius under 250 m")


def make_reward(args: argparse.Namespace):
    def reward(step):
        value = args.progress_scale * step.progress - args.crash * (step.on_track == 0)
        sideslip = torch.as_tensor(step.sideslip).abs()
        if args.slide:
            value = value + args.slide * sideslip.clamp(max=1.5)
        if args.brake:
            value = value - args.brake * torch.as_tensor(step.pedal).clamp(max=0.0).abs()
        if args.smooth:
            value = value - args.smooth * (step.steering - step.previous_steering) ** 2
        if args.slack:
            # Minimum-time driving is always at full throttle, at top speed, or at the grip limit
            used = torch.sqrt(step.longitudinal_acceleration**2 + step.lateral_acceleration**2) / Car.GRIP
            limited = (step.pedal >= 0.95) | (step.speed >= step.max_speed - 0.5)
            value = value - args.slack * (1 - used).clamp(min=0.0) * ~limited
        if args.straight:
            shortfall = (1 - sideslip / np.radians(args.straight_angle)).clamp(min=0.0)
            if args.straight_where == "corners":
                shortfall = shortfall * (torch.as_tensor(step.curvature).abs() > 1 / 250)
            value = value - args.straight * shortfall
        return value

    return reward


def _frame_physics(state: torch.Tensor, applied: torch.Tensor):
    """One simulation frame of every car, as BatchRacingEnv._frame, written as a pure function."""
    dt = 1 / gpu_env.SIMULATION_FREQUENCY
    travel = Car.STEERING_RATE * dt
    steering = state[:, 6] + torch.clamp(applied[:, 1] - state[:, 6], -travel, travel)
    pedal = applied[:, 0].clamp(-Car.GRIP, Car.GRIP)
    car = tuple(state[:, i] for i in range(6))
    h = dt / Car.SUBSTEPS
    add = lambda values, derivative, k: tuple(x + k * dx for x, dx in zip(values, derivative))
    for _ in range(Car.SUBSTEPS):
        k1 = gpu_env._derivative(car, steering, pedal)
        k2 = gpu_env._derivative(add(car, k1, h / 2), steering, pedal)
        k3 = gpu_env._derivative(add(car, k2, h / 2), steering, pedal)
        k4 = gpu_env._derivative(add(car, k3, h), steering, pedal)
        car = tuple(x + h / 6 * (a + 2 * b + 2 * c + d) for x, a, b, c, d in zip(car, k1, k2, k3, k4))
    x, y, heading, speed, lateral_speed, yaw_rate = car
    clamped = speed.clamp(Car.MIN_SPEED, Car.MAX_SPEED)
    sideslip = torch.atan2(lateral_speed, clamped.clamp(min=1.0))
    lateral_acceleration = gpu_env._derivative(car, steering, pedal)[4] + yaw_rate * clamped
    state = torch.stack([x, y, heading, clamped, lateral_speed, yaw_rate, steering], -1)
    return state, sideslip, lateral_acceleration


_compiled_physics = torch.compile(_frame_physics, dynamic=False)


class FastRacingEnv(BatchRacingEnv):
    """
    BatchRacingEnv with the car physics compiled into a few fused GPU kernels (same results).

    With ``drift_starts`` > 0, that fraction of random starts begins mid-slide: random speed (``drift_speed``),
    slide angle (``drift_angle``, either way), yaw rate and wheel angle, with the car still travelling along
    the track. The grader never starts like this; it only shows the policy states it would rarely reach.
    """

    drift_starts = 0.0
    drift_speed = (10.0, 60.0)
    drift_angle = (0.0, 70.0)

    def _frame(self, dt: float) -> None:
        self.state, self.sideslip, self.lateral_acceleration = _compiled_physics(self.state, self.applied)

    def _reset_rows(self, rows: torch.Tensor) -> None:
        super()._reset_rows(rows)
        if not self.drift_starts:
            return
        n, device = self.num_envs, self.device
        uniform = lambda low, high: low + (high - low) * torch.rand(n, device=device, generator=self.generator)
        drifting = rows & ~self.standing & (torch.rand(n, device=device, generator=self.generator) < self.drift_starts)
        side = torch.where(torch.rand(n, device=device, generator=self.generator) < 0.5, -1.0, 1.0)
        speed, slip = uniform(*self.drift_speed), side * torch.deg2rad(uniform(*self.drift_angle))
        state = self.state.clone()
        state[:, 2] = state[:, 2] - slip.to(state.dtype)   # velocity keeps pointing along the track
        state[:, 3] = (speed * torch.cos(slip)).to(state.dtype)
        state[:, 4] = (speed * torch.sin(slip)).to(state.dtype)
        state[:, 5] = uniform(-2.0, 2.0).to(state.dtype)
        state[:, 6] = uniform(-0.25, 0.25).to(state.dtype)
        self.state = torch.where(drifting[:, None], state, self.state)
        self.sideslip = torch.where(drifting, torch.atan2(state[:, 4], state[:, 3].clamp(min=1.0)), self.sideslip)
        self.speed_before = torch.where(drifting, state[:, 3], self.speed_before)


def make_training_env(args: argparse.Namespace) -> FastRacingEnv:
    env = FastRacingEnv(TRACK, make_reward(args), dynamics=DYNAMICS, n_envs=args.n_envs, device="cuda",
                        seed=args.seed, random_start=True, duration=args.duration,
                        standing_start_envs=int(round(args.standing * args.n_envs)), standing_duration=300.0,
                        standing_laps=3)
    env.drift_starts = args.drift_starts
    env.drift_speed = tuple(float(x) for x in args.drift_speed.split(","))
    env.drift_angle = tuple(float(x) for x in args.drift_angle.split(","))
    return env


def start_run(algorithm: str, args: argparse.Namespace) -> Path:
    """Seeds torch and creates the run's folder, holding its settings."""
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")  # TF32 matrix multiplies on the GPU
    folder = RUNS / f"{algorithm}_current" / args.name
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(vars(args), indent=2))
    return folder


def race(policy):
    """The graded event: a deterministic 3-lap time trial from a standing start, in the real simulator."""
    return time_trial(lambda obs: policy.predict(obs, deterministic=True)[0], TRACK, laps=3, dynamics=DYNAMICS)


def score(result) -> tuple:
    """Higher is better: laps completed first, then the fastest laps 2 + 3."""
    laps = result.lap_times
    if len(laps) == 3:
        return (3, -(laps[1] + laps[2]))
    return (len(laps), float(result.laps_covered))


def reached_target(laps: list[float]) -> bool:
    return len(laps) == 3 and laps[1] < TARGET_LAP and laps[2] < TARGET_LAP


class RaceAndKeepBest(BaseCallback):
    """Races the real time trial every ``eval_seconds``, keeps the best weights, stops on budget or target."""

    def __init__(self, folder: Path, cpu_model, name: str, minutes: float, eval_seconds: float,
                 steps: float | None = None):
        super().__init__()
        self.folder, self.cpu_model, self.name = folder, cpu_model, name
        self.minutes, self.eval_seconds, self.steps = minutes, eval_seconds, steps
        self.began = time.time()
        self.next_eval = self.began + eval_seconds
        self.best = None
        self.target_reached = False

    def _on_step(self) -> bool:
        now = time.time()
        out_of_time = self.num_timesteps >= self.steps if self.steps else now - self.began > 60 * self.minutes
        if now < self.next_eval and not out_of_time:
            return True
        self.next_eval = now + self.eval_seconds
        self.cpu_model.policy.load_state_dict({k: v.cpu() for k, v in self.model.policy.state_dict().items()})
        result = race(self.cpu_model)
        laps = [round(t, 3) for t in result.lap_times]
        record = {"minutes": round((now - self.began) / 60, 2), "steps": self.num_timesteps, "laps": laps,
                  "covered": round(float(result.laps_covered), 3)}
        with open(self.folder / "evals.jsonl", "a") as log:
            log.write(json.dumps(record) + "\n")
        print(f"[{self.name}] {record}", flush=True)
        if self.best is None or score(result) > self.best[0]:
            self.best = (score(result), laps, record)
            self.model.save(self.folder / "best.pt")
        if reached_target(laps):
            self.target_reached = True
            return False
        return not out_of_time


def keep_training_logs(folder: Path, name: str) -> None:
    kept = CURVES / name
    shutil.rmtree(kept, ignore_errors=True)
    kept.mkdir(parents=True)
    if (folder / "tb").is_dir():
        shutil.copytree(folder / "tb", kept / "tb")
    if (folder / "evals.jsonl").exists():
        shutil.copy(folder / "evals.jsonl", kept / "evals.jsonl")


def promote(folder: Path, args: argparse.Namespace, best: tuple | None, algorithm: str, load) -> None:
    """
    Replaces the algorithm's best policy with this run's if it is better, keeps the run's logs, deletes its folder.

    :param best: (score, lap times, record of when it was found) of the run's best policy, or None
    :param load: loads a saved model for the CPU, to race it for the video
    """
    best_folder, video_folder = RUNS / f"{algorithm}_best", VIDEOS / f"{algorithm}_best"
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / f".{algorithm}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        summary_path = best_folder / "summary.json"
        previous = json.loads(summary_path.read_text()) if summary_path.exists() else None
        better = best is not None and (previous is None or tuple(best[0]) > tuple(previous["score"]))
        if better:
            shutil.rmtree(best_folder, ignore_errors=True)
            best_folder.mkdir(parents=True)
            shutil.copy(folder / "best.pt", best_folder / "model.pt")
            shutil.copy(folder / "evals.jsonl", best_folder / "evals.jsonl")
            summary = {"name": args.name, "score": list(best[0]), "laps": best[1], "found_at": best[2],
                       "config": vars(args)}
            summary_path.write_text(json.dumps(summary, indent=2))
            model = load(best_folder / "model.pt")
            shutil.rmtree(video_folder, ignore_errors=True)
            time_trial(lambda obs: model.predict(obs, deterministic=True)[0], TRACK, laps=3, dynamics=DYNAMICS,
                       video=str(video_folder))
        print(f"[{args.name}] run best {best[1] if best else None}; "
              f"{'NEW BEST' if better else 'best kept'}: {best[1] if better else (previous or {}).get('laps')}",
              flush=True)
    if args.keep_as and (folder / "best.pt").exists():
        shutil.copy(folder / "best.pt", args.keep_as)
    keep_training_logs(folder, args.name)
    shutil.rmtree(folder, ignore_errors=True)
