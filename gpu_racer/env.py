"""Thousands of ``racer_rl`` racing envs stepped at once as tensors, on any torch device.

It reproduces ``make_env(track, reward, dynamics="dynamic", ...)``: the same car
model, track checks, observation, ``Step`` fields and episode ends. ``check.py``
verifies it against the real env step by step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from gymnasium import spaces

from racer_rl import Step, Track
from racer_rl.env import RacingEnv
from racer_rl.observation import CURVATURE_SCALE, LOOKAHEAD
from racer_rl.vehicle import DynamicRaceCar as Car

from gpu_racer.track import TrackTables, wrap_to_pi

CONFIG = RacingEnv.default_config()
SIMULATION_FREQUENCY = CONFIG["simulation_frequency"]
POLICY_FREQUENCY = CONFIG["policy_frequency"]
FRAMES = SIMULATION_FREQUENCY // POLICY_FREQUENCY
STEERING_RANGE = CONFIG["action"]["steering_range"]
ACCELERATION_RANGE = CONFIG["action"]["acceleration_range"]


@dataclass(frozen=True)
class TensorStep(Step):
    """A ``Step`` whose fields are tensors with one entry per env, plus the arc length."""

    s: torch.Tensor
    """ Arc length of the car along the lap [m] """


class BatchRacingEnv:
    """
    ``n_envs`` racing envs as one batch. ``step`` takes and returns tensors of shape
    ``[n_envs, ...]`` and resets finished envs itself; the observation each finished
    env ended on is in ``info["final_observation"]``.

    The reward function receives a ``TensorStep``, so it must work element-wise:
    arithmetic and comparisons, not ``if``, ``min`` or ``math``. For example
    ``step.progress - 10.0 * (step.on_track == 0)`` works here and in ``make_env``.

    With ``random_start``, the first ``standing_start_envs`` envs still use standing
    starts, and end after ``standing_laps`` laps or ``standing_duration`` seconds.
    """

    def __init__(
        self,
        track: str = "laguna_seca",
        reward: Callable[[TensorStep], torch.Tensor] | None = None,
        dynamics: str = "dynamic",
        n_envs: int = 1024,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        random_start: bool = True,
        duration: float = 60.0,
        laps: int | None = None,
        standing_start_envs: int = 0,
        standing_duration: float = 300.0,
        standing_laps: int | None = 3,
        seed: int = 0,
        auto_reset: bool = True,
    ) -> None:
        if dynamics != "dynamic":
            raise NotImplementedError("only dynamics='dynamic' is ported")
        self.device = torch.device(device)
        self.dtype = dtype
        # Track queries and bookkeeping in float64, except on Apple GPUs, which lack it
        self.precise = torch.float32 if self.device.type == "mps" else torch.float64
        self.num_envs = n_envs
        self.reward = reward
        self.auto_reset = auto_reset
        self.track = TrackTables(Track.from_name(track), self.device, self.precise)
        self.lookahead = torch.as_tensor(LOOKAHEAD, dtype=self.precise, device=self.device)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7 + len(LOOKAHEAD),), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

        if not random_start:
            standing_start_envs, standing_duration, standing_laps = n_envs, duration, laps

        def per_env(standing_value, other_value, dtype) -> torch.Tensor:
            values = torch.full((n_envs,), other_value, dtype=dtype, device=self.device)
            values[:standing_start_envs] = standing_value
            return values

        self.standing = per_env(True, False, torch.bool)
        self.step_limit = per_env(_steps_to_reach(standing_duration), _steps_to_reach(duration), torch.long)
        no_limit = lambda laps: float("inf") if laps is None else float(laps)
        self.lap_limit = per_env(no_limit(standing_laps), no_limit(laps), self.precise)

        def zeros(dtype, *shape) -> torch.Tensor:
            return torch.zeros(n_envs, *shape, dtype=dtype, device=self.device)

        self.state = zeros(dtype, 7)  # x, y, heading, speed, lateral_speed, yaw_rate, steering_angle
        self.applied = zeros(dtype, 2)  # acceleration [m/s2] and steering [rad] the car executes
        self.previous_command = zeros(dtype, 2)  # pedal and steering in [-1, 1]
        self.sideslip = zeros(dtype)
        self.lateral_acceleration = zeros(dtype)
        self.speed_before = zeros(dtype)
        self.steps = zeros(torch.long)
        self.segment = zeros(torch.long)
        self.progress = zeros(self.precise)
        self.lateral = zeros(self.precise)
        self.half_width = zeros(self.precise)
        self.on_track = zeros(torch.bool)
        self.distance = zeros(self.precise)
        self.laps = zeros(self.precise)
        self.time = zeros(self.precise)
        self.lap_start_time = zeros(self.precise)

    def reset(self) -> torch.Tensor:
        self._reset_rows(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        return self.observe()

    def close(self) -> None:
        """Nothing to release; here so code written for SB3's VecEnv runs unchanged."""

    def _reset_rows(self, rows: torch.Tensor) -> None:
        n = self.num_envs
        uniform = lambda low, high: low + (high - low) * torch.rand(
            n, dtype=self.precise, device=self.device, generator=self.generator
        )
        start = torch.where(self.standing, 0.0, uniform(0.0, self.track.length))
        lateral = torch.where(self.standing, 0.0, uniform(-0.5, 0.5) * self.track.half_width(start))
        speed = torch.where(self.standing, 0.0, uniform(0.0, 20.0))
        state = torch.zeros(n, 7, dtype=self.precise, device=self.device)
        state[:, :2] = self.track.position(start, lateral)
        state[:, 2] = self.track.heading(start)
        state[:, 3] = speed
        self.state[rows] = state[rows].to(self.dtype)
        for tensor in (self.applied, self.previous_command, self.sideslip, self.lateral_acceleration,
                       self.steps, self.distance, self.laps, self.time, self.lap_start_time):
            tensor[rows] = 0
        self.speed_before[rows] = speed[rows].to(self.dtype)
        self.segment[rows] = self.track.segment_at(start)[rows]
        self._locate(rows)

    def _locate(self, rows: torch.Tensor | None = None) -> None:
        segment, s, lateral = self.track.locate(self.state[:, :2].to(self.precise), self.segment)
        half_width = self.track.half_width(s)
        if rows is None:
            rows = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.segment = torch.where(rows, segment, self.segment)
        self.progress = torch.where(rows, s, self.progress)
        self.lateral = torch.where(rows, lateral, self.lateral)
        self.half_width = torch.where(rows, half_width, self.half_width)
        self.on_track = torch.where(rows, lateral.abs() <= half_width, self.on_track)

    def observe(self) -> torch.Tensor:
        speed, yaw_rate, steering = self.state[:, 3], self.state[:, 5], self.state[:, 6]
        heading_error = wrap_to_pi(self.state[:, 2].to(self.precise) - self.track.heading(self.progress))
        curvatures = self.track.curvature(self.progress[:, None] + self.lookahead)
        return torch.cat(
            [
                torch.stack(
                    [
                        (speed / Car.MAX_SPEED).to(self.precise),
                        self.lateral / self.half_width,
                        torch.sin(heading_error),
                        torch.cos(heading_error),
                        (steering / STEERING_RANGE[1]).to(self.precise),
                        self.sideslip.to(self.precise),
                        yaw_rate.to(self.precise),
                    ],
                    -1,
                ),
                CURVATURE_SCALE * curvatures,
            ],
            -1,
        ).to(torch.float32)

    def step(self, actions: torch.Tensor):
        dt = 1 / POLICY_FREQUENCY
        actions = torch.as_tensor(actions, device=self.device).to(self.dtype).clamp(-1.0, 1.0)
        # ContinuousAction.get_action's lmap, then RaceCar.act
        command = _lmap(actions[:, 0], ACCELERATION_RANGE).clamp(-1.0, 1.0)
        acceleration = command * torch.where(command >= 0, Car.MAX_ACCELERATION, Car.MAX_BRAKING)
        steering_command = _lmap(actions[:, 1], STEERING_RANGE)
        self.applied = torch.stack([acceleration, steering_command], -1)
        self.speed_before = self.state[:, 3].clone()
        self.time = self.time + dt
        self.steps = self.steps + 1
        for _ in range(FRAMES):
            self._frame(1 / SIMULATION_FREQUENCY)

        previous = self.progress
        self._locate()
        half_lap = self.track.length / 2
        step_distance = torch.remainder(self.progress - previous + half_lap, self.track.length) - half_lap
        before = self.distance
        self.distance = self.distance + step_distance
        finish = (self.laps + 1) * self.track.length
        lap_completed = self.distance >= finish
        crossing = self.time - dt * (1 - (finish - before) / (self.distance - before))
        lap_time = torch.where(lap_completed, crossing - self.lap_start_time, torch.nan)
        self.lap_start_time = torch.where(lap_completed, crossing, self.lap_start_time)
        self.laps = self.laps + lap_completed

        observation = self.observe()
        pedal = acceleration / torch.where(acceleration >= 0, Car.MAX_ACCELERATION, Car.MAX_BRAKING)
        command = torch.stack([pedal, steering_command / STEERING_RANGE[1]], -1)
        cast = lambda tensor: tensor.to(self.dtype)
        car = self.state
        step = TensorStep(
            progress=cast(step_distance),
            speed=car[:, 3],
            lateral=cast(self.lateral),
            heading_error=cast(wrap_to_pi(car[:, 2].to(self.precise) - self.track.heading(self.progress))),
            on_track=self.on_track,
            pedal=command[:, 0],
            steering=command[:, 1],
            steering_actual=car[:, 6] / STEERING_RANGE[1],
            previous_pedal=self.previous_command[:, 0],
            previous_steering=self.previous_command[:, 1],
            longitudinal_acceleration=(car[:, 3] - self.speed_before) / dt,
            lateral_acceleration=self.lateral_acceleration,
            sideslip=self.sideslip,
            yaw_rate=car[:, 5],
            curvature=cast(self.track.curvature(self.progress)),
            lap_time=cast(lap_time),
            time=cast(self.time),
            dt=dt,
            max_speed=Car.MAX_SPEED,
            grip=Car.GRIP,
            half_width=cast(self.half_width),
            s=cast(self.progress),
        )
        self.previous_command = command
        if self.reward is None:
            reward = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
        else:
            reward = torch.broadcast_to(torch.as_tensor(self.reward(step), device=self.device), (self.num_envs,))
        terminated = ~self.on_track
        truncated = (self.laps >= self.lap_limit) | (self.steps >= self.step_limit)
        info = {"final_observation": observation, "distance": self.distance, "laps": self.laps,
                "lap_time": lap_time}
        if self.auto_reset:
            done = terminated | truncated
            self._reset_rows(done)
            observation = torch.where(done[:, None], self.observe(), observation)
        return observation, reward, terminated, truncated, info

    def _frame(self, dt: float) -> None:
        """One simulation frame of every car: ``_turn_wheels``, then ``DynamicRaceCar.step``."""
        travel = Car.STEERING_RATE * dt
        steering = self.state[:, 6] + torch.clamp(self.applied[:, 1] - self.state[:, 6], -travel, travel)
        pedal = self.applied[:, 0].clamp(-Car.GRIP, Car.GRIP)
        state = tuple(self.state[:, i] for i in range(6))
        h = dt / Car.SUBSTEPS
        add = lambda state, derivative, step: tuple(x + step * dx for x, dx in zip(state, derivative))
        for _ in range(Car.SUBSTEPS):
            k1 = _derivative(state, steering, pedal)
            k2 = _derivative(add(state, k1, h / 2), steering, pedal)
            k3 = _derivative(add(state, k2, h / 2), steering, pedal)
            k4 = _derivative(add(state, k3, h), steering, pedal)
            state = tuple(x + h / 6 * (a + 2 * b + 2 * c + d) for x, a, b, c, d in zip(state, k1, k2, k3, k4))
        x, y, heading, speed, lateral_speed, yaw_rate = state
        clamped = speed.clamp(Car.MIN_SPEED, Car.MAX_SPEED)
        self.sideslip = torch.atan2(lateral_speed, clamped.clamp(min=1.0))
        self.lateral_acceleration = _derivative(state, steering, pedal)[4] + yaw_rate * clamped
        self.state = torch.stack([x, y, heading, clamped, lateral_speed, yaw_rate, steering], -1)


def _lmap(value: torch.Tensor, interval: list[float]) -> torch.Tensor:
    """highway-env's ``utils.lmap`` from [-1, 1]."""
    return interval[0] + (value - (-1.0)) * (interval[1] - interval[0]) / (1.0 - (-1.0))


def _derivative(state: tuple, steering: torch.Tensor, pedal: torch.Tensor) -> tuple:
    """``DynamicRaceCar._derivative`` for every car at once."""
    _, _, heading, speed, lateral_speed, yaw_rate = state
    arm = Car.WHEELBASE / 2
    low, high = Car.BLEND_SPEEDS
    dynamic = ((speed - low) / (high - low)).clamp(0.0, 1.0)

    front_grip = torch.sqrt(torch.clamp((Car.GRIP / 2) ** 2 - (pedal / 2) ** 2, min=0.0))
    rear_grip = torch.sqrt(torch.clamp((Car.REAR_GRIP * Car.GRIP / 2) ** 2 - (pedal / 2) ** 2, min=0.0))
    front_slip = steering - torch.atan2(lateral_speed + arm * yaw_rate, speed)
    rear_slip = -torch.atan2(lateral_speed - arm * yaw_rate, speed)
    tyre = lambda slip: torch.sin(Car.TYRE_SHAPE * torch.atan(Car.TYRE_STIFFNESS * slip))
    # The sim skips the tyre forces below the blend; they are finite there, so zeroing
    # them by `dynamic` gives the same result
    front_force = torch.where(dynamic > 0, front_grip * tyre(front_slip), 0.0)
    rear_force = torch.where(dynamic > 0, rear_grip * tyre(rear_slip), 0.0)

    tan_beta = 0.5 * torch.tan(steering)
    relaxation = 0.1
    kinematic_lateral = (speed * tan_beta - lateral_speed) / relaxation
    kinematic_yaw = (speed * tan_beta / arm - yaw_rate) / relaxation

    cos, sin = torch.cos(steering), torch.sin(steering)
    pedal = torch.where(speed >= Car.MAX_SPEED, pedal.clamp(max=0.0), pedal)
    acceleration = pedal + dynamic * (yaw_rate * lateral_speed - front_force * sin)
    lateral = dynamic * (front_force * cos + rear_force - yaw_rate * speed)
    yaw = dynamic * arm * (front_force * cos - rear_force) / Car.YAW_INERTIA
    return (
        speed * torch.cos(heading) - lateral_speed * torch.sin(heading),
        speed * torch.sin(heading) + lateral_speed * torch.cos(heading),
        yaw_rate,
        acceleration,
        lateral + (1 - dynamic) * kinematic_lateral,
        yaw + (1 - dynamic) * kinematic_yaw,
    )


def _steps_to_reach(duration: float) -> int:
    """Policy steps until the env's clock, which adds 1 / POLICY_FREQUENCY each step, reaches ``duration``.

    The count is taken in float64 the way the env accumulates its clock, so a time
    limit ends on the same step as in ``make_env`` whatever precision the batch uses.
    """
    if duration == float("inf"):
        return 2**62
    clock, steps = 0.0, 0
    while clock < duration:
        clock += 1 / POLICY_FREQUENCY
        steps += 1
    return steps
