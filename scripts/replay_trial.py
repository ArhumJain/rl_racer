"""Re-drive saved time trials in the simulator, to watch them and check their lap times.

    uv run python scripts/replay_trial.py laps/teammate.npz [more.npz ...] [--render]

The trials are the files written by ``time_trial(..., save_trial="laps/name")``. Each
one's recorded actions are fed back to the car from the same standing start, which
writes the usual time-trial video to ``videos/replay/<name>/`` (or shows it in a
window with ``--render``) and compares the lap times with the recorded ones.
To draw several recorded trials racing each other instead, use ``race_video.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pygame

from racer_rl import time_trial


def replay_policy(actions: np.ndarray, in_window: bool):
    """A policy that ignores the observation and plays back the recorded actions in order."""
    remaining = iter(actions)
    started = False

    def policy(observation):
        nonlocal started
        if in_window and not started:
            pygame.Window.from_display_module().focus()  # macOS opens the window behind the terminal
        started = True
        return next(remaining, np.zeros(2))

    return policy


def replay(path: Path, render: bool) -> None:
    print(render)
    trial = np.load(path)
    track, dynamics, laps = str(trial["track"]), str(trial["dynamics"]), int(trial["laps"])
    recorded = [round(float(t), 3) for t in trial["lap_times"]]
    print(f"{path.name}: {track}, {dynamics} dynamics, {laps} laps, recorded {recorded}", flush=True)
    if render:
        print(f"  playing in a window in real time ({sum(recorded):.0f} s)", flush=True)
    result = time_trial(
        replay_policy(trial["actions"], render),
        track,
        laps=laps,
        dynamics=dynamics,
        render=render,
        video=None if render else str(Path("videos/replay") / path.stem),
    )
    replayed = [round(t, 3) for t in result.lap_times]
    verdict = "matches" if np.allclose(replayed, recorded, atol=1e-3) and len(replayed) == len(recorded) else "DIFFERS"
    print(f"  replayed {replayed} ({verdict})" + (f", video {result.video}" if result.video else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trials", nargs="+", type=Path, help="trial files (.npz)")
    parser.add_argument("--render", action="store_true", help="show each trial in a window instead of saving a video")
    args = parser.parse_args()
    for path in args.trials:
        replay(path, args.render)


if __name__ == "__main__":
    main()
