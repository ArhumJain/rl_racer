"""How saved policies drive lap 2 of the real 3-lap time trial: speed, pedal, sliding, track width and grip used.

    PYTHONPATH=. uv run python experiments/diagnose.py runs/ppo_best/model.pt [more model.pt ...]
"""

import sys

import numpy as np

from racer_rl import make_env
from racer_rl.vehicle import DynamicRaceCar
from train_ppo import load_policy

for path in sys.argv[1:]:
    model = load_policy(path)
    env = make_env("laguna_seca", dynamics="dynamic", random_start=False, laps=3, duration=600)
    observation, _ = env.reset()
    base, rows = env.unwrapped, []
    while True:
        action = model.predict(observation, deterministic=True)[0]
        observation, _, terminated, truncated, info = env.step(action)
        car = base.vehicle
        rows.append((len(base.lap_times), base.lateral, base.half_width, car.speed, car.lateral_speed,
                     np.degrees(abs(car.sideslip)), action[0], abs(base.track.curvature(base.progress)),
                     car.lateral_acceleration))
        if terminated or truncated:
            break
    lap2 = np.array(rows)
    lap2 = lap2[lap2[:, 0] == 1]
    lateral, half_width, speed, lateral_speed, slide, pedal, curvature, cornering = lap2[:, 1:].T
    width_used = np.abs(lateral) / half_width
    grip_used = np.hypot(cornering, np.gradient(speed) * 10) / DynamicRaceCar.GRIP
    corner = curvature > 1 / 150
    print(f"{path}: laps {[round(t, 3) for t in info['lap_times']]}")
    print(f"  speed: mean {speed.mean():.1f} m/s, top (with sliding) {np.hypot(speed, lateral_speed).max():.1f} m/s")
    print(f"  pedal: full throttle {np.mean(pedal > 0.95):.1%}, braking {np.mean(pedal < 0):.1%}, "
          f"full brake {np.mean(pedal < -0.95):.1%}")
    print(f"  slide: mean {slide.mean():.1f} deg, max {slide.max():.1f} deg, above 10 deg {np.mean(slide > 10):.1%}")
    print(f"  width used: mean {width_used.mean():.2f}, max {width_used.max():.3f}, "
          f"within 0.5 m of an edge {np.mean(half_width - np.abs(lateral) < 0.5):.1%}")
    print(f"  grip used: mean {grip_used.mean():.2f}, above 0.9 {np.mean(grip_used > 0.9):.1%}; "
          f"in corners (radius < 150 m) {grip_used[corner].mean():.2f}")
