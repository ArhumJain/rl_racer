"""Collect training curves (episode return and real-sim laps vs gradient steps) of the curve_* reruns into one JSON."""

import ast
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Gradient steps per rollout: epochs x minibatches, with minibatches = cars x steps per car / minibatch size
RUNS = {
    "curve_ppo": dict(kind="ppo", cars=8192, horizon=32, per_rollout=10 * 8192 * 32 // 32768),
    "curve_ppo_ft": dict(kind="ppo", cars=8192, horizon=64, per_rollout=10 * 8192 * 64 // 65536),
    "curve_epo": dict(kind="epo", cars=32768, horizon=32, per_rollout=10 * 32768 * 32 // 32768),
    "curve_sac": dict(kind="sac"),
}


def find_logs(name: str) -> Path:
    for folder in (Path("experiments/curves") / name, Path("runs/ppo_current") / name,
                   Path("runs/sac_current") / name):
        if (folder / "tb").is_dir():
            return folder
    raise FileNotFoundError(name)


def scalars(folder: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    events = sorted((folder / "tb").rglob("events.*"))[-1]
    accumulator = EventAccumulator(str(events.parent), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags()["scalars"]:
        return np.array([]), np.array([])
    points = accumulator.Scalars(tag)
    return np.array([p.step for p in points], float), np.array([p.value for p in points], float)


def main() -> None:
    curves = {}
    for name, spec in RUNS.items():
        folder = find_logs(name)
        if spec["kind"] == "sac":
            steps, returns = scalars(folder, "rollout/ep_rew_mean")
            update_steps, updates = scalars(folder, "train/n_updates")
            to_gradient = lambda s: np.interp(s, update_steps, updates)
        else:
            tag = "epo/master_fitness" if spec["kind"] == "epo" else "rollout/ep_rew_mean"
            steps, returns = scalars(folder, tag)
            per_step = spec["per_rollout"] / (spec["cars"] * spec["horizon"])
            to_gradient = lambda s, per_step=per_step: np.asarray(s) * per_step
        evals = [json.loads(line) for line in open(folder / "evals.jsonl")]
        finished = [e for e in evals if len(e["laps"]) == 3]
        curves[name] = {
            "return_gradient_steps": to_gradient(steps).tolist(), "return": returns.tolist(),
            "eval_gradient_steps": to_gradient([e["steps"] for e in finished]).tolist(),
            "eval_laps23": [e["laps"][1] + e["laps"][2] for e in finished],
            "evals": len(evals), "crashed_evals": len(evals) - len(finished),
        }
        print(name, len(returns), "return points,", len(finished), "of", len(evals), "evals finished")
    es = [ast.literal_eval(line.split("] ", 1)[1]) for line in open("experiments/logs/esS1_l2.log")
          if line.startswith("[esS1_l2] {")]
    real = [d for d in es if len(d.get("real_laps", [])) == 3]
    curves["esS1_l2"] = {
        "generation": [d["generation"] for d in es if len(d["centre_gpu_laps"]) == 3],
        "centre_laps23": [d["centre_gpu_laps"][1] + d["centre_gpu_laps"][2] for d in es if len(d["centre_gpu_laps"]) == 3],
        "real_generation": [d["generation"] for d in real],
        "real_laps23": [d["real_laps"][1] + d["real_laps"][2] for d in real],
    }
    print("esS1_l2", len(es), "generations,", len(real), "real-sim checks")
    Path("experiments/curves").mkdir(parents=True, exist_ok=True)
    Path("experiments/curves/curves.json").write_text(json.dumps(curves))


if __name__ == "__main__":
    main()
