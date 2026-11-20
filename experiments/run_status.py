"""Print each named run's progress from its log: time, steps, latest and best real-sim laps, crashed checks."""
import ast
import sys

for name in sys.argv[1:]:
    rows = [ast.literal_eval(line.split("] ", 1)[1]) for line in open(f"experiments/logs/{name}.log")
            if line.startswith(f"[{name}] {{")]
    finished = [row for row in rows if len(row.get("laps", [])) == 3]
    best = min(finished, key=lambda row: row["laps"][1] + row["laps"][2]) if finished else None
    last = rows[-1]
    best_text = f"{best['laps']} (laps 2+3 {best['laps'][1] + best['laps'][2]:.2f})" if best else "none"
    print(f"{name}: {last['minutes']:.0f} min, {last['steps'] / 1e6:.0f}M steps | latest {last['laps']} | "
          f"best {best_text} | crashed {len(rows) - len(finished)} of {len(rows)}")
