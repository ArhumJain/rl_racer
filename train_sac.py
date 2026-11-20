"""SAC on the GPU racing env (gpu_racer.sac), from scratch or warm-started from a trained PPO policy.

    uv run python train_sac.py --name base --minutes 30 [--option value ...]

Runs, promotion and kept logs work as training.py describes, with the algorithm name "sac".
"""

from __future__ import annotations

import argparse

import torch

from gpu_racer import PPO
from gpu_racer.sac import SAC, ReplayBuffer
from training import RaceAndKeepBest, add_environment_arguments, make_training_env, promote, race, start_run

ALGORITHM = "sac"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add("--name", required=True)
    add("--minutes", type=float, default=30.0)
    add("--seed", type=int, default=0)
    add("--init", default=None, help="model.pt to start from (weights only)")
    add("--keep-as", default=None, help="also copy the run's best model here before its folder is deleted")
    add("--n-envs", type=int, default=2048)
    add_environment_arguments(add)
    # SAC
    add("--batch", type=int, default=8192)
    add("--gradient-steps", type=int, default=2, help="updates per vectorised env step")
    add("--buffer", type=int, default=4_000_000)
    add("--gamma", type=float, default=0.99)
    add("--tau", type=float, default=0.005)
    add("--lr", type=float, default=3e-4)
    add("--target-entropy", type=float, default=-2.0)
    add("--ent-init", type=float, default=0.05)
    add("--pi", default="256,256")
    add("--qf", default="512,512,512")
    add("--compile-update", type=int, default=1, help="torch.compile the SAC update")
    # warm start from a trained gpu_racer PPO policy (plain network, no position features)
    add("--from-ppo", default=None, help="PPO model.pt whose driving SAC starts from")
    add("--ppo-noise", type=float, default=0.05, help="action noise while the PPO policy fills the replay buffer")
    add("--ppo-steps", type=int, default=400, help="vectorised steps the PPO policy drives to fill the buffer")
    add("--bc-updates", type=int, default=3000, help="imitation updates of SAC's actor on PPO's actions")
    add("--critic-warmup", type=int, default=5000, help="critic-only updates before SAC proper starts")
    add("--start-log-std", type=float, default=-3.0, help="SAC actor's initial log std after imitation")
    add("--eval-seconds", type=float, default=60.0)
    return parser.parse_args()


def start_from_ppo(model: SAC, env, args: argparse.Namespace) -> None:
    """
    Gives SAC a PPO policy's driving as its starting point.

    1. The PPO policy drives every car (deterministic action plus small noise); those transitions fill the replay
       buffer, so the critics first learn from competent driving rather than crashes.
    2. SAC's actor imitates PPO's deterministic actions (tanh of its mean regressed onto them), and its
       exploration starts small.
    3. The critics train alone on that buffer with the actor frozen, so their first gradients for the actor are
       informed rather than random.
    """
    ppo = PPO.load(args.from_ppo, device="cuda")
    model.buffer = ReplayBuffer(model.buffer_size, model.device)
    observations, targets = [], []
    observation = env.reset()
    with torch.no_grad():
        for _ in range(args.ppo_steps):
            deterministic = ppo.policy.mean(observation).clamp(-1.0, 1.0)
            action = (deterministic + args.ppo_noise * torch.randn_like(deterministic)).clamp(-1.0, 1.0)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            real_next = torch.where(done[:, None], info["final_observation"], next_observation)
            model.buffer.add(observation, action, reward.float(), real_next, terminated.float())
            observations.append(observation)
            targets.append(deterministic)
            observation = next_observation
    observations, targets = torch.cat(observations), torch.cat(targets).clamp(-0.995, 0.995)
    print(f"[{args.name}] PPO policy filled the buffer with {model.buffer.size} transitions", flush=True)

    optimizer = torch.optim.Adam(list(model.actor.body.parameters()) + list(model.actor.mean.parameters()), lr=1e-3)
    for update in range(args.bc_updates):
        index = torch.randint(0, len(observations), (8192,), device=observations.device)
        imitated, _ = model.actor(observations[index], deterministic=True)
        loss = torch.nn.functional.mse_loss(imitated, targets[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        model.actor.log_std.weight.zero_()
        model.actor.log_std.bias.fill_(args.start_log_std)
    print(f"[{args.name}] imitation loss after {args.bc_updates} updates: {loss.item():.5f}", flush=True)

    cpu_actor = SAC("MlpPolicy", None, device="cpu", policy_kwargs=model.policy_kwargs)
    cpu_actor.actor.load_state_dict({k: v.cpu() for k, v in model.actor.state_dict().items()})
    result = race(cpu_actor)
    print(f"[{args.name}] imitated actor, real sim: laps {[round(x, 3) for x in result.lap_times]}, "
          f"covered {result.laps_covered:.2f}", flush=True)

    model.actor.requires_grad_(False)
    alpha = model.log_alpha.detach().exp()
    for update in range(args.critic_warmup):
        observation_b, action_b, reward_b, next_b, terminal_b = model.buffer.sample(model.batch_size)
        with torch.no_grad():
            next_action, next_log_prob = model.actor(next_b)
            q1_next, q2_next = model.critic_target(next_b, next_action)
            target = reward_b + model.gamma * (1 - terminal_b) * (torch.minimum(q1_next, q2_next) - alpha * next_log_prob)
        q1, q2 = model.critic(observation_b, action_b)
        loss = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)
        model.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model.critic_optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(model.critic_target.parameters(), model.critic.parameters()):
                target_parameter.lerp_(parameter, model.tau)
    model.actor.requires_grad_(True)
    print(f"[{args.name}] critic warm-up done ({args.critic_warmup} updates), loss {loss.item():.4f}", flush=True)


def main():
    args = parse_arguments()
    folder = start_run(ALGORITHM, args)
    env = make_training_env(args)
    policy_kwargs = dict(net_arch=dict(pi=[int(w) for w in args.pi.split(",")], qf=[int(w) for w in args.qf.split(",")]))
    model = SAC("MlpPolicy", env, learning_rate=args.lr, buffer_size=args.buffer, batch_size=args.batch, gamma=args.gamma,
                tau=args.tau, gradient_steps=args.gradient_steps, ent_coef_init=args.ent_init,
                target_entropy=args.target_entropy, policy_kwargs=policy_kwargs, device="cuda",
                tensorboard_log=str(folder / "tb"), seed=args.seed)
    if args.init:
        saved = torch.load(args.init, map_location="cuda", weights_only=False)
        model.actor.load_state_dict(saved["actor"])
        model.critic.load_state_dict(saved["critic"])
        model.critic_target.load_state_dict(saved["critic_target"])
    if args.from_ppo:
        start_from_ppo(model, env, args)
    if args.compile_update:
        model._update = torch.compile(model._update)
    keeper = RaceAndKeepBest(folder, SAC("MlpPolicy", None, device="cpu", policy_kwargs=policy_kwargs), args.name,
                             args.minutes, args.eval_seconds)
    try:
        model.learn(total_timesteps=10**13, callback=keeper)
    finally:
        promote(folder, args, keeper.best, ALGORITHM, lambda path: SAC.load(path, device="cpu"))
    print(f"[{args.name}] finished; target reached: {keeper.target_reached}", flush=True)


if __name__ == "__main__":
    main()
