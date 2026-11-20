# # Racing with RL
#
# Train a policy that drives a car around Laguna Seca to achieve the fastest lap time!

from pathlib import Path
import numpy as np

from racer_rl import (
    Step,
    Track,
    format_time,
    make_env,
    rollout,
    show_video,
    time_trial,
)

TRACK = "laguna_seca"
DYNAMICS = "dynamic" # "kinematic"

# **Observation**, 19 dim:
#
# | index | content |
# | --- | --- |
# | 0 | speed / top speed |
# | 1 | offset from the centerline / half the track width (±1 at the edges) |
# | 2, 3 | sin and cos of the heading relative to the track |
# | 4 | where the front wheels are / full lock (±1) |
# | 5 | sideslip [rad] |
# | 6 | yaw rate [rad/s] |
# | 7 to 18 | curvature of the track 0, 10, 20, 30, 45, 60, 80, 100, 130, 160, 200 and 250 m ahead, times 50 m |
#
# **Action**, 2 dim, each in [-1, 1], applied at 10 Hz:
#
# | index | content |
# | --- | --- |
# | 0 | pedal: throttle if positive, brake if negative |
# | 1 | steering: right if positive |
#
# **Episode.** Ends when the centre of the car leaves the track or timeout


# ## 2. reward function
#
# A reward function takes a `Step`, which describes what happened during one step of the environment, and returns a number.
#
# | field | meaning |
# | --- | --- |
# | `progress` | distance covered [m]|
# | `speed` | speed  [m/s] |
# | `lateral` | offset from the centerline [m], positive to the right |
# | `heading_error` | heading relative to the direction of the track [rad], positive to the right |
# | `on_track` | bool |
# | `pedal`, `steering` | [-1, 1] |
# | `steering_actual` |  [-1, 1]; it lags behind `steering` due to max steering rate |
# | `previous_pedal`, `previous_steering` | the action of the previous step |
# | `longitudinal_acceleration` | [m/s²] |
# | `lateral_acceleration` | [m/s²], positive to the right |
# | `sideslip` | angle between where the car points and where it is going [rad]; large when the car slides |
# | `yaw_rate` | [rad/s], positive to the right |
# | `curvature` | curvature of the track at the car [1/m], positive in right-hand corners |
# | `lap_time` | time of the lap completed during this step [s], otherwise `None` |
# | `time` | time since the start of the episode [s] |
# | `dt` | duration of a step [s] |
# | `max_speed`, `grip` | top speed [m/s] and tyre grip [m/s²] of the car |
# | `half_width` | half the track width where the car is [m]: the car is on track while `abs(lateral) <= half_width` |


# the thing we ultimately want is fast lap time. but how to design a "proper" reward function? 

def reward(step: Step) -> float:
    'TODO'
    reward = step.progress                                   # the objective, dense
    # reward -= 0.1 * (step.steering - step.previous_steering) ** 2 
    # element-wise, so it works on the GPU env's tensors and on make_env's floats alike
    reward = reward - 10.0 * (step.on_track == 0)
     

    return reward


# ## 3. Your Racing Agent
#
# Good implementations of various RL algorithms can be found in [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).
# You should read the differences between those algorithms.


from gpu_racer import PPO, BatchRacingEnv
from stable_baselines3.common.callbacks import BaseCallback


def main():
    # ### Check your reward function
    #
    # Here we can do some sanity checks with some dummy policies


    drivers = {
        "step on gas": lambda obs: np.array([1.0, 0.0]),
        "not moving": lambda obs: np.array([0.0, 0.0]),
    }

    for name, driver in drivers.items():
        r = rollout(driver, reward, TRACK, dynamics=DYNAMICS)
        ending = "off track" if r["off_track"] else "driving"
        print(
            f"{name:22s} return {r['return']:8.1f}   "
            f"{r['distance']:5.0f} m in {r['time']:4.1f} s, {ending}"
        )



    RUN = Path("runs/your_policy")  

    train_env = BatchRacingEnv(
        TRACK,
        reward,
        dynamics=DYNAMICS,
        n_envs=4096,
        device="cuda",
        seed=42,
    )


    # TODO train your policy here
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=32,
        batch_size=16384,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=dict(pi=[512, 512], vf=[512, 512]), log_std_init=-1.0),
        device="cuda",
        verbose=1,
        tensorboard_log=str(RUN / "tb"),
    )

    model.learn(total_timesteps=100_000_000)
    model.save(RUN / "model")
    train_env.close()

    #
    # Define `policy`: a function that takes an observation and returns an action.

    # TODO set your trained policy here
    your_policy = model

    def policy(obs: np.ndarray) -> np.ndarray:
        return your_policy.predict(obs, deterministic=True)[0]

    # ## 4. Test your policy by racing it
    #
    # Here we will perform a single car run in the time trial format of 3 laps. (is your policy stable enough to finish at least 3 laps?)
    #
    # The whole time trial is saved, submit this `.npz` file to the Google Drive linked in the assignment `.pdf` or Piazza


    # note 1st lap starts from 0 speed so it will likely be slower than other laps
    result = time_trial(policy, TRACK, video="videos/your_policy", dynamics=DYNAMICS,laps=3, save_trial="laps/your_name_or_nickname")
    print(result)
    show_video(result.video)


if __name__ == "__main__":
    main()
