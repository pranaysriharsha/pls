import os
import gym

from pls.algorithms.ppo_shielded import PPO_shielded
import numpy as np


def main(config_folder, config, model_at_step, n_test_episodes, monitor_cls):
    """
    Evaluate the given policy by executing it in the given environment.
    Calls stable_baselines3's default evaluate_policy function.

    :param config_folder: location of the config file
    :param config: a dict containing the configuration
    :param model_at_step: load a snapshot of the policy trained after model_at_step steps.
                          If given "end", load the last saved model
    :param n_test_episodes: number of episode to run
    :param monitor_cls:
    :return: mean and standard deviation of reward
    """

    # initialize the environment for evaluation
    env = gym.make(config["env"], **config["eval_env_features"])

    env = monitor_cls(
        env,
        allow_early_resets=False,
    )

    # load the trained policy
    if model_at_step == "end":
        path = os.path.join(config_folder, "model.zip")
    else:
        path = os.path.join(
            config_folder, "model_checkpoints", f"rl_model_{model_at_step}_steps.zip"
        )

    model = PPO_shielded.load(path, env)

    episode_rewards = []
    n_violations = 0

    for _ in range(n_test_episodes):
        obs = env.reset()
        done = False
        cumulative_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            cumulative_reward += reward
            if config.get("render", False):
                env.render()
            
            if done:
                if info.get("episode", {}).get("violate_constraint", False):
                    n_violations += 1

        episode_rewards.append(cumulative_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward, n_violations
