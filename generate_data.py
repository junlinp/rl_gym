import gymnasium as gym
import numpy as np
import json
import os

from pid_controller import Controller, MountainCarController

from data_io import save_episode_data, load_episode_data

expert_policy_mapping = {
    "CartPole-v1": Controller,
    "MountainCarContinuous-v0": MountainCarController,
}

def main():
    # Create the CartPole environment
    #env_name = "CartPole-v1"
    env_name = "MountainCarContinuous-v0"
    env = gym.make(env_name)

    episode_num = 64
    train_episode_data = []
    for episode in range(episode_num):
        observation, info = env.reset()
        controller = expert_policy_mapping[env_name]()
        episode_reward = 0
        step = 0
        # Run episode until done
        while True:
            env.render()
            action = controller.observe(observation)
            # Step the environment
            next_observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            sample = {
                "episode": episode,
                "step": step,
                "state": observation.tolist(),
                "action": action.tolist(),
                "next_state": next_observation.tolist(),
                "reward": reward,
                "done": done
            }
            train_episode_data.append(sample)
            # If episode is done, break the loop
            if done:
                break
            observation = next_observation
            step += 1
        print(f"Train Episode {episode + 1} completed with total reward: {episode_reward}")
    # Save episode data
    save_episode_data(train_episode_data, env_name)
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
