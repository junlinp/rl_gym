import gymnasium as gym
import numpy as np
import torch
from critic_actor import ActorCritic
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.optim import AdamW
import json
import os
def train_in_simulation(env_name='CartPole-v1', num_episodes=1000, initial_lr=0.001, lr_schedule='onecycle'):
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Initialize agent with AdamW optimizer
    agent = ActorCritic(state_dim, action_dim, lr=initial_lr)
    # Training loop
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False

        observations = []
        actions = []
        rewards = []
        while not done:
            observations.append(state)
            # Select action using current policy
            action, _ = agent.select_action(torch.tensor(state))
            actions.append(action)
            # Take action in environment
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            # if terminated:            
                # reward += 100
            if terminated:
                reward -= 100
            if truncated:
                reward += 200

            if done:
                observations.append(next_state)

            rewards.append(reward)
            # Update agent
            
            state = next_state
            episode_reward += reward
        assert len(observations) == len(actions) + 1 == len(rewards) + 1
        train_result = agent.update(observations, actions, rewards)

        online_train_result_dir = "online_train_result"
        if not os.path.exists(online_train_result_dir):
            os.makedirs(online_train_result_dir)
        with open(f"{online_train_result_dir}/train_result_{episode}.json", "w") as f:
            json.dump(train_result, f)

        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}")
    
    env.close()
    return agent

def evaluate_agent(agent, env_name='CartPole-v1', num_episodes=10):
    env = gym.make(env_name)
    total_rewards = []
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        while not done and not truncated:
            action, _ = agent.select_action(torch.tensor(state))
            next_state, reward, done, truncated, info = env.step(action.item())
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}, Reward: {episode_reward:.2f}")
    
    env.close()
    return np.mean(total_rewards)

if __name__ == "__main__":
    env_name = 'CartPole-v1'
    #env_name = 'Acrobot-v1'
    # Train agent in simulation
    print("Training agent in simulation...")
    trained_agent = train_in_simulation(
        env_name=env_name,
        num_episodes=20000,
    )
    
    # Evaluate trained agent
    print("\nEvaluating trained agent...")
    mean_reward = evaluate_agent(trained_agent, env_name=env_name)
    print(f"\nAverage evaluation reward: {mean_reward:.2f}")

    # Save trained agent
    os.makedirs("trained_agents", exist_ok=True)
    actor_path = f"trained_agents/{env_name}_actor.pth"
    critic_path = f"trained_agents/{env_name}_critic.pth"
    torch.save(trained_agent.actor.state_dict(), actor_path)
    torch.save(trained_agent.critic.state_dict(), critic_path)
    print(f"Trained agent saved to {actor_path} and {critic_path}")
