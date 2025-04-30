import gymnasium as gym
import numpy as np
import torch
from critic_actor import ActorCritic

def evaluate_agent(num_episodes=10, render=True):
    # Create the CartPole environment
    env = gym.make('CartPole-v1', render_mode='human')
    
    # Load the trained agent
    checkpoint = torch.load('trained_agent.pth')
    agent = ActorCritic(state_dim=4, action_dim=2)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    
    # Evaluation loop
    total_rewards = []
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        
        # Run episode until done
        while True:
            if render:
                env.render()
                
            # Select action using the trained agent
            action, _ = agent.select_action(observation)
            
            # Step the environment
            observation, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            # If episode is done, break the loop
            if done or truncated:
                break
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1} completed with total reward: {episode_reward}")
    
    # Close the environment
    env.close()
    
    # Print evaluation statistics
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Min Reward: {min(total_rewards)}")
    print(f"Max Reward: {max(total_rewards)}")

if __name__ == "__main__":
    evaluate_agent(num_episodes=10, render=True)