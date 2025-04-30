import torch
from critic_actor import ActorCritic
import numpy as np
from data_io import load_episode_data
import gymnasium as gym
from tqdm import tqdm

def train_actor_critic(train_data, validation_data, state_dim, action_dim, num_max_steps=1000, lr=0.001):
    # Initialize the actor-critic agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    agent = ActorCritic(state_dim, action_dim, lr=lr, device=device)

    # Convert episode data into PyTorch tensors
    all_states = []
    all_actions = []
    all_rewards = []
    all_states_validation = []
    all_actions_validation = []
    all_rewards_validation = []

    for episode in train_data:
        all_states.append(torch.FloatTensor(episode['observations']))
        all_actions.append(torch.LongTensor(episode['actions']))
        all_rewards.append(torch.FloatTensor(episode['rewards']))
    for episode in validation_data:
        all_states_validation.append(torch.FloatTensor(episode['observations']))
        all_actions_validation.append(torch.LongTensor(episode['actions']))
        all_rewards_validation.append(torch.FloatTensor(episode['rewards']))
    
    # Concatenate all episodes
    states = torch.cat(all_states)
    actions = torch.cat(all_actions)
    rewards = torch.cat(all_rewards)
    states_validation = torch.cat(all_states_validation)
    actions_validation = torch.cat(all_actions_validation)
    rewards_validation = torch.cat(all_rewards_validation)
    # Create TensorDataset
    dataset = torch.utils.data.TensorDataset(states, actions, rewards)
    validation_dataset = torch.utils.data.TensorDataset(states_validation, actions_validation, rewards_validation)
    # Create DataLoader
    batch_size = 32
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    # Training loop
    for step in tqdm(range(num_max_steps)):
        total_actor_loss = 0
        for batch in tqdm(train_dataloader):
            states, actions, rewards = batch
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            actor_loss = agent.behavior_cloning_train(states, actions, rewards)
            total_actor_loss += actor_loss

        # validation the agent loss
        total_actor_accuracy_validation = 0
        for batch in tqdm(validation_dataloader):
            states, actions, rewards = batch
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)    
            predicted_actions, _ = agent.select_action(states)
            accuracy = actions == predicted_actions
            total_actor_accuracy_validation += torch.sum(accuracy)
        print(f"Step {step + 1}, Actor Loss: {total_actor_loss/len(train_dataloader):.2f}, Actor Accuracy: {total_actor_accuracy_validation/len(validation_dataset):.2f}")


        # Print progress
        if (step + 1) % 10 == 0:
            # Evaluate the trained agent in the environment
            env = gym.make('CartPole-v1')
            total_rewards = []
    
            for episode in range(16):  # Run 10 evaluation episodes
                observation, info = env.reset()
                episode_reward = 0
                done = False
                truncated = False
        
                while not (done or truncated):
                    # Select action using the trained agent
                    observation = torch.FloatTensor(observation).to(device)
                    action, _ = agent.select_action(observation)
            
                    # Take step in environment
                    observation, reward, done, truncated, info = env.step(action.item())

                    episode_reward += reward
        
                total_rewards.append(episode_reward)
            # Print average performance
            avg_reward = sum(total_rewards) / len(total_rewards)
            print(f"\nAverage reward over {len(total_rewards)} episodes: {avg_reward:.2f}")
            env.close()
    return agent
def split_training_data(training_data):
    #split the training data into training and validation
    training_data, validation_data = training_data[:int(len(training_data) * 0.8)], training_data[int(len(training_data) * 0.8):]
    return training_data, validation_data
# Example usage:

from pid_controller import Controller
class ExpertPolicy:
    def __init__(self):
        self.controller = Controller()
    def select_action(self, state):
        return self.controller.observe(state)
    def reset(self):
        self.controller = Controller()

def train_with_expert_policy(state_dim, action_dim, num_max_steps=1000, lr=0.001):
    # Initialize the actor-critic agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")
    agent = ActorCritic(state_dim, action_dim, lr=lr, device=device)


    env = gym.make('CartPole-v1')
    expert_policy = ExpertPolicy()
    for episode in range(num_max_steps):
        observation, info = env.reset()
        train_loss = 0
        train_reward = 0
        expert_observations = []
        expert_actions = []
        expert_rewards = []
        while True:
            action = expert_policy.select_action(observation)
            expert_observations.append(observation)
            expert_actions.append(action)
            observation, reward, done, truncated, info = env.step(action)
            train_reward += reward
            if done or truncated:
                break
        train_loss = agent.behavior_cloning_train(torch.FloatTensor(expert_observations), torch.LongTensor(expert_actions), torch.FloatTensor(expert_rewards))
        print(f"Episode {episode + 1}, Train Loss: {train_loss:.2f}, Train Reward: {train_reward:.2f}")
        accuracy = 0
        for i in range(len(expert_observations)):
            action, _ = agent.select_action(torch.FloatTensor(expert_observations[i]))
            same_action = action.item() == expert_actions[i]
            if same_action:
                accuracy += same_action
            else:
                break
        print(f"Episode {episode + 1}, Accuracy: {accuracy/len(expert_observations):.2f}")

        done = False
        truncated = False
        expert_policy.reset()
        observation, info = env.reset()

        eval_reward = 0
        while True:
            action, _ = agent.select_action(torch.FloatTensor(observation))
            observation, reward, done, truncated, info = env.step(action.item())
            eval_reward += reward
            if done or truncated:
                break
        print(f"Episode {episode + 1}, eval reward: {eval_reward:.2f}")
    env.close()
    return agent
if __name__ == "__main__":
    # Generate training data
    training_data = load_episode_data('./episode_data')
    
    # Define dimensions based on your environment
    state_dim = 4  # CartPole state dimension
    action_dim = 2  # CartPole action dimension

    #split the training data into training and validation
    # training_data, validation_data = split_training_data(training_data)
    # # Train the agent
    # trained_agent = train_actor_critic(
    #     train_data=training_data,
    #     validation_data=validation_data,
    #     state_dim=state_dim,
    #     action_dim=action_dim,
    #     num_max_steps=1000,
    #     lr=2e-5
    # )

    trained_agent = train_with_expert_policy(state_dim, action_dim, num_max_steps=1000, lr=2e-4)
    
    # Save the trained agent
    torch.save({
        'actor_state_dict': trained_agent.actor.state_dict(),
        'critic_state_dict': trained_agent.critic.state_dict(),
        'actor_optimizer_state_dict': trained_agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': trained_agent.critic_optimizer.state_dict()
    }, 'trained_agent.pth')
