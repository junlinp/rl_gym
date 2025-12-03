import gymnasium as gym
import numpy as np
import torch
from critic_actor import ActorCritic
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.optim import AdamW
import json
import os
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch.utils.data import Dataset, DataLoader
import wandb
from tqdm import tqdm
class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, history_length=2, future_length=2, 
                 hidden_dim=128, num_layers=3, dropout=0.1):
        super(WorldModel, self).__init__()

        self.history_length = history_length
        self.future_length = future_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection: combine state and action
        self.input_proj = nn.Linear(state_dim + action_dim, hidden_dim)
        
        # GRU encoder to process history sequence (deeper with dropout)
        self.encoder_gru = nn.GRU(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # GRU decoder for autoregressive future state prediction (deeper with dropout)
        self.decoder_gru = nn.GRU(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Output projection: predict next state from hidden state
        self.output_proj = nn.Linear(hidden_dim, state_dim)

    def forward(self, state:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        # state: (batch_size, history_length, state_dim)
        # action: (batch_size, history_length, action_dim) or (batch_size, history_length) if action_dim=1
        batch_size = state.shape[0]
        
        # Ensure action has the right shape
        if action.dim() == 2:
            # action is (batch_size, history_length), need to add dimension
            action = action.unsqueeze(-1)  # (batch_size, history_length, 1)
        
        # Concatenate state and action along feature dimension
        # (batch_size, history_length, state_dim + action_dim)
        x = torch.cat([state, action], dim=-1)
        
        # Project input to hidden dimension
        # (batch_size, history_length, hidden_dim)
        x = self.input_proj(x)
        x = F.relu(x)
        
        # Encode history sequence through GRU
        # encoder_out: (batch_size, history_length, hidden_dim)
        # hidden: (num_layers, batch_size, hidden_dim) for multi-layer GRU
        encoder_out, hidden = self.encoder_gru(x)
        
        # Use the last hidden state from the last layer as initial state for decoder
        # hidden: (num_layers, batch_size, hidden_dim)
        # Take the last layer's hidden state: (batch_size, hidden_dim)
        decoder_hidden = hidden[-1]  # Last layer's hidden state
        
        # Autoregressively predict future states
        future_states = []
        # For multi-layer GRU, we need to initialize all layers
        # Use the last layer's hidden state and replicate for other layers
        current_hidden = hidden  # (num_layers, batch_size, hidden_dim)
        
        # Use the last state as initial input for decoder
        last_state = state[:, -1:, :]  # (batch_size, 1, state_dim)
        last_action = action[:, -1:, :]  # (batch_size, 1, action_dim)
        decoder_input = torch.cat([last_state, last_action], dim=-1)  # (batch_size, 1, state_dim + action_dim)
        decoder_input = self.input_proj(decoder_input)  # (batch_size, 1, hidden_dim)
        decoder_input = F.relu(decoder_input)
        
        for _ in range(self.future_length):
            # Decode one step
            # decoder_out: (batch_size, 1, hidden_dim)
            # current_hidden: (1, batch_size, hidden_dim)
            decoder_out, current_hidden = self.decoder_gru(decoder_input, current_hidden)
            
            # Predict next state from decoder output
            next_state = self.output_proj(decoder_out.squeeze(1))  # (batch_size, state_dim)
            future_states.append(next_state)
            
            # Use predicted state and zero action (or last action) as next input
            # For simplicity, we'll use zero action for future predictions
            # In practice, you might want to use planned actions
            next_action = torch.zeros(batch_size, 1, self.action_dim, 
                                     device=state.device, dtype=state.dtype)
            decoder_input = torch.cat([next_state.unsqueeze(1), next_action], dim=-1)
            decoder_input = self.input_proj(decoder_input)
            decoder_input = F.relu(decoder_input)
        
        # Stack future states: (batch_size, future_length, state_dim)
        future_states = torch.stack(future_states, dim=1)
        
        return future_states
    
    def predict_next_state(self, state:torch.Tensor, action) -> torch.Tensor:
        """
        Predict single next state from current state and action.
        state: (batch_size, state_dim) or (state_dim,)
        action: (batch_size, action_dim) or (action_dim,) or scalar or tensor
        Returns: (batch_size, state_dim) or (state_dim,)
        """
        # Handle single sample case
        single_sample = state.dim() == 1
        if single_sample:
            state = state.unsqueeze(0)
        
        batch_size = state.shape[0]
        device = state.device
        
        # Ensure action has the right shape and type
        if isinstance(action, (int, float)):
            action = torch.tensor([[float(action)]], device=device, dtype=state.dtype)
        elif isinstance(action, torch.Tensor):
            if action.dim() == 0:
                action = action.unsqueeze(0).unsqueeze(0)
            elif action.dim() == 1:
                if action.shape[0] == 1:
                    action = action.unsqueeze(0)
                else:
                    action = action.unsqueeze(0)
            action = action.to(device)
            # Ensure action has action_dim dimension
            if action.shape[-1] != self.action_dim:
                if action.numel() == 1:
                    action = action.unsqueeze(-1)
                elif action.shape[-1] == 1 and self.action_dim == 1:
                    pass  # Already correct
                else:
                    action = action.unsqueeze(-1) if action.dim() == 1 else action
        
        # Ensure action has batch dimension
        if action.shape[0] != batch_size:
            if action.shape[0] == 1:
                action = action.repeat(batch_size, 1)
            else:
                action = action.unsqueeze(0)
        
        # Create history by repeating current state and action
        state_history = state.unsqueeze(1).repeat(1, self.history_length, 1)  # (batch_size, history_length, state_dim)
        action_history = action.unsqueeze(1).repeat(1, self.history_length, 1)  # (batch_size, history_length, action_dim)
        
        # Use forward to predict future states, take only the first one
        future_states = self.forward(state_history, action_history)  # (batch_size, future_length, state_dim)
        next_state = future_states[:, 0, :]  # (batch_size, state_dim)
        
        if single_sample:
            next_state = next_state.squeeze(0)
        
        return next_state

def update_world_model(world_model, state:torch.Tensor, action:torch.Tensor, reward:torch.Tensor) -> torch.Tensor:
    length_state = state.shape[0]
    model_history_length = world_model.history_length
    model_future_length = world_model.future_length

    optimizer = torch.optim.Adam(world_model.parameters(), lr=0.001)
    loss_list = []
    for epoch in range(10):
        for index in range(model_history_length - 1, length_state - model_future_length):
            optimizer.zero_grad()
            state_i = state[index - model_history_length + 1:index + 1, :]
            action_i = action[index - model_history_length + 1:index + 1, :]
            future_state_i = state[index + 1:index + model_future_length + 1, :]
            loss = update_world_model_once(world_model, state_i, action_i, future_state_i)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
    print(f"loss mean: {np.mean(np.array(loss_list))}")

def update_world_model_once(world_model, state:torch.Tensor, action:torch.Tensor, next_state:torch.Tensor) -> torch.Tensor:

    # state: (batch_size, history_length, state_dim)
    # action: (batch_size, history_length, action_dim)
    # next_state: (batch_size, future_length, state_dim)
    state = einops.rearrange(state, 'h s -> 1 h s')
    action = einops.rearrange(action, 'h a -> 1 h a')
    next_state = einops.rearrange(next_state, 'f s -> 1 f s')

    future_state = world_model(state, action)
    loss = F.l1_loss(future_state, next_state)
    return loss


class ImaginationAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ImaginationAgent, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        # forward the reward for the state and action
        x = torch.cat([state, action], dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def imagination(self, state, action):
        # imagination the next state and reward
        next_state = self.world_model(state, action)
        reward = self.forward(state, action)
        return next_state, reward


def imagination_train(imagination_agent, state, action, reward):
    optimizer = torch.optim.Adam(imagination_agent.parameters(), lr=0.001)

    input_state = state[:-1, :]
    input_action = action
    target_reward = reward
    # Ensure target_reward has the right shape: (batch_size, 1)
    if target_reward.dim() == 1:
        target_reward = target_reward.unsqueeze(-1)
    optimizer.zero_grad()
    predicted_reward = imagination_agent.forward(input_state, input_action)
    loss = F.l1_loss(predicted_reward, target_reward)
    loss.backward()
    optimizer.step()
    return loss

def generate_imagined_rollout(world_model, agent, imagination_agent, start_states, rollout_length=10, device=None):
    """
    Generate imagined rollouts using the world model and agent policy.
    
    Args:
        world_model: WorldModel instance for state prediction
        agent: ActorCritic agent for action selection
        imagination_agent: ImaginationAgent for reward prediction
        start_states: Starting states (num_starts, state_dim) or list of states
        rollout_length: Length of imagined rollout
        device: Device to use for tensors
    
    Returns:
        imagined_observations: List of imagined states (numpy arrays)
        imagined_actions: List of imagined actions (integers/scalars)
        imagined_rewards: List of imagined rewards (floats)
    """
    if device is None:
        device = next(world_model.parameters()).device
    
    # Convert start_states to tensor if needed
    if isinstance(start_states, list):
        start_states = torch.stack([torch.tensor(s, dtype=torch.float32) if isinstance(s, np.ndarray) else s 
                                    for s in start_states])
    elif isinstance(start_states, np.ndarray):
        start_states = torch.tensor(start_states, dtype=torch.float32)
    
    if start_states.dim() == 1:
        start_states = start_states.unsqueeze(0)
    
    start_states = start_states.to(device)
    num_starts = start_states.shape[0]
    
    all_imagined_observations = []
    all_imagined_actions = []
    all_imagined_rewards = []
    
    for start_idx in range(num_starts):
        current_state = start_states[start_idx]  # (state_dim,)
        imagined_observations = [current_state.clone().cpu().numpy()]
        imagined_actions = []
        imagined_rewards = []
        
        for step in range(rollout_length):
            # Agent selects action from current imagined state
            with torch.no_grad():
                action_tensor, _ = agent.select_action(current_state.unsqueeze(0))
                # Extract action value (should be integer for discrete actions)
                if action_tensor.numel() == 1:
                    action = action_tensor.item()
                else:
                    action = action_tensor.squeeze().item()
            
            # World model predicts next state
            with torch.no_grad():
                next_state = world_model.predict_next_state(current_state, action_tensor.squeeze())
            
            # Imagination agent predicts reward
            with torch.no_grad():
                # Prepare input: state and action as separate arguments
                # current_state is (state_dim,), need (1, state_dim)
                state_for_reward = current_state.unsqueeze(0)  # (1, state_dim)
                
                # action_tensor: ensure it's (1, action_dim)
                # action_tensor from select_action is typically (1,) or scalar
                if action_tensor.dim() == 0:
                    # Scalar: convert to (1, 1)
                    action_for_reward = action_tensor.float().view(1, 1)
                elif action_tensor.dim() == 1:
                    # 1D tensor: ensure shape (1, action_dim)
                    if action_tensor.shape[0] == 1:
                        action_for_reward = action_tensor.float().view(1, -1)
                    else:
                        action_for_reward = action_tensor.float().unsqueeze(0)
                else:
                    # Already 2D or more, take first element and reshape
                    action_for_reward = action_tensor.float().view(1, -1)
                
                # Ensure action_for_reward has exactly action_dim columns
                if action_for_reward.shape[1] != agent.action_dim:
                    if action_for_reward.shape[1] == 1 and agent.action_dim == 1:
                        pass  # Already correct
                    else:
                        # Reshape to (1, action_dim)
                        action_for_reward = action_for_reward[:, :agent.action_dim] if action_for_reward.shape[1] > agent.action_dim else action_for_reward
                        if action_for_reward.shape[1] < agent.action_dim:
                            # Pad with zeros if needed
                            padding = torch.zeros(1, agent.action_dim - action_for_reward.shape[1], 
                                                 device=action_for_reward.device, dtype=action_for_reward.dtype)
                            action_for_reward = torch.cat([action_for_reward, padding], dim=1)
                
                # ImaginationAgent.forward expects (state, action) as separate args
                predicted_reward = imagination_agent(state_for_reward, action_for_reward)
                reward = predicted_reward.item()
            
            imagined_actions.append(int(action))
            imagined_rewards.append(float(reward))
            imagined_observations.append(next_state.clone().cpu().numpy())
            
            # Update current state for next iteration
            current_state = next_state
        
        all_imagined_observations.append(imagined_observations)
        all_imagined_actions.append(imagined_actions)
        all_imagined_rewards.append(imagined_rewards)
    
    return all_imagined_observations, all_imagined_actions, all_imagined_rewards

def train_in_simulation(env_name='CartPole-v1', num_episodes=1000, initial_lr=0.001, lr_schedule='onecycle', is_resume=False):
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    # Get actual action space size (for discrete actions, use n; for continuous, use shape[0])
    action_dim = 1
    
    # Initialize agent with AdamW optimizer
    agent = ActorCritic(state_dim, action_dim, lr=initial_lr)
    # Create deeper WorldModel with configurable depth
    world_model = WorldModel(
        state_dim, 
        action_dim, 
        history_length=2, 
        future_length=2,
        hidden_dim=256,  # Increased hidden dimension
        num_layers=4,     # Deeper GRU: 4 layers
        dropout=0.1      # Dropout for regularization
    )
    imagination_agent = ImaginationAgent(state_dim, action_dim)

    if is_resume:
        if os.path.exists("world_model.pth"):
            world_model.load_state_dict(torch.load("world_model.pth"))
        if os.path.exists("imagination_agent.pth"):
            imagination_agent.load_state_dict(torch.load("imagination_agent.pth"))
        if os.path.exists("actor.pth"):
            agent.actor.load_state_dict(torch.load("actor.pth"))
        if os.path.exists("critic.pth"):
            agent.critic.load_state_dict(torch.load("critic.pth"))

    # Training loop
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False

        observations = []
        actions = []
        rewards = []
        while not done:
            observations.append(np.array(state))
            # Select action: use random actions from environment for first 1000 episodes
            if episode < 1000:
                # Use random action from environment action space
                action_value = env.action_space.sample()  # Returns numpy scalar for discrete spaces
                action = torch.tensor(action_value, dtype=torch.long)
            else:
                # Use agent's policy to select action
                action, _ = agent.select_action(torch.tensor(state))
            # Store as Python int for consistency (works with both real and imagined actions)
            actions.append(int(action.item()) if isinstance(action, torch.Tensor) else int(action))
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
                observations.append(np.array(next_state))

            rewards.append(reward)
            # Update agent
            state = next_state
            episode_reward += reward
        assert len(observations) == len(actions) + 1 == len(rewards) + 1
        
        # Update world model and imagination agent on real data
        # Convert observations to numpy array first to avoid warning
        observations_array = np.array(observations)
        
        # Convert actions to 2D tensor: (length, action_dim)
        # Ensure actions are valid (within action space range)
        actions_valid = [max(0, min(int(a), action_dim - 1)) for a in actions]  # Clamp to valid range
        actions_tensor = torch.tensor(actions_valid, dtype=torch.long)
        if actions_tensor.dim() == 1:
            actions_tensor = actions_tensor.unsqueeze(-1)  # (length, 1)
        
        # For world model, actions can be long/int
        update_world_model(world_model, torch.tensor(observations_array), actions_tensor, torch.tensor(rewards))
        
        # For imagination agent, actions need to be float
        actions_tensor_float = actions_tensor.float()
        imagination_train(imagination_agent, torch.tensor(observations_array), actions_tensor_float, torch.tensor(rewards))
        
        # Generate imagined rollouts from some states in the episode
        # Use states from the middle/end of episode for better imagination
        num_imagination_starts = min(3, len(observations) - 1)
        if num_imagination_starts > 0:
            start_indices = np.linspace(len(observations) // 2, len(observations) - 1, 
                                       num_imagination_starts, dtype=int)
            start_states = [observations[idx] for idx in start_indices]
            
            # Generate imagined rollouts
            device = next(world_model.parameters()).device
            imagined_observations_list, imagined_actions_list, imagined_rewards_list = \
                generate_imagined_rollout(world_model, agent, imagination_agent, 
                                         start_states, rollout_length=5, device=device)
        else:
            imagined_observations_list, imagined_actions_list, imagined_rewards_list = [], [], []
        
        # Combine real and imagined data for training
        all_observations = observations.copy()
        # Ensure all actions are valid Python ints within action space range
        all_actions = [max(0, min(int(a), action_dim - 1)) if not isinstance(a, (int, np.integer)) else max(0, min(int(a), action_dim - 1)) for a in actions]
        all_rewards = rewards.copy()
        
        # Add imagined rollouts to training data
        imagined_steps = 0
        for imagined_obs, imagined_act, imagined_rew in zip(imagined_observations_list, 
                                                             imagined_actions_list, 
                                                             imagined_rewards_list):
            # Skip the first observation (it's the starting state, already in real data)
            all_observations.extend(imagined_obs[1:])
            # Ensure imagined actions are valid Python ints within action space range
            all_actions.extend([max(0, min(int(a), action_dim - 1)) for a in imagined_act])
            all_rewards.extend(imagined_rew)
            imagined_steps += len(imagined_act)
        
        # Train agent on combined real + imagined data
        train_result = agent.update(all_observations, all_actions, all_rewards)
        
        # Log imagination usage
        if imagined_steps > 0 and (episode + 1) % 10 == 0:
            print(f"  Used {imagined_steps} imagined steps for training")

        online_train_result_dir = "online_train_result"
        if not os.path.exists(online_train_result_dir):
            os.makedirs(online_train_result_dir)
        with open(f"{online_train_result_dir}/train_result_{episode}.json", "w") as f:
            json.dump(train_result, f)

        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}")
    
    env.close()
    torch.save(world_model.state_dict(), "world_model.pth")
    torch.save(imagination_agent.state_dict(), "imagination_agent.pth")
    torch.save(agent.actor.state_dict(), "actor.pth")
    torch.save(agent.critic.state_dict(), "critic.pth")
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

def generate_rollout(world_model, observations, actions):
    rollouts = []
    model_history_length = world_model.history_length
    model_future_length = world_model.future_length
    length_state = len(observations)
    observations = np.array(observations)
    actions = np.array(actions).reshape(-1, 1)
    for index in range(model_history_length - 1, length_state - model_future_length):
        state_i = observations[index - model_history_length + 1:index + 1, :]
        action_i = actions[index - model_history_length + 1:index + 1, :]
        future_state_i = observations[index + 1:index + model_future_length + 1, :]
        rollouts.append((state_i, action_i, future_state_i))
    return rollouts



class RolloutDataset(Dataset):
    def __init__(self, rollouts):
        self.rollouts = rollouts

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, index):
        return self.rollouts[index]

def warm_up_world_model(world_model, env_name='CartPole-v1', num_episodes=10):

    wandb_handle = wandb.init(project="world_model_warm_up", name="world_model_warm_up")

    env = gym.make(env_name)
    total_rollouts = []
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        truncated = False

        observations = []
        actions = []
        while not done and not truncated:
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)

            observations.append(state)
            actions.append(action)
            state = next_state
            if done or truncated:
                observations.append(next_state)
                break
        rollouts = generate_rollout(world_model,observations, actions)
        total_rollouts.extend(rollouts)
    env.close()

    rollout_dataset = RolloutDataset(total_rollouts)
    rollout_dataloader = DataLoader(rollout_dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-4)

    global_step = 0
    for epoch in tqdm(range(1024)):
        for step, batch in enumerate(rollout_dataloader):
            optimizer.zero_grad()
            state_i, action_i, future_state_i = batch
            future_state = world_model(state_i, action_i)
            loss = F.l1_loss(future_state, future_state_i)
            loss.backward()
            optimizer.step()
            wandb_handle.log({"loss": loss.item()}, step=global_step + 1)
            global_step += 1



if __name__ == "__main__":
    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 1
    env.close()
    # Create deeper WorldModel for warm-up
    world_model = WorldModel(
        state_dim, 
        action_dim,
        history_length=2,
        future_length=2,
        hidden_dim=256,  # Increased hidden dimension
        num_layers=4,     # Deeper GRU: 4 layers
        dropout=0.1       # Dropout for regularization
    )
    warm_up_world_model(world_model, env_name=env_name, num_episodes=1024)
    torch.save(world_model.state_dict(), "world_model.pth")
    exit(0)

    #env_name = 'Acrobot-v1'
    # Train agent in simulation
    print("Training agent in simulation...")
    trained_agent = train_in_simulation(
        env_name=env_name,
        num_episodes=8000,
        is_resume=True
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
