import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import tqdm
from data_io import load_episode_data
from pid_controller import MountainCarController
import einops
from torch.distributions import Normal
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim * 2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state:torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        mean = x[:, :action_dim]
        log_std = x[:, action_dim:]
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)

        dist = Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action).sum(-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)

        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        x = self.fc1(x)
        x = self.model(x) + x
        x = self.layer_norm(x)
        return self.fc2(x)

class TD3(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, lr_actor=1e-4, lr_critic=1e-4,
                 gamma=0.99, tau=0.005, buffer_size=100000, batch_size=256, policy_noise=0.1,
                 noise_clip=0.5, policy_freq=2):
        super(TD3, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic1_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy weights from main networks to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)
        
        # Initialize replay buffer
        self.memory = deque(maxlen=buffer_size)
        self.demonstration_memory = deque(maxlen=buffer_size)

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self.gradient_step = 32
        self.alpha = 0.01

    def select_action(self, state:torch.Tensor, noise_scale=0.1):
        state = state.to(self.device)
        state = einops.rearrange(state, 'b -> 1 b')
        action, log_prob = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def store_policy_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def store_demonstration(self, state, action, reward, next_state, done):
        self.demonstration_memory.append((state, action, reward, next_state, done))

    def store_intervention(self, state, action, reward, next_state, done):
        self.store_policy_transition(state, action, reward, next_state, done)
        self.store_demonstration(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0, 0

        for i in range(self.gradient_step):
            intervention_batch = random.sample(self.demonstration_memory, self.batch_size // 2)
            policy_batch = random.sample(self.memory, self.batch_size // 2)

            batch = intervention_batch + policy_batch
            state, action, reward, next_state, done = zip(*batch)
        
            # Convert to tensors
            state = torch.FloatTensor(np.array(state)).to(self.device)
            action = torch.FloatTensor(np.array(action)).to(self.device)
            reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
            done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
            # Update critics
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                next_action, next_log_prob = self.actor_target(next_state)
            
                # Compute the target Q value
                target_q1 = self.critic1_target(next_state, next_action)
                target_q2 = self.critic2_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)
                target_q = reward + (1 - done) * self.gamma * target_q

            # Get current Q estimates
            current_q1 = self.critic1(state, action)
            current_q2 = self.critic2(state, action)

            # Compute critic loss
            critic1_loss = nn.MSELoss()(current_q1, target_q)
            critic2_loss = nn.MSELoss()(current_q2, target_q)

            # Optimize the critics
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            # Update the frozen target networks
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        # Compute actor loss
        action, log_prob = self.actor(state)
        actor_loss =  (self.alpha * log_prob- self.critic1(state, action)).mean()
            
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return actor_loss.item(), (critic1_loss.item() + critic2_loss.item()) / 2

    def behavior_update(self):
        batch = self.demonstration_memory
        state, action, reward, next_state, done = zip(*batch)
        
        # Convert to tensors
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        # Update critics
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            #noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action, next_log_prob = self.actor_target(next_state)
            
            # Compute the target Q value
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q

        # Get current Q estimates
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        # Compute critic loss
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        # Optimize the critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        # Delayed policy updates
        self.total_it += 1
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = (self.actor(state) - action).pow(2).mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target networks
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            return actor_loss.item(), (critic1_loss.item() + critic2_loss.item()) / 2
        return 0, (critic1_loss.item() + critic2_loss.item()) / 2

    def eval_episode(self, episode_data):
        # compute the true Q value from episode_data
        value = 0
        # sort the episode_data by step which larger is first
        episode_data.sort(key=lambda x: x['step'], reverse=True)
        print(f"Episode data length: {len(episode_data)}")
        true_q_value = {}
        for episode in episode_data:
            state = episode['state']
            action = episode['action']
            next_state = episode['next_state']
            reward = episode['reward']
            done = episode['done']
            value = reward + self.gamma * value
            true_q_value[episode['step']] = {
                'state': state,
                'action': action,
                'value': value
            }

        loss = 0
        for step, q_value in true_q_value.items():
            state = torch.tensor([q_value['state']], dtype=torch.float32).to(self.device)
            action = torch.tensor([q_value['action']], dtype=torch.float32).to(self.device)
            value = q_value['value']
            q_value = self.critic1_target(state, action)
            print(f"Step {step}, Q value: {q_value}, reward: {value}, diff: {q_value - value}")
            loss += (q_value.item() - value) ** 2
        print(f"Loss: {loss}")



if __name__ == "__main__":
    import gymnasium as gym
    env = gym.make('MountainCarContinuous-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = TD3(state_dim, action_dim)

    # load the episode_data from the episode_data/{env_name} directory
    episode_data = load_episode_data(f'episode_data/MountainCarContinuous-v0')
    print(f"Episode data length: {len(episode_data)}")

    for episode in episode_data:
        episode_index = episode['episode']
        state = episode['state']
        action = episode['action']
        next_state = episode['next_state']
        reward = episode['reward']
        done = episode['done']
        if done:
            reward = 1
        else:
            reward = 0
        agent.store_intervention(state, action, reward, next_state, done)


    #for i in tqdm.tqdm(range(0000)):
        # actor_loss, critic_loss = agent.behavior_update()
        # print(f"Episode {i} finished with actor_loss {actor_loss} and critic_loss {critic_loss}")

    
    expert_policy = MountainCarController()
    expert_take_over_step = 300
    # Training loop
    for episode in range(1000):
        state, _ = env.reset()
        episode_reward = 0
        step = 0
        while True:
            #env.render()
            if step < expert_take_over_step:
                action = agent.select_action(torch.tensor(np.array(state), dtype=torch.float32))
            else:
                action = expert_policy.observe(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated:
                reward = 1
            else:
                reward = 0
            if step < expert_take_over_step:
                agent.store_policy_transition(state, action, reward, next_state, done)
            else:
                agent.store_demonstration(state, action, reward, next_state, done)

            agent.update()

            state = next_state
            episode_reward += reward
            if done:
                print(f"Episode {episode} terminated {terminated} truncated {truncated}")
                break
            if step == expert_take_over_step:
                print(f"Step reach {expert_take_over_step}, control taken over by expert policy")
            step += 1
            # print(f"Episode {episode} finished with actor_loss {actor_loss} and critic_loss {critic_loss}")
        print(f"Episode {episode} finished with reward {episode_reward}")