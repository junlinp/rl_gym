import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, activation_fn=nn.SiLU):
        super(ResidualBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_fn()
        )
    
    def forward(self, x):
        return self.model(x) + x
        
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.hidden_dim = 1024
        self.state_projector = nn.Linear(state_dim, self.hidden_dim)
        self.layer_num = 32
        # self.model = nn.ModuleList([
        #     ResidualBlock(self.hidden_dim) for _ in range(self.layer_num // 4)
        # ])
        self.model = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        self.action_projector = nn.Linear(self.hidden_dim, action_dim)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def forward(self, state):
        x = F.relu(self.state_projector(state))
        # for layer in self.model:
        #     x = layer(x)
        x = self.model(x)
        action_probs = F.softmax(self.action_projector(x), dim=-1)
        return action_probs

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.layer_norm = nn.LayerNorm(64)
        self.fc3 = nn.Linear(64, 1)
        self.layer_norm = nn.LayerNorm(64)
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.layer_norm(x)
        value = self.fc3(x)
        return value

class ActorCritic:
    def __init__(self, state_dim, action_dim, lr=0.001, device=None, lr_schedule='cosine_warm'):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
        self.device = device

        if lr_schedule == 'onecycle':
            self.actor_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.actor_optimizer, max_lr=lr, total_steps=1000)
            self.critic_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.critic_optimizer, max_lr=lr, total_steps=1000)
        elif lr_schedule == 'cosine_warm':
            self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.actor_optimizer, T_0=100, T_mult=2)
            self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.critic_optimizer, T_0=100, T_mult=2)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state:torch.Tensor):
        action_probs = self.actor(state)
        action = torch.multinomial(action_probs, 1)
        return action, action_probs[action]
    
    def update(self, observations, actions, rewards) -> dict:
        gamma = 0.99
        steps = len(actions)

        expected_value = rewards
        state = torch.FloatTensor(observations[:-1]).to(self.device)
        V_last = self.critic(state[-1])
        R = 0
        # Start from the end of `rewards` and accumulate reward sums
        # into the `expected_value` array
        for i in range(steps - 1, -1, -1):
            expected_value[i] += gamma * R
            R = expected_value[i]
        expected_value = torch.FloatTensor(expected_value).to(self.device)
        # print(f"expected_value: {expected_value}")
        # print(f"expected_value.shape: {expected_value.shape}")
        #print(f"V_last: {V_last}")

        V_last = einops.repeat(V_last, '1 -> b 1', b = steps)
        # print(f"V_last: {V_last}")
        #print(f"V_last.shape: {V_last.shape}")

        expected_value = einops.rearrange(expected_value, 'b -> b 1')
        #expected_value += V_last

        action = torch.LongTensor(actions).to(self.device)
        next_state = torch.FloatTensor(observations[1:]).to(self.device)
        reward = einops.rearrange(torch.FloatTensor(rewards).to(self.device), 'b -> b 1')
        


        # # Calculate value estimates
        current_value = self.critic(state)
        next_value = self.critic(next_state)

        # # Calculate TD error
        #print(f"expected_value: {expected_value.shape}")
        #print(f"current_value: {current_value.shape}")

        #advantage = expected_value - current_value
        #critic_loss = (expected_value - current_value).pow(2).mean()
        advantage =  reward - current_value + gamma * next_value
        #print(f"current_value: {current_value}, expected_value: {expected_value}, advantage: {advantage}") 

        #print(f"critic_loss: {critic_loss.item()}")
        critic_loss = (advantage).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # # Update critic
        
        # # Update actor
        action_probs = self.actor(state)
        one_hot = F.one_hot(action, action_probs.shape[1])
        # print(f"one_hot: {one_hot}")
        # print(f"action_probs: {action_probs}")
        # print(f"action: {action}")
        #print(f"action_probs: {action_probs[action]}")

        selected_probs = action_probs * one_hot
        #print(f"selected_probs: {selected_probs}")
        #print(f"advantage: {advantage}")
        #print(f"advantage.shape: {advantage.shape}")
        actor_loss = (-(selected_probs) * advantage.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #print(f"critic_loss: {critic_loss.item()}, actor_loss: {actor_loss.item()}")
        return {
            "expected_value": expected_value.detach().numpy().tolist(),
            "value": current_value.detach().numpy().tolist(),
            "advantage": advantage.detach().numpy().tolist(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
        }

    def behavior_cloning_train(self, state, action, done):
        #state = state[0, :]
        #action = action[0]
        action_one_hot = F.one_hot(action, self.action_dim)
        for i in range(1):
        # Convert inputs to tensors (they should already be tensors from the DataLoader)
            action_probs = self.actor(state)  # state is already a batch tensor
        # Calculate cross entropy loss between predicted probabilities and true actions
        #actor_loss = F.cross_entropy(action_probs, action)  # action is already a batch tensor
        # print(f"action_probs: {action_probs}")
        # print(f"action_probs.shape: {action_probs.shape}")
        # print(f"action: {action}")
        # print(f"action.shape: {action.shape}")

        # print(f"action_probs.shape: {action_probs.shape}")
        # print(f"action_one_hot.shape: {action_one_hot.shape}")
        # print(f"action_probs: {action_probs}")
        # print(f"action: {action}")
        # print(f"action_one_hot: {action_one_hot}")
            selected_log_action_probs = -torch.log(action_probs) * action_one_hot
            #print(f"selected_log_action_probs: {selected_log_action_probs}")

            actor_loss = selected_log_action_probs.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_scheduler.step()

        #print(f"one step actor_loss: {actor_loss.item()}")
        return actor_loss.item()

