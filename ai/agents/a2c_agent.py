import torch
import torch.nn as nn
import torch.optim as optim
from ..models.a2c_model import ActorCriticModel

class A2CAgent:
    def __init__(self, state_size, action_size, gamma=0.99, learning_rate=0.001):
        self.gamma = gamma
        self.model = ActorCriticModel(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        policy, _ = self.model(state)
        action = torch.distributions.Categorical(policy).sample()
        return action.item()

    def compute_loss(self, log_probs, values, rewards):
        rewards = torch.FloatTensor(rewards)
        Qvals = torch.zeros(len(rewards))
        for t in reversed(range(len(rewards))):
            Qvals[t] = rewards[t] + (self.gamma * Qvals[t + 1] if t + 1 < len(rewards) else 0)
        
        values = torch.cat(values)
        advantage = Qvals - values
        actor_loss = -(torch.stack(log_probs) * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        return actor_loss + critic_loss

    def learn(self, log_probs, values, rewards):
        loss = self.compute_loss(log_probs, values, rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
