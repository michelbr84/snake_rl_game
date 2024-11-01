import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from ai.models.ppo_model import PPOActorCriticModel

class PPOAgent:
    def __init__(self, state_size, action_size, gamma=0.99, learning_rate=0.001, clip_epsilon=0.2, k_epochs=4):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.k_epochs = k_epochs
        self.model = PPOActorCriticModel(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        policy, value = self.model(state)
        dist = Categorical(policy)  # Cria a distribuição categórica para a política
        action = dist.sample()  # Amostra uma ação
        return action.item(), dist.log_prob(action), value  # Retorna a ação, log_prob e o valor estimado

    def compute_gae(self, rewards, values):
        values = values + [0]  # Inclui um valor final zero para simplificar o cálculo do GAE
        gae = 0
        returns = []
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * gae
            returns.insert(0, gae + values[t])
        return returns

    def learn(self, states, actions, old_log_probs, rewards, values):
        returns = self.compute_gae(rewards, values)
        returns = torch.cat([r.unsqueeze(0) for r in returns]).detach()
        old_log_probs = torch.cat(old_log_probs).detach()
        values = torch.cat(values).detach()
        advantages = returns - values

        for _ in range(self.k_epochs):
            log_probs, state_values = [], []
            for i, state in enumerate(states):
                policy, value = self.model(state)
                dist = Categorical(policy)
                log_prob = dist.log_prob(actions[i])
                log_probs.append(log_prob)
                state_values.append(value)

            log_probs = torch.stack(log_probs)
            state_values = torch.stack(state_values).squeeze()

            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - state_values).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
