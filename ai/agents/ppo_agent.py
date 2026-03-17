import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from ai.models.ppo_model import PPOActorCriticModel

class PPOAgent:
    def __init__(self, state_size, action_size, gamma=0.99, learning_rate=0.001, clip_epsilon=0.2, k_epochs=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.k_epochs = k_epochs
        self.model = PPOActorCriticModel(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        policy, value = self.model(state)
        dist = Categorical(policy)  # Cria a distribuição categórica para a política
        action = dist.sample()  # Amostra uma ação
        return action.item(), dist.log_prob(action), value  # Retorna a ação, log_prob e o valor estimado

    def compute_gae(self, rewards, values):
        values = list(values) + [0]
        gae = 0
        returns = []
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * gae
            returns.insert(0, gae + values[t])
        return returns

    def learn(self, states, actions, old_log_probs, rewards, values):
        returns = self.compute_gae(rewards, [v.item() for v in values])
        returns = torch.FloatTensor(returns).to(self.device)
        old_log_probs = torch.cat(old_log_probs).detach().to(self.device)
        values_t = torch.cat(values).detach().to(self.device)
        advantages = returns - values_t

        states_t = torch.stack(states).to(self.device)
        actions_t = torch.stack(actions).to(self.device)

        for _ in range(self.k_epochs):
            log_probs, state_values = [], []
            for i in range(len(states_t)):
                policy, value = self.model(states_t[i])
                dist = Categorical(policy)
                log_prob = dist.log_prob(actions_t[i])
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
