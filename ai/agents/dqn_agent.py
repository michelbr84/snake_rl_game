import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ai.memory.replay_memory import ReplayMemory  # Importação ajustada para refletir a estrutura do projeto
from ai.models.dqn_model import DQNModel

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, batch_size=64, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  # Fator de desconto
        self.epsilon = epsilon  # Taxa de exploração inicial
        self.epsilon_min = epsilon_min  # Taxa mínima de exploração
        self.epsilon_decay = epsilon_decay  # Decaimento da taxa de exploração
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_size)  # Inicializa a memória de replay

        # Modelos DQN
        self.model = DQNModel(state_size, action_size)
        self.target_model = DQNModel(state_size, action_size)
        self.update_target_model()  # Sincroniza o modelo alvo com o modelo principal

        # Otimizador
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def update_target_model(self):
        """Atualiza o modelo alvo copiando os pesos do modelo principal."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Armazena uma experiência na memória de replay."""
        self.memory.push((state, action, reward, next_state, done))

    def act(self, state):
        """Escolhe uma ação usando uma política ε-greedy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Escolhe ação aleatória
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values, dim=1).item()  # Ação com maior valor Q

    def replay(self):
        """Treina o modelo usando uma amostra de experiências armazenadas."""
        if len(self.memory) < self.batch_size:
            return
        minibatch = self.memory.sample(self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            reward = torch.tensor(reward)
            target = reward + self.gamma * torch.max(self.target_model(next_state)).detach() * (1 - done)
            target_f = self.model(state)
            target_f[action] = target

            # Calcula e aplica o gradiente
            loss = nn.MSELoss()(self.model(state), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Decai a taxa de exploração
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
