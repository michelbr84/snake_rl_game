import torch
import torch.nn as nn

class ActorCriticModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCriticModel, self).__init__()
        
        # Camadas compartilhadas
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Cabeça do ator (policy head)
        self.policy_head = nn.Linear(128, action_size)
        
        # Cabeça do crítico (value head)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Cálculo da política e valor
        policy = torch.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x)
        
        return policy, value
