import torch
import numpy as np
import csv
from ai.agents.ppo_agent import PPOAgent
from game.game_env import SnakeEnv

# Configurações do treinamento
EPISODES = 1000
STATE_SIZE = 7
ACTION_SIZE = 4
LOG_FILE = "ppo_training_data.csv"

# Função para salvar os dados de cada episódio em um arquivo CSV
def log_episode_data(filename, episode, reward):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode, reward])

def train_ppo():
    env = SnakeEnv()
    agent = PPOAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    scores = []

    # Cria o cabeçalho do arquivo CSV
    with open(LOG_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Reward"])

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        log_probs, values, rewards, states, actions = [], [], [], [], []

        while not done:
            action, log_prob, value = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            states.append(torch.FloatTensor(state))
            actions.append(torch.tensor(action))
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)

            state = next_state
            total_reward += reward

        # Aprendizado PPO com clipping e múltiplas iterações
        agent.learn(states, actions, log_probs, rewards, values)

        # Registro dos dados do episódio no CSV
        log_episode_data(LOG_FILE, episode, total_reward)

        # Monitoramento de métricas
        scores.append(total_reward)
        print(f"Episode {episode}/{EPISODES} - Reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    train_ppo()
