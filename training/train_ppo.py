import torch
import numpy as np
import csv
import os
from ai.agents.ppo_agent import PPOAgent
from game.game_env import SnakeEnv

# Configurações do treinamento
EPISODES = 1000
STATE_SIZE = 11
ACTION_SIZE = 4
LOG_FILE = "ppo_training_data.csv"
CHECKPOINT_DIR = "checkpoints/ppo/"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")

# Função para salvar os dados de cada episódio em um arquivo CSV
def log_episode_data(filename, episode, reward):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([episode, reward])

# Função para salvar o checkpoint
def save_checkpoint(agent, episode):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = {
        'model_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'episode': episode
    }
    torch.save(checkpoint, CHECKPOINT_FILE)
    print(f"Checkpoint salvo no episódio {episode}")

# Função para carregar o checkpoint, se existir
def load_checkpoint(agent):
    if os.path.isfile(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=agent.device)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode'] + 1
        print(f"Checkpoint carregado. Retomando do episódio {start_episode}")
        return start_episode
    return 0

def train_ppo():
    env = SnakeEnv()
    agent = PPOAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    scores = []

    # Carrega o checkpoint, se existir
    start_episode = load_checkpoint(agent)

    # Cria o cabeçalho do arquivo CSV apenas se começando do zero
    if start_episode == 0:
        with open(LOG_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward"])

    for episode in range(start_episode, EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False
        log_probs, values, rewards, states, actions = [], [], [], [], []

        while not done:
            action, log_prob, value = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

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

        # Salva checkpoint periodicamente
        if (episode + 1) % 100 == 0:
            save_checkpoint(agent, episode)

        # Monitoramento de métricas
        scores.append(total_reward)
        print(f"Episode {episode}/{EPISODES} - Reward: {total_reward}")

    # Salva o modelo PPO final
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(agent.model.state_dict(), os.path.join(CHECKPOINT_DIR, 'final_model.pth'))
    print(f"Modelo final salvo em '{CHECKPOINT_DIR}final_model.pth'")

    env.close()

if __name__ == "__main__":
    train_ppo()
