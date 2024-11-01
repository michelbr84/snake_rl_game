import torch
import numpy as np
import csv
import os
from ai.agents.dqn_agent import DQNAgent
from game.game_env import SnakeEnv

# Configurações do treinamento
EPISODES = 1000
STATE_SIZE = 7
ACTION_SIZE = 4
LOG_FILE = "dqn_training_data.csv"
CHECKPOINT_DIR = "checkpoints/dqn/"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")

# Função para salvar os dados de cada episódio em um arquivo CSV
def log_episode_data(filename, episode, reward, epsilon):
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Episode", "Reward", "Epsilon"])
        
        writer.writerow([episode, reward, epsilon])

# Função para salvar o checkpoint
def save_checkpoint(agent, episode, epsilon):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = {
        'model_state_dict': agent.model.state_dict(),
        'target_model_state_dict': agent.target_model.state_dict(),
        'epsilon': epsilon,
        'episode': episode
    }
    torch.save(checkpoint, CHECKPOINT_FILE)
    print(f"Checkpoint salvo no episódio {episode} em '{CHECKPOINT_FILE}'.")

# Função para carregar o checkpoint, se existir
def load_checkpoint(agent):
    if os.path.isfile(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        epsilon = checkpoint['epsilon']
        start_episode = checkpoint['episode'] + 1
        print(f"Checkpoint carregado. Retomando do episódio {start_episode} com epsilon={epsilon:.2f}")
        return start_episode, epsilon
    return 1, 1.0  # Começa do início, se não houver checkpoint

def train_dqn():
    env = SnakeEnv()
    agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    scores = []

    # Carrega o checkpoint, se existir
    start_episode, agent.epsilon = load_checkpoint(agent)

    for episode in range(start_episode, EPISODES + 1):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            state = next_state
            total_reward += reward

        # Atualiza a rede alvo periodicamente
        if episode % 10 == 0:
            agent.update_target_model()

        # Salva checkpoint periodicamente
        if episode % 100 == 0:
            save_checkpoint(agent, episode, agent.epsilon)

        # Registro dos dados do episódio no CSV
        log_episode_data(LOG_FILE, episode, total_reward, agent.epsilon)

        scores.append(total_reward)
        print(f"Episode {episode}/{EPISODES} - Reward: {total_reward} - Epsilon: {agent.epsilon:.2f}")

    # Salva o modelo DQN final
    torch.save(agent.model.state_dict(), os.path.join(CHECKPOINT_DIR, 'final_model.pth'))
    print(f"Modelo final salvo em '{CHECKPOINT_DIR}final_model.pth'.")

    env.close()

if __name__ == "__main__":
    train_dqn()
