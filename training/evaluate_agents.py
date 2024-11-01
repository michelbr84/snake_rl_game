import numpy as np
import torch
from game.game_env import SnakeEnv
from ai.agents.dqn_agent import DQNAgent
from ai.agents.a2c_agent import A2CAgent
from ai.agents.ppo_agent import PPOAgent

# Configuração
AGENT_TYPE = 'DQN'  # Alterar para 'A2C' ou 'PPO' para outros agentes
EPISODES = 100  # Número de episódios de avaliação

def evaluate_agent(agent, env, episodes=100):
    total_rewards = []
    steps = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step_count = 0

        while not done:
            action = agent.act(state)  # Executa a ação no estado atual
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            step_count += 1

        total_rewards.append(episode_reward)
        steps.append(step_count)

    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(steps)

    print(f"Avaliação de {episodes} episódios:")
    print(f"Recompensa Média: {avg_reward}")
    print(f"Passos Médios até a Colisão: {avg_steps}")

    return avg_reward, avg_steps

if __name__ == "__main__":
    env = SnakeEnv()

    # Carrega o agente de acordo com o tipo escolhido
    if AGENT_TYPE == 'DQN':
        agent = DQNAgent(state_size=7, action_size=4)
        agent.model.load_state_dict(torch.load('checkpoints/dqn/trained_model.pth'))
    elif AGENT_TYPE == 'A2C':
        agent = A2CAgent(state_size=7, action_size=4)
        agent.model.load_state_dict(torch.load('checkpoints/a2c/trained_model.pth'))
    elif AGENT_TYPE == 'PPO':
        agent = PPOAgent(state_size=7, action_size=4)
        agent.model.load_state_dict(torch.load('checkpoints/ppo/trained_model.pth'))
    
    # Avalia o agente
    evaluate_agent(agent, env, episodes=EPISODES)
    env.close()
