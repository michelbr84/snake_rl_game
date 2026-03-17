import torch
import time
import sys
import os
from game.game_env import SnakeEnv
from ai.agents.dqn_agent import DQNAgent
from ai.agents.ppo_agent import PPOAgent

# Configurações
STATE_SIZE = 11
ACTION_SIZE = 4

# Modelos disponíveis (final_model, checkpoint, AgentClass)
MODELS = {
    'dqn': ('checkpoints/dqn/final_model.pth', 'checkpoints/dqn/checkpoint.pth', DQNAgent),
    'ppo': ('checkpoints/ppo/final_model.pth', 'checkpoints/ppo/checkpoint.pth', PPOAgent),
}

def load_trained_agent(agent_type='ppo'):
    if agent_type not in MODELS:
        print(f"Agente '{agent_type}' não encontrado. Opções: {list(MODELS.keys())}")
        sys.exit(1)

    final_path, checkpoint_path, AgentClass = MODELS[agent_type]
    agent = AgentClass(state_size=STATE_SIZE, action_size=ACTION_SIZE)

    # Tenta carregar final_model primeiro, depois checkpoint
    if os.path.isfile(final_path):
        agent.model.load_state_dict(torch.load(final_path, map_location='cpu'))
        print(f"Modelo final {agent_type.upper()} carregado!")
    elif os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        episode = checkpoint.get('episode', '?')
        print(f"Checkpoint {agent_type.upper()} carregado (episódio {episode})")
    else:
        print(f"Nenhum modelo encontrado para '{agent_type}'. Treine primeiro com:")
        print(f"  python training/train_{agent_type}.py")
        sys.exit(1)

    # Desativa exploração para usar apenas a política treinada
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0

    return agent

def run_game_with_agent(agent, env):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        env.render()
        if isinstance(agent, PPOAgent):
            action, _, _ = agent.act(state)
        else:
            action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        state = next_state
        time.sleep(0.1)

    print(f"Fim do jogo! Pontuação total: {total_reward}")

if __name__ == "__main__":
    # Escolhe o agente via argumento de linha de comando (padrão: ppo)
    agent_type = sys.argv[1] if len(sys.argv) > 1 else 'ppo'

    env = SnakeEnv()
    agent = load_trained_agent(agent_type)
    run_game_with_agent(agent, env)
    env.close()
