import torch
import time
import sys
from game.game_env import SnakeEnv
from ai.agents.dqn_agent import DQNAgent
from ai.agents.ppo_agent import PPOAgent

# Configurações
STATE_SIZE = 11
ACTION_SIZE = 4

# Modelos disponíveis
MODELS = {
    'dqn': ('checkpoints/dqn/final_model.pth', DQNAgent),
    'ppo': ('checkpoints/ppo/final_model.pth', PPOAgent),
}

def load_trained_agent(agent_type='ppo'):
    if agent_type not in MODELS:
        print(f"Agente '{agent_type}' não encontrado. Opções: {list(MODELS.keys())}")
        sys.exit(1)

    model_path, AgentClass = MODELS[agent_type]
    agent = AgentClass(state_size=STATE_SIZE, action_size=ACTION_SIZE)

    try:
        agent.model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except FileNotFoundError:
        print(f"Modelo não encontrado em '{model_path}'. Treine primeiro com:")
        print(f"  python training/train_{agent_type}.py")
        sys.exit(1)

    # Desativa exploração para usar apenas a política treinada
    if hasattr(agent, 'epsilon'):
        agent.epsilon = 0

    print(f"Modelo {agent_type.upper()} carregado com sucesso!")
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
