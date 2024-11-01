import torch
import time
from game.game_env import SnakeEnv
from ai.agents.dqn_agent import DQNAgent

# Caminho para o modelo treinado
MODEL_PATH = "checkpoints/dqn/final_model.pth"

def load_trained_agent(state_size, action_size, model_path):
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    agent.model.load_state_dict(torch.load(model_path))
    agent.epsilon = 0  # Define epsilon para 0, para que o agente não explore e use apenas a política treinada
    print("Modelo carregado com sucesso!")
    return agent

def run_game_with_agent(agent, env):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        env.render()  # Renderiza o jogo para visualização
        action = agent.act(state)  # Ação baseada no estado atual
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

        # Tempo para deixar a execução visível
        time.sleep(0.1)

    print(f"Fim do jogo! Pontuação total: {total_reward}")

if __name__ == "__main__":
    # Inicializa o ambiente
    env = SnakeEnv()

    # Configurações do ambiente e agente
    STATE_SIZE = 7  # Tamanho do estado conforme definido no ambiente
    ACTION_SIZE = 4  # Cima, Baixo, Esquerda, Direita

    # Carrega o agente treinado
    agent = load_trained_agent(STATE_SIZE, ACTION_SIZE, MODEL_PATH)

    # Roda o jogo com o agente treinado
    run_game_with_agent(agent, env)

    # Encerra o ambiente ao final
    env.close()
