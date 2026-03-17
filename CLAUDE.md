# Snake RL Game - Claude Code Instructions

## Project Overview
Jogo Snake com agentes de Reinforcement Learning (DQN, A2C, PPO) usando Pygame e Gymnasium.

## Tech Stack
- **Python 3.10+** com ambiente virtual `.venv`
- **Gymnasium** (migrado de gym) para interface RL
- **PyTorch** para redes neurais dos agentes
- **Pygame** para renderização do jogo
- **pytest** para testes

## Project Structure
```
game/           → Lógica do jogo (snake.py, game_env.py, config.py)
ai/agents/      → Agentes RL (dqn_agent.py, a2c_agent.py, ppo_agent.py)
ai/models/      → Redes neurais (dqn_model.py, a2c_model.py, ppo_model.py)
ai/memory/      → Replay buffer (replay_memory.py)
training/       → Scripts de treinamento e avaliação
tests/          → Testes unitários (pytest)
checkpoints/    → Modelos treinados salvos
assets/         → Sprites e sons
notebooks/      → Análises Jupyter
```

## Key Conventions

### Environment (game_env.py)
- Gymnasium API: `reset()` → `(obs, info)`, `step()` → `(obs, reward, terminated, truncated, info)`
- State shape: 7 features `[head_x, head_y, food_x, food_y, screen_w, screen_h, snake_len]`
- Action space: Discrete(4) → UP=0, DOWN=1, LEFT=2, RIGHT=3
- Rewards definidos em `config.py`: REWARD_FOOD=10, PENALTY_DEATH=-10, STEP_REWARD=0.1

### Code Style
- Python PEP 8
- Comentários em português (pt-BR)
- Arquivos UTF-8
- Type hints quando apropriado

### Commands
```bash
# Ativar ambiente virtual
.venv/Scripts/activate          # Windows
source .venv/bin/activate       # Linux/Mac

# Rodar o jogo com agente treinado
python run_game.py

# Treinar agentes
python training/train_dqn.py
python training/train_a2c.py
python training/train_ppo.py

# Avaliar agentes
python training/evaluate_agents.py

# Rodar testes
python -m pytest tests/ -v

# TensorBoard
tensorboard --logdir logs/
```

## Rules

### Non-negotiable
- NUNCA commitar arquivos `.env`, secrets ou credenciais
- NUNCA commitar arquivos `__pycache__/` ou `.pyc`
- NUNCA modificar checkpoints treinados sem confirmação do usuário
- SEMPRE rodar `pytest` após modificar código Python
- SEMPRE usar Gymnasium (NÃO gym que está deprecated)
- SEMPRE manter compatibilidade da API do environment (state shape, action space)

### Testing
- Testes em `tests/` usando pytest
- Testar agentes: inicialização, seleção de ação, learning step
- Testar ambiente: reset, step, colisão, geração de comida
- Testar modelos: output shapes das redes neurais

### Training
- Checkpoints salvos a cada 100 episódios em `checkpoints/`
- Métricas registradas em CSV e logs
- `config.py` centraliza parâmetros ajustáveis
- SHOW_RENDER=False para treinamento headless
