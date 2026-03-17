
# Snake RL Game

Este projeto é uma implementação do clássico jogo Snake com agentes de Aprendizado por Reforço (Reinforcement Learning) usando três algoritmos principais: **DQN**, **A2C** e **PPO**. O objetivo é treinar agentes que possam aprender a jogar Snake de forma autônoma, maximizando a pontuação ao consumir alimentos e evitando colisões.

### Estrutura do Projeto

```
snake_rl_game/
├── game/                  # Contém a lógica do jogo e a interface do ambiente RL
│   ├── snake.py           # Implementação do jogo Snake
│   ├── game_env.py        # Ambiente Gymnasium para integração com RL
│   └── config.py          # Configurações de parâmetros do jogo
├── ai/                    # Implementações dos agentes e modelos RL
│   ├── agents/            # Agentes para DQN, A2C e PPO
│   │   ├── dqn_agent.py   # Agente DQN
│   │   ├── a2c_agent.py   # Agente A2C
│   │   └── ppo_agent.py   # Agente PPO (com suporte GPU/CUDA)
│   ├── models/            # Modelos de redes neurais dos agentes
│   └── memory/            # Memória de replay para experiência do DQN
│       └── replay_memory.py
├── training/              # Scripts de treinamento e avaliação
│   ├── train_dqn.py       # Script para treinar o agente DQN
│   ├── train_a2c.py       # Script para treinar o agente A2C
│   ├── train_ppo.py       # Script para treinar o agente PPO
│   ├── run_ppo_gpu.py     # Script para treinar PPO headless com GPU
│   └── evaluate_agents.py # Script para avaliar agentes treinados
├── tests/                 # Testes para o ambiente, agentes e modelos
├── assets/                # Sprites e sons para o jogo
├── notebooks/             # Análises e visualização de dados de treinamento
├── checkpoints/           # Checkpoints e salvamento de modelos treinados
├── CLAUDE.md              # Instruções para Claude Code
├── requirements.txt       # Dependências do projeto
├── setup.py               # Script de instalação do projeto
├── README.md              # Documentação do projeto
└── run_game.py            # Executa o jogo com o agente treinado
```

---

### Pré-requisitos

- **Python 3.10+**
- **GPU com CUDA** (opcional, mas recomendado para treinamento)
- **Bibliotecas**:
  - `Pygame`
  - `Gymnasium` (substitui o antigo `gym`)
  - `Torch` (com CUDA para GPU)
  - Outras dependências listadas no arquivo `requirements.txt`

### Instalação

1. Clone este repositório:
   ```bash
   git clone https://github.com/michelbr84/snake_rl_game.git
   cd snake_rl_game
   ```

2. Crie um ambiente virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate       # Para Linux/Mac
   .venv\Scripts\activate          # Para Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. (Opcional) Para suporte GPU, instale o PyTorch com CUDA:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```

### Configuração

- **Arquivo de Configuração (`game/config.py`)**: Parâmetros como tamanho da tela, velocidade da cobra, e recompensas podem ser ajustados no arquivo `config.py` para facilitar experimentos.
- **Variáveis de Hiperparâmetros**: Cada agente possui hiperparâmetros específicos (taxa de aprendizado, gama, epsilon, etc.), definidos nos scripts de treinamento em `training/`.

### Estado do Ambiente (11 features)

O ambiente utiliza um estado inteligente com 11 features binárias/normalizadas:

| Feature | Descrição |
|---|---|
| `danger_straight` | Perigo em frente (parede ou corpo) |
| `danger_left` | Perigo à esquerda |
| `danger_right` | Perigo à direita |
| `dir_up/down/left/right` | Direção atual da cobra (one-hot) |
| `food_up/down/left/right` | Direção relativa da comida |

### Executando o Jogo

Para rodar o jogo com o agente treinado:

```bash
python run_game.py          # Usa PPO por padrão
python run_game.py ppo      # Especifica PPO
python run_game.py dqn      # Especifica DQN
```

---

### Treinamento dos Agentes

Os scripts de treinamento para cada agente estão na pasta `training/`. Ao rodar um script, ele treina o agente correspondente, salva checkpoints e gera logs do progresso.

#### DQN
```bash
python training/train_dqn.py
```

#### A2C
```bash
python training/train_a2c.py
```

#### PPO
```bash
python training/train_ppo.py
```

#### PPO com GPU (headless, recomendado)
```bash
python training/run_ppo_gpu.py
```

> Cada script salva checkpoints a cada 100 episódios e o modelo final em `checkpoints/<agente>/final_model.pth`. Os dados de treinamento são registrados em arquivos `.csv`.

---

### Avaliação dos Agentes

Após o treinamento, os agentes podem ser avaliados:

```bash
python training/evaluate_agents.py
```

### Testes

```bash
python -m pytest tests/ -v
```

---

### Componentes do Projeto

1. **game/snake.py**: Lógica do jogo, colisão, movimento da cobra, e geração de comida.
2. **game/game_env.py**: Interface Gymnasium com estado inteligente (11 features) e reward shaping.
3. **ai/agents/**: Implementações dos agentes (DQN, A2C, PPO com suporte GPU).
4. **ai/models/**: Redes neurais específicas para cada agente.
5. **training/**: Scripts de treinamento com checkpoints, resume, e logs CSV.
6. **tests/**: Testes unitários (pytest) para o jogo, agentes e modelos.
7. **assets/**: Sprites e sons para o jogo.
8. **checkpoints/**: Modelos treinados salvos.

---

### Melhorias Futuras

- **Aprimoramento de Hiperparâmetros**: Ajuste fino para melhorar o desempenho dos agentes.
- **Implementação de Modelos Avançados**: Experimentação com DDPG e SAC.
- **Visualizações Mais Detalhadas**: Gráficos de métricas como perda e taxas de sucesso.

---

### Contribuições

Este projeto é aberto a contribuições. Para contribuir:

1. Crie um fork do projeto.
2. Crie uma nova branch (`git checkout -b feature/sua-feature`).
3. Faça um commit com suas alterações (`git commit -am 'Adiciona nova feature'`).
4. Envie suas alterações para o repositório remoto (`git push origin feature/sua-feature`).
5. Abra um Pull Request.

---

### Licença

Este projeto é licenciado sob a licença MIT.
