
# Snake RL Game

Este projeto é uma implementação do clássico jogo Snake com agentes de Aprendizado por Reforço (Reinforcement Learning) usando três algoritmos principais: **DQN**, **A2C** e **PPO**. O objetivo é treinar agentes que possam aprender a jogar Snake de forma autônoma, maximizando a pontuação ao consumir alimentos e evitando colisões.

### Estrutura do Projeto

```
snake_rl_game/
├── game/                  # Contém a lógica do jogo e a interface do ambiente RL
│   ├── snake.py           # Implementação do jogo Snake
│   ├── game_env.py        # Ambiente do Gym para integração com RL
│   └── config.py          # Configurações de parâmetros do jogo
├── ai/                    # Implementações dos agentes e modelos RL
│   ├── agents/            # Agentes para DQN, A2C e PPO
│   │   ├── dqn_agent.py   # Agente DQN
│   │   ├── a2c_agent.py   # Agente A2C
│   │   ├── ppo_agent.py   # Agente PPO
│   │   └── base_agent.py  # Classe base com métodos comuns
│   ├── models/            # Modelos de redes neurais dos agentes
│   ├── memory/            # Memória de replay para experiência do DQN
│       └── replay_memory.py
├── training/              # Scripts de treinamento e avaliação
│   ├── train_dqn.py       # Script para treinar o agente DQN
│   ├── train_a2c.py       # Script para treinar o agente A2C
│   ├── train_ppo.py       # Script para treinar o agente PPO
│   ├── evaluation.py      # Script para avaliar o agente treinado
│   └── utils.py           # Funções auxiliares para treinamento e avaliação
├── tests/                 # Testes para o ambiente, agentes e modelos
├── assets/                # Sprites e sons para o jogo
├── notebooks/             # Análises e visualização de dados de treinamento
├── logs/                  # Armazenamento dos logs de treinamento
├── checkpoints/           # Checkpoints e salvamento de modelos treinados
├── requirements.txt       # Dependências do projeto
├── setup.py               # Script de instalação do projeto
├── README.md              # Documentação do projeto
└── run_game.py            # Executa o jogo com o agente treinado
```

---

### Pré-requisitos

- **Python 3.8+**
- **Bibliotecas**:
  - `Pygame`
  - `Gym`
  - `Torch`
  - Outras dependências listadas no arquivo `requirements.txt`

### Instalação

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/snake_rl_game.git
   cd snake_rl_game
   ```

2. Crie um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Para Linux/Mac
   venv\Scripts\activate     # Para Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

### Configuração

- **Arquivo de Configuração (`game/config.py`)**: Parâmetros como tamanho da tela, velocidade da cobra, e recompensas podem ser ajustados no arquivo `config.py` para facilitar experimentos.
- **Variáveis de Hiperparâmetros**: Cada agente possui hiperparâmetros específicos (taxa de aprendizado, gama, epsilon, etc.), definidos nos scripts de treinamento em `training/`.

### Executando o Jogo

Para rodar o jogo com o agente treinado:

```bash
python run_game.py
```

Este comando executa o jogo Snake, onde o agente RL irá interagir com o ambiente em tempo real.

---

### Treinamento dos Agentes

Os scripts de treinamento para cada agente estão na pasta `training/`. Ao rodar um script, ele treina o agente correspondente, salva checkpoints e gera logs do progresso. 

#### DQN
Para treinar o agente **DQN**:
```bash
python training/train_dqn.py
```

#### A2C
Para treinar o agente **A2C**:
```bash
python training/train_a2c.py
```

#### PPO
Para treinar o agente **PPO**:
```bash
python training/train_ppo.py
```

> Cada um desses scripts salva checkpoints a cada 100 episódios e registra métricas de desempenho em arquivos `.log` e `.csv`.

---

### Avaliação dos Agentes

Após o treinamento, os agentes podem ser avaliados usando o script `evaluation.py`:

```bash
python training/evaluation.py
```

Este script carrega o modelo treinado e executa uma série de jogos para medir a performance, registrando métricas de desempenho.

### Notebooks de Análise

Na pasta `notebooks/`, há notebooks (`exploration_analysis.ipynb` e `reward_analysis.ipynb`) para explorar as recompensas e o comportamento do agente durante o treinamento. Eles permitem visualizar o equilíbrio entre exploração e exploração e o impacto das recompensas no aprendizado.

---

### Testes

A pasta `tests/` contém testes unitários para garantir a integridade do código:

- **test_game_env.py**: Testes para o ambiente do jogo.
- **test_agents.py**: Testes para os agentes (DQN, A2C, PPO).
- **test_models.py**: Testes para verificar a integridade das redes neurais.

Execute os testes com:
```bash
python -m unittest discover -s tests
```

---

### Componentes do Projeto

1. **game/snake.py**: Contém a lógica do jogo, incluindo detecção de colisão, movimento da cobra, e geração de comida.
2. **game/game_env.py**: Interface do ambiente Gym, definindo o espaço de ação, recompensas, e estados para integração com os agentes.
3. **ai/agents/**: Implementações dos agentes (DQN, A2C, PPO), com métodos para treinamento e seleção de ações.
4. **ai/models/**: Modelos de redes neurais específicas para cada agente.
5. **training/**: Scripts para treinamento e avaliação dos agentes, com checkpoints e registros de dados.
6. **tests/**: Testes unitários e de integração para o jogo, agentes e redes neurais.
7. **assets/**: Contém sprites e sons para o jogo, melhorando a experiência visual e sonora.
8. **checkpoints/**: Salva os checkpoints dos modelos treinados para retomar o treinamento.

---

### Melhorias Futuras

- **Aprimoramento de Hiperparâmetros**: Ajuste fino de hiperparâmetros para melhorar o desempenho dos agentes.
- **Implementação de Modelos Avançados**: Experimentação com outros algoritmos de RL como DDPG e SAC.
- **Visualizações Mais Detalhadas**: Adição de gráficos de métricas como perda de aprendizado e taxas de sucesso.

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
