# Configurações gerais do jogo
SCREEN_WIDTH = 600        # Largura da tela em pixels
SCREEN_HEIGHT = 400       # Altura da tela em pixels
BLOCK_SIZE = 20           # Tamanho de cada bloco (usado para a cobra e a comida)
SNAKE_SPEED = 10          # Velocidade inicial da cobra (ajustável para experimentos)

# Cores em RGB
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Modo de exibição
SHOW_RENDER = True        # Se True, renderiza o jogo; se False, executa em segundo plano (útil para treinamento)

# Parâmetros de recompensa (para fácil ajuste durante experimentos)
REWARD_FOOD = 10          # Recompensa ao comer comida
PENALTY_DEATH = -10       # Penalidade ao colidir
STEP_REWARD = 0.1         # Recompensa mínima por cada passo (para incentivar exploração)
