import pygame
import gym
from gym import spaces
import numpy as np
from .snake import SnakeGame
from .config import SCREEN_WIDTH, SCREEN_HEIGHT, REWARD_FOOD, PENALTY_DEATH, STEP_REWARD

class SnakeEnv(gym.Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()

        # Configuração do ambiente SnakeGame
        self.game = SnakeGame()
        
        # Definindo o espaço de ação: 4 ações (Cima, Baixo, Esquerda, Direita)
        self.action_space = spaces.Discrete(4)
        
        # Espaço de observação: Estado inclui posição da cabeça, posição da comida e tamanho da cobra
        self.observation_space = spaces.Box(
            low=0, 
            high=max(SCREEN_WIDTH, SCREEN_HEIGHT), 
            shape=(7,),  # Ajustado para 7 elementos no estado
            dtype=np.int32
        )

    def reset(self):
        # Reinicia o jogo e retorna o estado inicial
        self.game.reset_game()
        return self._get_state()

    def step(self, action):
        # Mapeia o índice de ação para uma direção do jogo
        directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.game.change_to = directions[action]
        
        # Avança o jogo por um passo
        game_over = self.game.play_step()
        
        # Calcula a recompensa
        reward = self._calculate_reward(game_over)
        
        # Obtém o novo estado
        state = self._get_state()
        
        # Define a flag `done`
        done = game_over

        return state, reward, done, {}

    def _get_state(self):
        # Estado definido pela posição da cabeça, posição da comida, e comprimento da cobra
        head_x, head_y = self.game.snake_pos[0]
        food_x, food_y = self.game.food_pos
        length_of_snake = len(self.game.snake_pos)
        
        # Retorna o estado em forma de array
        return np.array([head_x, head_y, food_x, food_y, SCREEN_WIDTH, SCREEN_HEIGHT, length_of_snake], dtype=np.int32)

    def _calculate_reward(self, game_over):
        reward = 0
        
        # Recompensa positiva ao comer a comida
        if self.game.snake_pos[0] == self.game.food_pos:
            reward += REWARD_FOOD
        
        # Penalidade ao bater nas bordas ou no corpo
        if game_over:
            reward += PENALTY_DEATH
        
        # Recompensa mínima para encorajar a cobra a explorar
        reward += STEP_REWARD
        
        return reward

    def render(self, mode='human'):
        # Renderiza o jogo se estiver no modo 'human'
        if mode == 'human':
            self.game.draw_elements()
            pygame.display.flip()

    def close(self):
        # Encerra o ambiente e o jogo
        pygame.quit()
