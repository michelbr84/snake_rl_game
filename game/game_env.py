import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .snake import SnakeGame
from .config import SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE, REWARD_FOOD, PENALTY_DEATH, STEP_REWARD

class SnakeEnv(gym.Env):
    # Estado com 11 features: perigos (3), direção (4), comida relativa (4)
    STATE_SIZE = 11

    def __init__(self):
        super(SnakeEnv, self).__init__()

        self.game = SnakeGame()
        self.action_space = spaces.Discrete(4)  # UP=0, DOWN=1, LEFT=2, RIGHT=3

        # Estado normalizado entre 0 e 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.STATE_SIZE,),
            dtype=np.float32
        )
        self._prev_distance = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.game.reset_game()
        self._prev_distance = self._distance_to_food()
        return self._get_state(), {}

    def step(self, action):
        directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.game.change_to = directions[action]

        game_over = self.game.play_step()
        reward = self._calculate_reward(game_over)
        state = self._get_state()

        terminated = game_over
        truncated = False

        return state, reward, terminated, truncated, {}

    def _get_state(self):
        head_x, head_y = self.game.snake_pos[0]
        food_x, food_y = self.game.food_pos
        direction = self.game.snake_direction
        body = self.game.snake_pos

        # Pontos adjacentes à cabeça
        point_up = [head_x, head_y - BLOCK_SIZE]
        point_down = [head_x, head_y + BLOCK_SIZE]
        point_left = [head_x - BLOCK_SIZE, head_y]
        point_right = [head_x + BLOCK_SIZE, head_y]

        # Perigo em frente, à esquerda e à direita (relativo à direção atual)
        if direction == "UP":
            danger_straight = self._is_collision(point_up)
            danger_left = self._is_collision(point_left)
            danger_right = self._is_collision(point_right)
        elif direction == "DOWN":
            danger_straight = self._is_collision(point_down)
            danger_left = self._is_collision(point_right)
            danger_right = self._is_collision(point_left)
        elif direction == "LEFT":
            danger_straight = self._is_collision(point_left)
            danger_left = self._is_collision(point_down)
            danger_right = self._is_collision(point_up)
        else:  # RIGHT
            danger_straight = self._is_collision(point_right)
            danger_left = self._is_collision(point_up)
            danger_right = self._is_collision(point_down)

        # Direção atual (one-hot)
        dir_up = direction == "UP"
        dir_down = direction == "DOWN"
        dir_left = direction == "LEFT"
        dir_right = direction == "RIGHT"

        # Comida relativa à cabeça
        food_up = food_y < head_y
        food_down = food_y > head_y
        food_left = food_x < head_x
        food_right = food_x > head_x

        state = np.array([
            danger_straight, danger_left, danger_right,
            dir_up, dir_down, dir_left, dir_right,
            food_up, food_down, food_left, food_right
        ], dtype=np.float32)

        return state

    def _is_collision(self, point):
        x, y = point
        # Borda
        if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT:
            return True
        # Corpo
        if point in self.game.snake_pos[1:]:
            return True
        return False

    def _distance_to_food(self):
        head_x, head_y = self.game.snake_pos[0]
        food_x, food_y = self.game.food_pos
        return abs(head_x - food_x) + abs(head_y - food_y)

    def _calculate_reward(self, game_over):
        if game_over:
            return PENALTY_DEATH

        # Comeu comida
        if self.game.snake_pos[0] == self.game.food_pos:
            self._prev_distance = self._distance_to_food()
            return REWARD_FOOD

        # Reward shaping: recompensa por se aproximar da comida
        curr_distance = self._distance_to_food()
        if curr_distance < self._prev_distance:
            reward = STEP_REWARD  # Aproximou
        else:
            reward = -STEP_REWARD  # Afastou
        self._prev_distance = curr_distance

        return reward

    def render(self, mode='human'):
        # Renderiza o jogo se estiver no modo 'human'
        if mode == 'human':
            self.game.draw_elements()
            pygame.display.flip()

    def close(self):
        # Encerra o ambiente e o jogo
        pygame.quit()
