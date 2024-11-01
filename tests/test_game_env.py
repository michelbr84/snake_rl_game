import unittest
import numpy as np
from game.game_env import SnakeEnv
from game.snake import SnakeGame
from game.config import SCREEN_WIDTH, SCREEN_HEIGHT  # Importação das dimensões da tela

class TestGameEnv(unittest.TestCase):
    def setUp(self):
        self.env = SnakeEnv()
        self.game = SnakeGame()

    def test_reset_environment(self):
        state = self.env.reset()
        self.assertEqual(len(state), 7)  # Verifica se o estado inicial tem o tamanho correto
        self.assertIsInstance(state, np.ndarray)

    def test_step_environment(self):
        self.env.reset()
        action = 0  # Suponha que a ação seja 'CIMA'
        state, reward, done, _ = self.env.step(action)
        self.assertEqual(len(state), 7)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)

    def test_game_initialization(self):
        # Verifica se o jogo é inicializado corretamente com a posição inicial da cobra e comida
        self.assertEqual(len(self.game.snake_pos), 4)  # Atualizado para refletir o comprimento inicial da cobra
        self.assertIsInstance(self.game.food_pos, list)

    def test_food_generation(self):
        # Garante que a comida é gerada dentro dos limites da tela
        food_pos = self.game.generate_food()
        self.assertTrue(0 <= food_pos[0] < SCREEN_WIDTH)
        self.assertTrue(0 <= food_pos[1] < SCREEN_HEIGHT)

    def test_collision_detection(self):
        # Testa a colisão com o próprio corpo
        self.game.snake_pos = [[100, 100], [90, 100], [80, 100]]
        self.game.snake_pos[0] = [90, 100]  # Posiciona a cabeça no corpo para simular colisão
        self.assertTrue(self.game.check_collision())

if __name__ == '__main__':
    unittest.main()
