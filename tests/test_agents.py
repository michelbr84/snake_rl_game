import unittest
import numpy as np
from ai.agents.dqn_agent import DQNAgent
from ai.agents.a2c_agent import A2CAgent
from ai.agents.ppo_agent import PPOAgent

STATE_SIZE = 7
ACTION_SIZE = 4

class TestDQNAgent(unittest.TestCase):
    def setUp(self):
        self.agent = DQNAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)

    def test_dqn_initialization(self):
        self.assertIsNotNone(self.agent.model)
        self.assertIsNotNone(self.agent.target_model)

    def test_dqn_action_selection(self):
        state = np.zeros(STATE_SIZE)
        action = self.agent.act(state)
        self.assertIn(action, range(ACTION_SIZE))

    def test_dqn_learning_step(self):
        # Adiciona uma experiência e verifica se a memória armazena corretamente
        self.agent.remember(np.zeros(STATE_SIZE), 1, 1, np.ones(STATE_SIZE), False)
        self.assertEqual(len(self.agent.memory), 1)

class TestA2CAgent(unittest.TestCase):
    def setUp(self):
        self.agent = A2CAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)

    def test_a2c_initialization(self):
        self.assertIsNotNone(self.agent.model)

    def test_a2c_action_selection(self):
        state = np.zeros(STATE_SIZE)
        action = self.agent.act(state)
        self.assertIn(action, range(ACTION_SIZE))

class TestPPOAgent(unittest.TestCase):
    def setUp(self):
        self.agent = PPOAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE)

    def test_ppo_initialization(self):
        self.assertIsNotNone(self.agent.model)

    def test_ppo_action_selection(self):
        state = np.zeros(STATE_SIZE)
        action, _, _ = self.agent.act(state)
        self.assertIn(action, range(ACTION_SIZE))

if __name__ == '__main__':
    unittest.main()
