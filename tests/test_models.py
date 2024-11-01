import unittest
import torch
from ai.models.dqn_model import DQNModel
from ai.models.a2c_model import ActorCriticModel
from ai.models.ppo_model import PPOActorCriticModel

STATE_SIZE = 7
ACTION_SIZE = 4

class TestDQNModel(unittest.TestCase):
    def setUp(self):
        self.model = DQNModel(state_size=STATE_SIZE, action_size=ACTION_SIZE)

    def test_dqn_output_shape(self):
        state = torch.zeros(STATE_SIZE)
        q_values = self.model(state)
        self.assertEqual(q_values.shape, (ACTION_SIZE,))

class TestA2CModel(unittest.TestCase):
    def setUp(self):
        self.model = ActorCriticModel(state_size=STATE_SIZE, action_size=ACTION_SIZE)

    def test_a2c_output_shapes(self):
        state = torch.zeros(STATE_SIZE)
        policy, value = self.model(state)
        self.assertEqual(policy.shape, (ACTION_SIZE,))
        self.assertEqual(value.shape, (1,))

class TestPPOModel(unittest.TestCase):
    def setUp(self):
        self.model = PPOActorCriticModel(state_size=STATE_SIZE, action_size=ACTION_SIZE)

    def test_ppo_output_shapes(self):
        state = torch.zeros(STATE_SIZE)
        policy, value = self.model(state)
        self.assertEqual(policy.shape, (ACTION_SIZE,))
        self.assertEqual(value.shape, (1,))

if __name__ == '__main__':
    unittest.main()
