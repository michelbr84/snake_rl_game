import random
from collections import deque

class ReplayMemory:
    def __init__(self, capacity):
        """
        Inicializa a memória de replay com uma capacidade máxima.
        :param capacity: Capacidade máxima da memória.
        """
        self.memory = deque(maxlen=capacity)

    def push(self, experience):
        """
        Armazena uma nova experiência na memória.
        :param experience: Tupla (estado, ação, recompensa, próximo_estado, terminal).
        """
        self.memory.append(experience)

    def sample(self, batch_size):
        """
        Retorna uma amostra aleatória de experiências da memória.
        :param batch_size: Tamanho do lote de amostras.
        :return: Lista de experiências aleatórias.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Retorna o número de experiências atualmente na memória.
        :return: Inteiro representando o tamanho atual da memória.
        """
        return len(self.memory)
