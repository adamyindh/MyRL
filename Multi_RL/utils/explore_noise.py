"""
定义随机噪声，用于探索
DDPG 和 TD3 算法中都需要使用探索噪声
"""

import numpy as np

class GaussNoise:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self, action):
        return action + np.random.normal(self.mean, self.std)