"""
在这个文件中会定义 OffSampler 类，继承自 base.py 模块中的 BaseSampler 类（基类，且是一个抽象类）
"""
from typing import List
from Multi_RL.trainer.sampler.base import BaseSampler, Experience

class OffSampler(BaseSampler):
    # sample_batch_size 和 noise_params 在构造 sampler 时传入的参数 kwargs 中就有
    # 因此这里不需要再额外指定
    def __init__(self, **kwargs):
        # 根据传入的参数，初始化基类中的一些属性信息
        super().__init__(**kwargs)


    # 在 BaseSampler 类中还定义了 sample 函数，这里是对内部采样函数 _sample 的定义
    def _sample(self,) -> List[Experience]:
        """
        与环境交互，获得采样的数据信息
        最后返回的是以 Experience 类的对象组成的列表
        """
        batch_data = []
        # horizon = sample_batch_size // num_envs，即每个环境与环境交互的步数
        for _ in range(self.horizon):
            # experiences 是以列表形式返回的，因此才能通过 .extend() 函数添加到列表 batch_data 中
            # batch_data 通过调用 extend 函数，将 experiences 里存储的所有experience类对象全都添加到了batch_data中
            experiences = self._step()
            # experiences 是一个列表（可迭代对象）
            # 使用 extend 会将 experiences 中的每个元素（即每个 Experience 实例）逐个添加到 batch_data 中
            batch_data.extend(experiences)
        return batch_data