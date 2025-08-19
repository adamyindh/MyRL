"""
创建 replay buffer 用于收集与环境交互得到的采样数据
是 trainer 的组成部分之一，trainer 由 sampler（收集数据） 和 buffer（存储数据） 组成
"""

import numpy as np
import sys
import torch
from collections import deque

__all__ = ["ReplayBuffer"]

def combined_shape(length: int, shape=None):
    """
    创建一个元组，表示组合后的数组形状（shape）
    输入参数：
    length：表示第一个维度的大小（如批量大小、缓冲区长度等）。
    shape：表示剩余维度的形状（可以是整数、元组或 None）。
    输出值：
    如果 shape 是 None，返回 (length,)。
    如果 shape 是标量（如整数），返回 (length, shape)。
    如果 shape 是元组，返回 (length, *shape)（将元组展开为多个维度）

    例如：输入(100, 4) --> 返回(100, 4)
    输入(100, (4, 84, 84)) --> 返回(10000, 4, 84, 84)
    """
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


# class ReplayBuffer:
#     """
#     在 replay_buffer 中通过均匀采样获得样本
#     """
#     def __init__(self, **kwargs):
#         self.obsv_dim = kwargs["obsv_dim"]
#         self.act_dim = kwargs["action_dim"]
#         self.max_size = kwargs["buffer_max_size"]

#         # 存储数据的字典，存储数据的是experience对象，里面的数据有：(s,a,r,s',done,log_prob)
#         # 构建用于存储数据的 NumPy 数组
#         self.buf = {
#             "obs": np.zeros(
#                 combined_shape(self.max_size, self.obsv_dim), dtype=np.float32
#             ),
#             "act": np.zeros(
#                 combined_shape(self.max_size, self.act_dim), dtype=np.float32
#             ),
#             "rew": np.zeros(self.max_size, dtype=np.float32),
#             "obs2": np.zeros(
#                 combined_shape(self.max_size, self.obsv_dim), dtype=np.float32
#             ),
#             "done": np.zeros(self.max_size, dtype=np.float32),
#             "logp": np.zeros(self.max_size, dtype=np.float32),
#         }

#         # 这 2 个属性的作用暂时还未知
#         self.ptr, self.size, = (0, 0,)



class ReplayBuffer:
    """
    在 replay_buffer 中通过均匀采样获得样本
    """
    def __init__(self, **kwargs):
        self.obsv_dim = kwargs["obsv_dim"]
        self.act_dim = kwargs["action_dim"]
        self.max_size = kwargs["buffer_max_size"]
        self.n_steps=kwargs["n_steps"]

        # 存储数据的字典，存储数据的是experience对象，里面的数据有：(s,a,r,s',done,log_prob)*n
        # 构建用于存储数据的 NumPy 数组
        self.buf = {
            "obs": np.zeros(
                combined_shape(self.max_size, (self.n_steps,self.obsv_dim)), dtype=np.float32
            ),
            "act": np.zeros(
                combined_shape(self.max_size,(self.n_steps,self.act_dim)), dtype=np.float32
            ),
            "rew": np.zeros((self.max_size,self.n_steps), dtype=np.float32),
            "obs2": np.zeros(
                combined_shape(self.max_size, (self.n_steps,self.obsv_dim)), dtype=np.float32
            ),
            "done": np.zeros((self.max_size,self.n_steps), dtype=np.float32),
            "logp": np.zeros((self.max_size,self.n_steps), dtype=np.float32),
        }

        # 这 2 个属性的作用暂时还未知
        self.ptr, self.size, = (0, 0,)


    def __len__(self):
        # 获取当前 replay buffer 的大小（存储了多少个数据）
        return self.size

    def __get_RAM__(self):
        """
        计算当前回放缓冲区占用的内存大小（单位：MB）
        sys.getsizeof 是 Python 内置函数，用于返回对象占用的内存字节数。
        例如：self.buf 是一个形状为 (1000000, 4) 的 float32 数组，完整内存占用为 16 MB（1,000,000 * 4 * 4 字节）
        self.size：当前缓冲区中已存储的样本数量（实际使用量）
        最后除以1000000：将字节数转换为 MB（1 MB = 1,000,000 字节）。
        self.size / self.max_size 是当前数据所占的比例
        """
        return int(sys.getsizeof(self.buf)) * self.size / (self.max_size * 1000000)
    
    # def store(
    #     self,
    #     obs: np.ndarray,
    #     act: np.ndarray,
    #     rew: np.ndarray,
    #     next_obs: np.ndarray,
    #     done: np.ndarray,
    #     logp: np.ndarray,
    # ) -> None:
    #     """
    #     向 replay_buffer 中存储数据的函数，目前只存储 6 种数据，也就是 sampler 获取的数据
    #     但是 sampler 对象调用 sample 函数返回的是列表形式的 experience 对象
    #     但注意，这里存储数据到表格中，传入 store 中的数据的 shape 应该是没有 batch_size 维度的
    #     因为 self.buf["obs"] 是一个指定了 batch_size 维度的 NumPy 数组，self.ptr 确定了数组的 index
    #     所以 self.buf["obs"][self.ptr] 相当于为传入参数提前规划好了“空间”
    #     对于 obs（状态向量），传入的数据一定是一个 shape=[obs_dim] 的向量，只有一个维度
    #     对于 rew（标量），传入的数据一定是一个具体地float值，例如1.0
    #     """
    #     self.buf["obs"][self.ptr] = obs
    #     self.buf["act"][self.ptr] = act
    #     self.buf["rew"][self.ptr] = rew
    #     self.buf["obs2"][self.ptr] = next_obs
    #     self.buf["done"][self.ptr] = done
    #     self.buf["logp"][self.ptr] = logp

    #     # self.ptr 就是self.buf["obs"]（或其他定义好的NumPy数组）的索引
    #     # 设置self.ptr是为了能够（在有max_size的限制下）及时更新self.buf["obs"]中的数据
    #     self.ptr = (self.ptr + 1) % self.max_size

    #     # 存入一个数据后，更新 replay buffer 的大小
    #     self.size = min(self.size + 1, self.max_size)

    def store(
        self,
        obs: deque,  # 明确参数类型为 deque
        act: deque,
        rew: deque,
        next_obs: deque,
        logp: deque,
        done: deque,

        ) -> None:
        """处理 deque 类型的n步数据，转换为 NumPy 数组后存储"""
        # 1. 将 deque 转换为 NumPy 数组（核心步骤）
        # 转换 obs: deque([obs1, obs2, ..., obsn]) → (n_steps, obsv_dim) 的二维数组
        obs_np = np.squeeze(np.array(obs))  # 先转为列表，再转为数组
        act_np = np.squeeze(np.array(act))
        rew_np = np.squeeze(np.array(rew))
        next_obs_np = np.squeeze(np.array(next_obs))
        done_np = np.squeeze(np.array(done))
        logp_np = np.squeeze(np.array(logp))


        # 3. 存储转换后的 NumPy 数组
        self.buf["obs"][self.ptr] = obs_np
        self.buf["act"][self.ptr] = act_np
        self.buf["rew"][self.ptr] = rew_np
        self.buf["obs2"][self.ptr] = next_obs_np
        self.buf["done"][self.ptr] = done_np
        self.buf["logp"][self.ptr] = logp_np

        # 4. 更新指针和大小（不变）
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, samples: list) -> None:
        """
        函数作用：将批量样本逐个添加到回放缓冲区（放到缓存区主要是self.store函数起的作用）中
        lambda 是匿名函数，传入的参数是sample，返回的结果是：self.store(*sample)
        *sample 是解包操作，将 sample 元组展开为多个参数，*sample=(obs,act,...,logp)
        map函数：对 samples 列表中的每个元素应用 lambda 函数
        对每个samples列表中的 sample（是 Experience 对象），调用 self.store(*sample)
        list(...)：将 map 生成的迭代器强制转换为列表，由于 self.store 返回为None，因此最终是：[None, None, ...]
        这里使用 list 的原因：在 Python 3 中，map 返回的是惰性迭代器，
        必须强制消费才能触发所有 self.store 调用。如果省略 list()，代码不会实际执行存储操作！

        # 新添加的解释
        但是要注意，samples里面的sample（是 Experience 对象）中，每一个元素值（例如obs，rew）都不应该有batch_size的维度
        即：obs=[1,2,...,17], rew=1.0
        因此，sampler 类中关于 sample 存储的函数还需要修正，确保sample中的每一个元素值都不应该有batch_size的维度

        对于单环境，直接在创建Experience对象的时候，令obs=self.obs[0].copy()即可（列表里面只有1个元素）
        对于多环境，可能要返回由多个Experience对象组成的列表
        """
        # 原 GOPS 的代码为：
        # list(map(lambda sample: self.store(*sample), samples))

        # 原来的代码复杂且难懂，修改得到的等价代码
        for sample in samples:
            self.store(*sample)


    def sample_batch(self, batch_size: int) -> dict:
        # 在训练的时候，从 replay buffer 中采样批量的数据用于训练
        # idx 表示从 [0, self.size-1] 中随机选取的 batch_size 个索引
        # 例如：np.random.randint(0, 10, 5) = array([6, 0, 9, 3, 1])
        idxes = np.random.randint(0, self.size, size=batch_size)

        batch = {}

        # k 表示 self.buf 中的键，例如"obs", "rew"
        # v 表示 self.buf 中这些键的具体取值，例如self.buf["obs"]是一个shape=[max_size, obs_dim]的NumPy数组
        for k, v in self.buf.items():
            if isinstance(v, np.ndarray):
                # 获取 v 中对应 idxes 的取值，并且存放到字典 batch 中，键是 k
                batch[k] = torch.as_tensor(v[idxes], dtype=torch.float32)
            else:
                batch[k] = v[idxes].array2tensor()

        # 最终返回的是字典，键保持和self.buf一致，值是从中随机（均匀）采样得到的
        return batch