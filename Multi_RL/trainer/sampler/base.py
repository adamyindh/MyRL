"""
创建 sampler 类的基类
"""

from abc import ABCMeta, abstractmethod
from collections import deque
from Multi_RL.utils.common_utils import get_env_type, get_env_id
from Multi_RL.create_pkg.create_envs import create_envs
from Multi_RL.create_pkg.create_alg import create_approx_contrainer
from Multi_RL.utils.explore_noise import GaussNoise

from typing import List, NamedTuple, Union, Tuple
import numpy as np
import torch

import time
from Multi_RL.utils.tensorboard_setup import tb_tags

class Experience(NamedTuple):
    """
    Experience 是一个使用 NamedTuple 定义的 ** 具名元组（Named Tuple）** 类。
    具名元组是 Python 中一种轻量级的数据结构，它继承自普通元组，但每个字段都有一个名称，使得数据的访问更加直观。
    不可变性：类似于普通元组，一旦创建，其属性值不可修改。
    结构化数据：每个实例包含强化学习中的一个经验样本，通常表示一个状态转换：(obs,actions,...,log_probs)
    属性访问：可以通过名称（如 exp.obs）或索引（如 exp[0]）访问字段
    """

    obs:np.ndarray
    action:np.ndarray
    rewards:np.ndarray
    next_obs:np.ndarray
    log_probs:np.ndarray
    done:np.ndarray




class BaseSampler(metaclass=ABCMeta):
    """
    Sampler类（包含OnSampler和OffSampler）的基类
    metaclass=ABCMeta 是用于定义 ** 抽象基类（Abstract Base Class, ABC）**的语法
    抽象基类是一种特殊的类，它不能被实例化，而是作为其他类的基类，强制子类实现特定的方法
    ABCMeta 的作用：Python 中抽象基类的元类（metaclass）。元类是创建类的类，Python 会使用元类来构建 BaseSampler 类
    通过指定 metaclass=ABCMeta，会有：
    1、BaseSampler 类是一个抽象基类，不能被直接实例化
    2、子类必须实现所有使用 @abstractmethod 装饰器标记的抽象方法，否则子类也会被视为抽象类，无法实例化

    有个问题：BaseSampler 是个抽象基类，那么定义其子类 OffSampler 的实例时，参数怎么传入到 BaseSampler 中？
    """
    def __init__(
        self,
        # sample_batch_size 在初始脚本中就初始化了这个参数的值
        # 在 init_args.py 中将其赋值给了 args["batch_size_per_sampler"]
        **kwargs,
    ):
        # kwargs["env_name"]的结果类似于："gym_HalfCheetah-v4"
        self.env_id = get_env_id(kwargs["env_name"])

        # 环境的数量，用于辨别是否为向量化的环境
        self.num_envs = kwargs["env_num"]

        # 根据 env_id 创建环境，用于与环境交互
        # 初始脚本里创建了 envs，应该是为了获取环境中的一些参数信息，例如动作的上下限等
        # 通过 create_envs 函数已经创建出了环境并对环境的动作空间设置了随机种子
        # 因此不再参考 GOPS 里对创建出的环境设置随机种子：env.seed(seed)
        # 另外注意，创建环境的过程参考 CleanRL，可能创建多个环境，因此统一用 envs 来表示环境
        # 暂时使用 1 个环境，后续用到多个环境的话，再添加
        self.envs = create_envs(**kwargs)

        # 创建函数估计器（主要包括q网络和policy网络）
        self.networks = create_approx_contrainer(**kwargs)

        self.sample_batch_size = kwargs["sample_batch_size"]
        self.action_type = kwargs["action_type"]
        self.reward_scale = kwargs["reward_scale"]

        # 采样时与环境交互的步数，
        self.horizon = self.sample_batch_size // self.num_envs

        # 是否在策略网络上添加噪声
        self.noise_params = kwargs["noise_params"]
        # 如果不为 None，那就设置噪声的类型
        if self.noise_params is not None:
            if self.action_type == "continu":
                # 传入的 noise_params 也是一个字典，包含噪声的 mean 和 std
                self.noise_processor = GaussNoise(**self.noise_params)
            else:
                raise RuntimeError("Only continuous action space is supported!")
            
        # 目标状态（目标值）
        self.target_value = kwargs["target_value"]
            
        # 记录总的采样数
        self.total_sample_number = 0

        # CleanRL 中，在使用 reset 函数的时候设置了随机种子
        # 这里使用的 seed 正是创建 self.envs 时动作空间中设置的随机种子，即 kwargs["env_seed"]
        # 注意：初始脚本中的 env_seed 使用了 2 次，一次是在创建 self.envs 中，设置动作空间的随机种子
        # 另一次就是在这里，对 reset 函数使用随机种子，但暂时必不知道这样设置随机种子有什么作用
        # 对于最新的HalfCheetah-v4环境，reset函数返回的是一个二元组(obs, info)，但是v4版本的info是空字典（参考GPT的回答）
        # 注意这里没有给出 self.info 的信息（但 GOPS 的原代码中给出了）
        # 使用reset设置seed创建并锁定了本环境专用的np.random.Generator，例如初始化随机状态
        # 简单来说，应该是关于reset的随机生成都由seed控制，但是action_space的随机生成动作由另外一个随机种子控制
        self.obs, _ = self.envs.reset(seed=kwargs["env_seed"])
        # # 设置 self.obs 的类型为 float32
        self.obs = np.float32(self.obs)
            
        # 在Sampler类中初始化n步缓冲区
        # n_steps为多步方法的步数，gamma为折扣因子
        self.n_steps = kwargs['n_steps']
        self.gamma = kwargs['gamma']
        self.n_step_buffers = [deque(maxlen=self.n_steps) for _ in range(self.num_envs)]  # 每个环境一个缓冲区
        self.tol_state=deque(maxlen=self.n_steps)
        self.tol_action=deque(maxlen=self.n_steps) 
        self.tol_next_state=deque(maxlen=self.n_steps)
        self.tol_done=deque(maxlen=self.n_steps)
        self.tol_rewards=deque(maxlen=self.n_steps)
        self.tol_log_probs=deque(maxlen=self.n_steps)


    def get_total_sample_num(self) -> int:
        return self.total_sample_number
    

    def get_total_sample_number(self) -> int:
        return self.total_sample_number
    

    def load_state_dict(self, state_dict):
        """
        加载模型的参数。self.networks 对象是一个包含多个网络的容器
        从保存的状态字典中恢复模型的参数，使得模型可以继续之前的训练，或者在新的环境中使用预训练的权重。
        调用load_state_dict函数，最终会得到一个包含所有子网络参数的字典，结构类似于：
        {
            'q1.weight': tensor(...),  # 第一个Q网络的权重
            'q1.bias': tensor(...),    # 第一个Q网络的偏置
            'q2.weight': tensor(...),  # 第二个Q网络的权重
            'q2.bias': tensor(...),    # 第二个Q网络的偏置
            'policy.weight': tensor(...),  # 策略网络的权重
            'policy.bias': tensor(...),    # 策略网络的偏置
        }
        """
        self.networks.load_state_dict(state_dict)


    # def _step(self,) -> List[Experience]:
    #     """
    #     返回的是一个由 Experience 对象组成的列表
    #     返回多个经验样本：函数通常执行一步或多步环境交互，收集多个状态转换并封装为 Experience 对象
    #     批量处理：列表中的每个元素对应一个时间步的经验，适用于批量训练或并行环境
    #     """
    #     # 如果是仅有 1 个环境
    #     # （在我们创建的环境中）根据测试，不管是单个环境（num_envs=1）还是多个并行的环境（num_envs=5）
    #     # 通过 reset 函数得到的 obs 的 shape 都是：[num_envs, obs_dim]
    #     # 随机策略网络生成的 logits 的 shape 是：[num_envs, 2 * act_dim]
    #     # 根据 logits 生成的动作的 shape 是：[num_envs, 1 * act_dim]，对应的 log_prob 的 shape 是：[num_envs]
    #     # 把 num_envs 视为 batch_size 也是可行的，因为 num_envs 本质上就决定了生成的 obs 的数量
    #     # GOPS 代码里对于单环境，obs 和 act 可能没有第 0 个关于批次的维度，因此需要使用 expand_dims
    #     # 因此，我们这里只需要参考 GOPS 关于多环境的代码即可，因为我们创建的env生成的obs个act自带第 0 个batch_size维度
    #     # 因此我们不需要像 GOPS 那样，分 num_envs 为 1 还是 5 来讨论
    #     obs_tensor = torch.from_numpy(self.obs)

    #     # 根据环境状态，获得动作的分布，然后根据动作采样动作
    #     logits = self.networks.policy(obs_tensor)
    #     action_distribution = self.networks.create_action_distributions(logits)
    #     actions, log_probs = action_distribution.sample()

    #     # 屏蔽 action 和 logp 的梯度，并转为 NumPy 类型的变量，确保数据类型为 float32
    #     actions = actions.detach().numpy().astype("float32")
    #     log_probs = log_probs.detach().numpy().astype("float32")

    #     # 如果考虑探索噪声，对动作再加上干扰
    #     if self.noise_params is not None:
    #         actions = self.noise_processor.sample(actions)

    #     # 如果是连续的空间，就对动作裁剪，使其在合理的上下限范围内
    #     if self.action_type == "continu":
    #         actions_clip = actions.clip(
    #             self.envs.action_space.low, self.envs.action_space.high
    #         )
    #     else:
    #         actions_clip = actions

    #     # 然后就是与环境交互，获得反馈的数据
    #     # 我们这里与环境交互的代码仍然保持与 CleanRL 一致，加"s"表示可能有多个环境
    #     # 下面这部分代码参考 CleanRL 的交互过程（因为 gym 环境是最新的）
    #     next_obs, rewards, terminations, truncations, infos = self.envs.step(actions_clip)

    #     # 将 next_obs, rewards 转化为 float32 类型的
    #     next_obs = np.float32(next_obs)
    #     rewards = np.float32(rewards)

    #     # 真实的下一个时刻的状态（可能会遇到 episode 结束的情况，next_obs 不准确）
    #     real_next_obs = next_obs.copy()
    #     for idx, trunc in enumerate(truncations):
    #         if trunc:
    #             real_next_obs[idx] = infos["final_observation"][idx]

    #     # 目前只是计算单步的奖励，在实现 DLAC 算法的时候，需要计算多步奖励【N步TD】
    #     # 根据不同的环境设置不同的 costs 以及对应的奖励，shape=[batch_size,]
    #     if self.env_id == "HalfCheetah-v4":
    #         costs = ((real_next_obs[:, 8] - self.target_value) ** 2) * self.reward_scale
    #         # 注意：costs 和 rewards 之间始终保持“负值”关系，因此后续仅可通过 rewards 来确定 costs
    #         rewards = -costs
    #     else:
    #         # 对于其他环境，先不重新设置 rewards
    #         rewards = rewards * self.reward_scale
        
    #     # 根据 terminations 和 truncations 确定 dones，shape=[batch_size,]
    #     dones = np.logical_or(terminations, truncations)

    #     # 最终想存储 (s,a,r,s',done,log_probs) 这 6 元组，存储 log_probs 因为涉及到熵
    #     # 分为单环境和多环境存储，多环境的obs会涉及多个状态，需要形成for循环，将每一个元素单独放在一个Experience对象里
    #     experiences = []
    #     for i in range(0, self.num_envs):
    #         experience = Experience(
    #             obs=self.obs.copy()[i],  # shape=[obs_dim]
    #             actions=actions_clip[i],  # shape=[act_dim]
    #             rewards=rewards[i],  # float 类型的变量
    #             next_obs=next_obs.copy()[i],  # shape=[obs_dim]
    #             dones=dones[i],  # bool 类型的变量
    #             log_probs=log_probs[i],  # float 类型的变量
    #         )
    #         # extend()：将可迭代对象（如列表、元组等）中的所有元素逐个添加到列表末尾
    #         # append(x)：将 x 作为单个元素添加到列表末尾
    #         # 使用 append 将 experience 作为单个元素添加到列表中，确保列表中的每个元素都是一个完整的 Experience 实例
    #         experiences.append(experience)

    #     # 更新状态
    #     self.obs = next_obs

    #     # 返回形式：将单个经验包装成一个列表，列表的长度为 self.num_envs
    #     return experiences

    @abstractmethod
    def _sample(self) -> Union[List[Experience], dict]:
        """
        Union 表示多态返回值，可以是 两种类型中的任意一种：由 Experience 对象组成的列表 or 一个字典
        _sample 是抽象函数，强调子类提供具体的实现，确保所有子类遵循统一的接口规范
        子类必须实现该方法，否则子类也会被视为抽象类，无法实例化
        不同的采样器（如OffSampler、OnSampler等）有完全不同的采样策略，都需要提供_sample()方法
        需要根据实际的采样方式具体定义 _sample 函数
        """
        pass


    # 修改_step为_nstep，累积n步经验
    def _nstep(self) -> List[Experience]:
    # 原_step代码：获取当前步的obs, action, reward, next_obs, done
        obs_tensor = torch.from_numpy(self.obs)
        logits = self.networks.policy(obs_tensor)
        action_distribution = self.networks.create_action_distributions(logits)
        actions, log_probs = action_distribution.sample()
        actions = actions.detach().numpy().astype("float32")
        log_probs = log_probs.detach().numpy().astype("float32")
        if self.noise_params is not None:
            actions = self.noise_processor.sample(actions)
        if self.action_type == "continu":
            actions_clip = actions.clip(
                self.envs.action_space.low, self.envs.action_space.high
            )
        else:
            actions_clip = actions
        next_obs, rewards, terminations, truncations, infos = self.envs.step(actions_clip)
        next_obs = np.float32(next_obs)
        rewards = np.float32(rewards)
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        if self.env_id == "HalfCheetah-v4":
            costs = ((real_next_obs[:, 8] - self.target_value) ** 2) * self.reward_scale
            rewards = -costs
        else:
            rewards = rewards * self.reward_scale
        dones = np.logical_or(terminations, truncations)

        experiences = []
        for i in range(self.num_envs):
            # 存储当前步的完整信息（包含所有需要记录的字段）
            self.n_step_buffers[i].append({
                'state': self.obs.copy()[i],          # 当前步状态
                'action': actions_clip[i], # 当前步动作
                'rewards': rewards[i],     # 当前步奖励
                'next_obs':next_obs.copy()[i],  #下一步状态
                'log_probs': log_probs[i],  # 采取当前步动作的概率
                'done': dones[i]          # 当前步是否为最终状态
            })
            
            
            self.tol_state.append(self.obs.copy())
            self.tol_action.append(actions_clip)
            self.tol_rewards.append(rewards)
            self.tol_next_state.append(next_obs.copy())
            self.tol_log_probs.append(log_probs)
            self.tol_done.append(dones)

            buffer = self.n_step_buffers[i]
            # 满足生成n步经验的条件（缓冲区满或终止）
            if len(buffer) == self.n_steps or (buffer and buffer[-1]['done']):

                # 2. 确定n步后的状态和最终终止标志
                final_step = buffer[-1]
                n_step_done = final_step['done']
                
 
                experience = Experience(
                    obs=self.tol_state,
                    action=self.tol_action,
                    rewards=self.tol_rewards,
                    next_obs=self.tol_next_state,
                    log_probs=self.tol_log_probs,
                    done=self.tol_done            
                )
                # 终止时清空缓冲区（避免跨回合污染）

                experiences.append(experience)

                if n_step_done:
                    self.n_step_buffers[i].clear()
        
                

        # 更新状态
        self.obs = next_obs


        return experiences

    
    def sample(self) -> Tuple[Union[List[Experience], dict], dict]:
        self.total_sample_number += self.sample_batch_size
        tb_info = dict()

        # time.perf_counter()是高精度计时器（通常纳秒级）
        start_time = time.perf_counter()

        # 根据不同定义的 sampler （在BaseSampler的子类中定义具体的方法）采样
        # 返回的 data 是列表，里面是 experience 对象
        data = self._sample()

        end_time = time.perf_counter()

        # 截止到目前，使用 tb_tags 存储日志信息的模块有：算法模块（sac.py）以及采样模块（off_sampler.py）
        tb_info[tb_tags["sampler_time"]] = (end_time - start_time) * 1000
        return data, tb_info
