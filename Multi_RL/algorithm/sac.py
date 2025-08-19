# 当使用 from sac import * 从 sac.py 模块导入所有内容时，只有 ApproxContainer 和 SAC 这两个对象会被导入
__all__ = ["ApproxContainer", "SAC"] 

import torch
import torch.nn as nn

from Multi_RL.utils.common_utils import get_apprfunc_dict
from Multi_RL.create_pkg.create_apprfunc import create_apprfunc

from copy import deepcopy
from torch.optim import Adam

import math
from typing import Any, Optional, Tuple, Dict
from torch import Tensor
import time
from Multi_RL.utils.tensorboard_setup import tb_tags


class ApproxContainer(nn.Module):
    """
    定义SAC算法中使用的所有的函数估计器，包含 1 个策略函数和 2 个价值函数（2个Q值）
    ApproxBase 是基类。先尝试不使用基类编写函数估计器，直接继承 nn.Module 看是否可以（参考CleanRL的实现）
    """
    def __init__(self, **kwargs):
        # kwargs 是关键字参数，是字典。如果没有传入这个参数的话，kwargs 会被当成一个空字典
        
        # 继承自 nn.Module，初始化父类，后面能直接调用.parameters()方法
        #  parameters() 方法会自动递归收集所有子模块（q1、q2、policy 等）和直接定义的参数（log_alpha）
        super().__init__()

        # 获取 q 网络的参数信息：策略网络的网络类型（如mlp），以及每一层的层数、激活函数等
        q_args = get_apprfunc_dict("value", **kwargs)
        self.q1: nn.Module = create_apprfunc(**q_args)
        self.q2: nn.Module = create_apprfunc(**q_args)

        # 构建 2 个 q 网络的目标网络
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        # 将目标网络的梯度信息关闭。目标网络梯度不更新，手动更新目标网络的参数
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # 获取策略网络的参数信息，然后基于这些信息构造策略网络
        policy_args = get_apprfunc_dict("policy", **kwargs)
        # create_apprfunc 函数返回的是构造的对应的类的实例
        self.policy: nn.Module = create_apprfunc(**policy_args)

        # 构造熵系数
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))

        # 创建所有网络的优化器
        self.q1_optimizer = Adam(self.q1.parameters(), lr=kwargs["q_learning_rate"])
        self.q2_optimizer = Adam(self.q2.parameters(), lr=kwargs["q_learning_rate"])
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )
        self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs["alpha_learning_rate"])

    def create_action_distributions(self, logits):
        """
        根据策略网络输出的logits，创建动作分布类的实例对象
        因此返回的例如是 TanhGaussDistribution 类的一个实例对象
        """
        return self.policy.get_act_dist_cls(logits)
    

class SAC:
    """
    定义 SAC 算法类，暂时也不设置继承其他类
    在创建 SAC 类的实例时：SAC(**args)
    构造函数中的参数kwargs是传入的args排除了指定的键（例如这里的"gamma""tau"，都是在构造函数中指定的）的结果
    """
    def __init__(
        self,
        tau: float = 0.005,
        alpha: float = math.e,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
        **kwargs: Any,
    ):
        # 指定的参数主要是SAC算法的一些必要参数，除了这些参数外的其他参数是用于构造估计函数网络的参数
        self.networks = ApproxContainer(**kwargs)
        self.tau = tau

        self.n_steps = kwargs['n_steps']
        self.gamma = kwargs['gamma']

        # 对网络中的 log_alpha 初始化
        self.networks.log_alpha.data.fill_(math.log(alpha))
        self.auto_alpha = auto_alpha
        if target_entropy is None:
            target_entropy = -kwargs["action_dim"]
        self.target_entropy = target_entropy


    @property
    def adjustable_parameters(self):
        """
        定义一组可动态调整的超参数
        允许在训练过程中动态修改这些参数，而无需重新实例化整个算法。
        通过 @property 装饰器，adjustable_parameters 成为一个只读属性
        """
        return ("gamma", "tau", "alpha", "auto_alpha", "target_entropy")
    

    def model_update(self, data: Dict[str, Tensor]):
        """
        用于模型（2个q网络、1个policy网络、1个参数alpha）的更新
        data 是训练数据，是包含训练数据的字典
        """
        start_time = time.time()

        # 首先，更新 q 网络
        loss_q, q1, q2 = self._q_update(data)

        # 然后，更新 policy 网络
        # 在此之间，向 data 中添加有关熵的数据，因为后面更新alpha的时候还需要使用这些数据
        obs = data["obs"]
        logits = self.networks.policy(obs)
        act_dist = self.networks.create_action_distributions(logits)
        # 注意：new_act, new_logp 含有 policy 网络梯度的信息，因此可以直接使用这些数据来更新 policy 网络
        # 但是在更新参数 alpha 的时候，需要使用不含有梯度信息的 new_logp（因为此时只关注alpha的更新）
        new_act, new_logp = act_dist.rsample()
        data.update({"new_act": new_act, "new_logp": new_logp})
        # 更新 policy 网络
        loss_policy, entropy = self._policy_update(data)

        # 如果设置了自动更新温度系数，那就更新参数 alpha
        if self.auto_alpha:
            self._alpha_update(data)

        # 最后，目标网络更新
        self._target_update()

        # 记录并返回 Tensorboard 存储的相关信息
        tb_info = {
            "SAC/critic_q1-RL iter": q1.item(),
            "SAC/critic_q2-RL iter": q2.item(),
            "SAC/entropy-RL iter": entropy.item(),
            "SAC/alpha-RL iter": self._get_alpha(),
            tb_tags["loss_critic"]: loss_q.item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        }

        return tb_info

    
    def _get_alpha(self, requires_grad: bool = False):
        alpha = self.networks.log_alpha.exp()
        if requires_grad:
            return alpha
        else:
            return alpha.item()

        
    def _q_update(self, data: Dict[str, Tensor]):
        """
        用于根据 q 网络的损失函数更新 q 网络
        函数名前的单下划线 _ 是一种约定俗成的命名规范，表示该函数是类的内部方法（非公开接口）
        data（数据字典）中的数据说明
        obs：当前状态
        act：根据当前状态采样得到的实际动作
        rew：奖励，rew=-cost，cost表示代价，即当前状态与目标状态之间的距离
        注意：在其他算法中我们仍然使用rew（在DLAC中使用cost），只不过我们需要对环境的rew重新定义，定义为-cost
        使用rew而非cost，是为了和一些算法中的其他结构保持一致（例如最大化rew与最大化熵保持一致）
        obs2：下一状态，根据单步 TD 还是多步 TD 而有所不同。具体指 N 步之后的状态。但是在 sac 算法中就指1步后的状态
        done：在到达 obs2 之后，一个 episode 是否结束
        """
        obs, act, rew, obs2, done = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        
        # 计算当前(s,a)下q网络的值
        q1 = self.networks.q1(obs, act)
        q2 = self.networks.q2(obs, act)

        # 计算 TD target，不需要网络的梯度信息
        with torch.no_grad():
            # 对于随机policy，输出下一个状态生成动作的均值和标准差
            next_logits = self.networks.policy(obs2)
            next_act_dist = self.networks.create_action_distributions(next_logits)
            next_act, next_logp = next_act_dist.rsample()
            next_q1 = self.networks.q1_target(obs2, next_act)
            next_q2 = self.networks.q2_target(obs2, next_act)
            next_q = torch.min(next_q1, next_q2)
            print(obs.shape)
            print(rew.shape)
            # backup 表示的就是 TD target，在 sac 算法中，使用一步 TD 来更新Q
            # 在 DLAC 算法中，cost 指的就是多步累积代价，此时的TD target = cost + gamma^N Q
            gamma_n = self.gamma ** self.n_steps  # 需传入n_step参数
            backup = rew + (1 - done) * gamma_n * (next_q - self._get_alpha() * next_logp)

        # 计算 2 个 q 网络的损失函数
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # q 网络更新
        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        loss_q.backward()
        self.networks.q1_optimizer.step()
        self.networks.q2_optimizer.step()

        return loss_q, q1.detach().mean(), q2.detach().mean()

    
    def _policy_update(self, data: Dict[str, Tensor]):
        # 更新 policy 依赖于 q 值，目标是使 q 最大化。在更新policy之前，需要屏蔽q网络的梯度
        for p in self.networks.q1.parameters():
            p.requires_grad = False
        for p in self.networks.q2.parameters():
            p.requires_grad = False

        # 准备计算数据
        obs, new_act, new_logp = (
            data["obs"],
            data["new_act"],
            data["new_logp"],
        )

        # 计算 policy 网络的损失函数
        q1 = self.networks.q1(obs, new_act)
        q2 = self.networks.q2(obs, new_act)
        # 损失函数：熵最大化 + q值最大化
        loss_policy = (self._get_alpha() * new_logp - torch.min(q1, q2)).mean()

        # policy 网络更新
        self.networks.policy_optimizer.zero_grad()
        loss_policy.backward()
        self.networks.policy_optimizer.step()

        # 计算策略的熵，注意 new_logp 含有策略网络的梯度信息，在计算熵时需要将梯度信息略去
        entropy = -new_logp.detach().mean()

        # 更新完 policy 网络之后，需要把之前屏蔽了梯度的 q 网络解除
        for p in self.networks.q1.parameters():
            p.requires_grad = True
        for p in self.networks.q2.parameters():
            p.requires_grad = True

        return loss_policy, entropy


    def _alpha_update(self, data: Dict[str, Tensor]):
        # 获取 new_logp 数据用来计算熵
        new_logp = data["new_logp"]

        # 计算关于 alpha 的损失函数，注意要保留待更新参数 alpha 的梯度信息
        # 但是要切断 new_logp 的梯度信息
        alpha = self._get_alpha(True)
        loss_alpha = -alpha * (new_logp.detach() + self.target_entropy).mean()

        # alpha 更新
        self.networks.alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.networks.alpha_optimizer.step()


    def _target_update(self,):
        # 基于更新后的 q 网络参数更新目标 q 网络
        # 目标网络始终不需要梯度信息，因此将其置于屏蔽梯度的环境中
        with torch.no_grad():
            # 软更新（Soft Update），也称为Polyak 平均
            polyak = 1 - self.tau
            for p, p_targ in zip(
                self.networks.q1.parameters(), self.networks.q1_target.parameters()
            ):
                # mul_ 和 add_ 分别为原地乘法和原地加法
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
            for p, p_targ in zip(
                self.networks.q2.parameters(), self.networks.q2_target.parameters()
            ):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
