# 当使用 from mol import * 从 mol.py 模块导入所有内容时，只有 ApproxContainer 和 mol 这两个对象会被导入
__all__ = ["ApproxContainer", "MOL"] 

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
    定义MOL算法中使用的所有的函数估计器，包含 1 个策略函数和 2 个价值函数（2个Q值）
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

        # 获取 V 网络的参数信息：策略网络的网络类型（如mlp），以及每一层的层数、激活函数等
        V_args = get_apprfunc_dict("value", **kwargs)
        self.q1: nn.Module = create_apprfunc(**q_args)

        # 获取策略网络的参数信息，然后基于这些信息构造策略网络
        policy_args = get_apprfunc_dict("policy", **kwargs)
        # create_apprfunc 函数返回的是构造的对应的类的实例
        self.policy: nn.Module = create_apprfunc(**policy_args)


        # 创建所有网络的优化器
        self.q1_optimizer = Adam(self.q1.parameters(), lr=kwargs["q_learning_rate"])
        self.q2_optimizer = Adam(self.q2.parameters(), lr=kwargs["q_learning_rate"])
        self.V_optimizer = Adam(self.V.parameters(), lr=kwargs["V_learning_rate"])
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )


    def create_action_distributions(self, logits):
        """
        根据策略网络输出的logits，创建动作分布类的实例对象
        因此返回的例如是 TanhGaussDistribution 类的一个实例对象
        """
        return self.policy.get_act_dist_cls(logits)
    

class MOL:
    """
    定义 MOL 算法类
    在创建 MOL 类的实例时：MOL(**args)
    构造函数中的参数kwargs是传入的args排除了指定的键（例如这里的"gamma""tau"，都是在构造函数中指定的）的结果
    """
    def __init__(
        self,
        gamma: float = 0.9,
        lamda: float = 0.9,
        eps = 0.1,
        **kwargs: Any,
    ):
        # 指定的参数主要是MOL算法的一些必要参数，除了这些参数外的其他参数是用于构造估计函数网络的参数
        self.networks = ApproxContainer(**kwargs)
        self.n_steps = kwargs['n_steps']
        self.gamma = kwargs['gamma']
        self.eps = 0.1
        self.lamd = 0.9


    @property
    def adjustable_parameters(self):
        """
        定义一组可动态调整的超参数
        允许在训练过程中动态修改这些参数，而无需重新实例化整个算法。
        通过 @property 装饰器，adjustable_parameters 成为一个只读属性
        """
        return ("gamma", 'lambda')
    

    def model_update(self, data: Dict[str, Tensor]):
        """
        用于模型（2个q网络、一个V网络、1个policy网络）的更新
        data 是训练数据，是包含训练数据的字典
        """
        start_time = time.time()

        # 首先，更新 q 网络
        loss_q, q1, q2 = self._q_update(data)
        # 之后，更新 V 网络
        loss_V,v=self._V_update(data)
        # 然后，更新 policy 网络
        obs = data["obs"]
        act = data["act"]
        act = data["act"]
        old_logp = data["logp"]
        
        

        logits = self.networks.policy(obs)
        act_dist = self.networks.create_action_distributions(logits)
        current_logp = act_dist.log_prob(act)  
        ratio = torch.exp(current_logp - old_logp)

        # 更新 policy 网络
        loss_policy = self._policy_update(data)


        # 最后，目标网络更新
        self._target_update()

        # 记录并返回 Tensorboard 存储的相关信息
        tb_info = {
            "MOL/critic_q1-RL iter": q1.item(),
            "MOL/critic_q2-RL iter": q2.item(),


            tb_tags["loss_critic"]: loss_q.item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        }

        return tb_info

    

        
    def _q_update(self, data: Dict[str, Tensor]):
        """
        用于根据 q 网络的损失函数更新 q 网络
        函数名前的单下划线 _ 是一种约定俗成的命名规范，表示该函数是类的内部方法（非公开接口）
        data（数据字典）中的数据说明
        因为是考虑n步数据的学习，所以下面的数据都是n步的
        obs：当前状态(共n项，顺序排列)
        act：根据当前状态采样得到的实际动作(共n项，与状态一一对应)
        rew：奖励，rew=-cost，cost表示代价，即当前状态与目标状态之间的距离(共n项，与状态一一对应)
        注意：在其他算法中我们仍然使用rew（在DLAC中使用cost），只不过我们需要对环境的rew重新定义，定义为-cost
        使用rew而非cost，是为了和一些算法中的其他结构保持一致（例如最大化rew与最大化熵保持一致）
        obs2：下一状态(共n项，与状态一一对应)
        done：在达到n步之后的状态时，一个 episode 是否结束
        """
        obs, act, rew, obs2, old_logp, done = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["logp"],
            data["done"],
        )
        
        # 计算当前n步的(sn,an)下q网络的值
        q1 = self.networks.q1(obs, act)
        q2 = self.networks.q2(obs, act)
        (batch_size,nstep)=q1.shape

        #计算重要性采样
        logits = self.networks.policy(obs)
        act_dist = self.networks.create_action_distributions(logits)
        current_logp = act_dist.log_prob(act)  
        ratio = torch.exp(current_logp - old_logp)
        ratio_clipped = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
        # 计算 TD target
        with torch.no_grad():
            # 对于随机policy，输出下一个状态生成动作的均值和标准差
            next_logits = self.networks.policy(obs2)
            next_act_dist = self.networks.create_action_distributions(next_logits)
            next_act, next_logp = next_act_dist.log()
            next_q1 = self.networks.q1_target(obs2, next_act)
            next_q2 = self.networks.q2_target(obs2, next_act)
            next_q = torch.min(next_q1, next_q2)
           #计算n步的RQ算子
            backup=[]
            c=1
            for i in range(batch_size):
                r=q1[i,0]
                r+=c*(rew[i,0]+ self.gamma*next_q1[i,0]-q1[i,0])
                for j in range(nstep-1):
                    c*=self.lamda*self.gamma*ratio[i,j+1]
                    r+=c*(rew[i,j+1]+ self.gamma*next_q1[i,j+1]-q1[i,j+1])
                
                backup.append(r)
            backup = torch.tensor(backup)


            

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

    def _V_update(self, data: Dict[str, Tensor]):
        # 更新 V 依赖于 q 值，目标为3项。在更新V之前，需要屏蔽q网络的梯度
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

        # 计算 V 网络的损失函数
        q1 = self.networks.q1(obs, new_act)
        q2 = self.networks.q2(obs, new_act)
        # 损失函数1：
        loss_V = (self._get_alpha() * new_logp - torch.min(q1, q2)).mean()

        # policy 网络更新
        self.networks.V_optimizer.zero_grad()
        loss_V.backward()
        self.networks.V_optimizer.step()

        # 更新完 policy 网络之后，需要把之前屏蔽了梯度的 q 网络解除
        for p in self.networks.q1.parameters():
            p.requires_grad = True
        for p in self.networks.q2.parameters():
            p.requires_grad = True

        return loss_policy, entropy


    
    def _policy_update(self, data: Dict[str, Tensor]):
        # 更新 policy 依赖于 q 值，目标是使 。在更新policy之前，需要屏蔽q网络的梯度
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

        # 损失函数：
        loss_policy = 

        # policy 网络更新
        self.networks.policy_optimizer.zero_grad()
        loss_policy.backward()
        self.networks.policy_optimizer.step()


        # 更新完 policy 网络之后，需要把之前屏蔽了梯度的 q 网络解除
        for p in self.networks.q1.parameters():
            p.requires_grad = True
        for p in self.networks.q2.parameters():
            p.requires_grad = True

        return loss_policy, entropy


    def _alpha_update(self, data: Dict[str, Tensor]):

        new_logp = data["new_logp"]

        # 计算关于 alpha 的损失函数，注意要保留待更新参数 alpha 的梯度信息
        # 但是要切断 new_logp 的梯度信息
        alpha = self._get_alpha(True)
        loss_alpha = -alpha * (  )

        # alpha 更新
        self.networks.alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.networks.alpha_optimizer.step()

    def _beta_update(self, data: Dict[str, Tensor]):

        new_logp = data["new_logp"]

        # 计算关于 beta 的损失函数，注意要保留待更新参数 alpha 的梯度信息
        beta = self._get_beta(True)
        loss_beta = -beta * (   )

        # beta 更新
        self.networks.beta_optimizer.zero_grad()
        loss_beta.backward()
        self.networks.beta_optimizer.step()


    def _target_update(self,):
        # 基于更新后的 q 网络参数更新目标 q 网络(需要吗？)
        # 目标网络始终不需要梯度信息，因此将其置于屏蔽梯度的环境中
        