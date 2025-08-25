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
        self.batch_size = kwargs['replay_batch_size']
        self.gamma = kwargs['gamma']
        self.eps = 0.1
        self.lamda = 0.9


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

        data['obs']=data['obs']-self.target_value
        data['obs2']=data['obs2']-self.target_value

        # 首先，更新 q 网络
        loss_q, q1, q2 = self._q_update(data)
        # 之后，更新 V 网络
        loss_V,v=self._V_update(data)
        # 然后，更新 policy 网络

        obs = data["obs"]
        act = data["act"]
        old_logp = data["logp"]
        
        


        # 注意：new_act, new_logp 含有 policy 网络梯度的信息，因此可以直接使用这些数据来更新 policy 网络
        # 但是在更新参数 alpha 的时候，需要使用不含有梯度信息的 new_logp（因为此时只关注alpha的更新）
        
        logits = self.networks.policy(obs)
        act_dist = self.networks.create_action_distributions(logits)
        new_act, new_logp = act_dist.rsample()
        data.update({"new_act": new_act, "new_logp": new_logp})
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
        obs, act, rew, obs2, old_logp = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["logp"],
        )
        
        # 计算当前n步的(sn,an)下q网络的值
        q1 = self.networks.q1(obs, act)
        q2 = self.networks.q2(obs, act)

        q = torch.min(q1, q2)
        logits = self.networks.policy(obs)
        act_dist = self.networks.create_action_distributions(logits)
        current_logp = act_dist.log_prob(act)  
        ratio = torch.exp(current_logp - old_logp)
        ratio_clipped = torch.clamp(ratio, 0, 1 + self.eps)
        # 计算 TD target
        with torch.no_grad():
            # 计算下一个状态的期望
            next_logits = self.networks.policy(obs2)
            next_act_dist = self.networks.create_action_distributions(next_logits)
            next_q1=[]
            next_q2=[]
            for _ in range(10):
                next_act, logp = next_act_dist.sample()
                next1_q1 = self.networks.q1_target(obs2, next_act)
                next1_q2 = self.networks.q2_target(obs2, next_act)
                next_q1.append(next1_q1)
                next_q2.append(next1_q2)
            # 转换为张量并求期望
            next_q1 = torch.stack(next_q1)  
            next_q1=next_q1.mean(dim=0)
            next_q2 = torch.stack(next_q2)  
            next_q2=next_q1.mean(dim=0)
            next_q = torch.min(next_q1, next_q2)
           #计算n步的RQ算子
            backup=[]
            c=1
            for i in range(self.batch_size):
                r=q[i,0]
                for j in range(self.n_steps):
                    c*=self.lamda*self.gamma*ratio_clipped[i,j]
                    r+=c*(rew[i,j]+ self.gamma*next_q[i,j]-q[i,j])
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

        obs, act,  obs2, old_logp = (
            data["obs"],
            data["act"],
            data["obs2"],
            data["logp"],
        )
        
        v = self.networks.v(obs)
        v2 = self.networks.v(obs2)

        (batch_size,n_steps,obs_dim)=obs.shape

        # 损失函数1：
        zero_obs=torch.zeros(obs_dim)
        loss_V1 =self.networks.v(zero_obs)

        # 损失函数2：
        logits = self.networks.policy(obs)
        act_dist = self.networks.create_action_distributions(logits)
        current_logp = act_dist.log_prob(act)  
        ratio = torch.exp(current_logp - old_logp)
        ratio_clipped = torch.clamp(ratio, 0, 1 + self.eps)

        loss_V2=[]
        c=1
        for i in range(self.batch_size):
            J=0
            for j in range(self.n_steps):
                Jn=max(v[i,0]-(1-self.lamda**j)*v2[i,j],0)
                c*=ratio_clipped[i,j]
                J+=(1-self.lamda)*(self.lamda**j)*Jn
            J+=(self.lamda**self.n_steps)*Jn
            loss_V2.append(J)
        loss_V2 = torch.tensor(loss_V2)


        # V 网络更新
        loss_V = a * loss_V1 + b * loss_V2.mean()

        self.networks.V_optimizer.zero_grad()
        loss_V.backward()
        self.networks.V_optimizer.step()

        return loss_V, v.detach().mean()


    
    def _policy_update(self, data: Dict[str, Tensor]):
        # 更新 policy 依赖于 q 值，目标是使Q值最大化的同时满足熵条件和V函数限制条件 。在更新policy之前，需要屏蔽q网络的梯度
        for p in self.networks.q1.parameters():
            p.requires_grad = False
        for p in self.networks.q2.parameters():
            p.requires_grad = False

        # 准备计算数据
        obs, new_act, new_logp, act, obs2, logp= (
            data["obs"],
            data["new_act"],
            data["new_logp"],
            data["act"],
            data["obs2"],
            data["logp"],
        )
        
        obs1=obs[:,0,:]
        new_act1=new_act[:,0,:]
        # 计算 policy 网络的损失函数
        q1 = self.networks.q1(obs1, new_act1)
        q2 = self.networks.q2(obs1, new_act1)
        
        ratio2=torch.exp(new_logp - logp)
        v = self.networks.v(obs)
        v2 = self.networks.v(obs2)

        logits = self.networks.policy(obs)
        act_dist = self.networks.create_action_distributions(logits)
        current_logp = act_dist.log_prob(act)  
        ratio = torch.exp(current_logp - logp)
        ratio_clipped = torch.clamp(ratio, 0, 1 + self.eps)

        loss_V2=[]
        c=1
        for i in range(self.batch_size):
            J=0
            for j in range(self.n_steps):
                Jn=max(v[i,0]-(1-self.lamda**j)*v2[i,j],0)
                c*=ratio_clipped[i,j]
                J+=(1-self.lamda)*(self.lamda**j)*Jn
            J+=(self.lamda**self.n_steps)*Jn
            loss_V2.append(J)
        loss_V2 = torch.tensor(loss_V2)

        # 损失函数：
        loss_policy = (self.omega * new_logp + self.beta * ratio2 * loss_V2 - torch.min(q1, q2)).mean()

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


    def _omega_update(self, data: Dict[str, Tensor]):

        omega = omega + deta * 


    def _beta_update(self, data: Dict[str, Tensor]):

        new_logp = data["new_logp"]

        # 计算关于 beta 的损失函数，注意要保留待更新参数 beta 的梯度信息
        beta = beta + deta * 


        # beta 更新
        self.networks.beta_optimizer.zero_grad()
        loss_beta.backward()
        self.networks.beta_optimizer.step()


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