"""
基于 create_apprfunc 函数创建，这里面的类都是 create_apprfunc.py 模块中注册表的"name"键的值
且对应着参数初始脚本中，关于apprfunc（包括值函数和policy）的name，例如 ActionValue、StochaPolicy 等
"""
"""
当执行 from mlp import * 时，只有 __all__ 列表中指定的这五个类会被导入到当前命名空间中
没有写在 __all__ 列表中的其他对象，如 mlp 函数、torch、nn 等，将不会被导入，除非 __all__ 列表未定义
如果 __all__ 列表未定义，from mlp import * 会导入所有【不以下划线（_）开头】的全局名称
"""
__all__ = [
    "ActionValue",
    "StateValue",
    "ActionValueDistri",
    "StochaPolicy",
    "DetermPolicy"
]

import torch
import torch.nn as nn
import warnings
from Multi_RL.utils.common_utils import get_activation_func
from Multi_RL.utils.act_distribution_cls import Action_Distribution_Cls

# 转置层：实现X^T * X操作
class TransposeMultiply(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # x的形状通常为(batch_size, feature_dim)
        # 计算X^T * X，结果形状为(batch_size, feature_dim, feature_dim)
        return torch.bmm(x.unsqueeze(1), x.unsqueeze(2)).squeeze(1)
        # 说明：
        # 1. unsqueeze(1)将x从(batch, d)变为(batch, 1, d)
        # 2. unsqueeze(2)将x从(batch, d)变为(batch, d, 1)
        # 3. bmm进行批量矩阵乘法，得到(batch, 1, 1)
        # 4. 最终输出形状为(batch, 1)


def positive_mlp(sizes, activation, output_activation=nn.Identity):
    """
    修改后的MLP网络，最后一层为X^T * X操作
    sizes：每一层输入的维度，例如：[obs_dim + act_dim, 64, 64]
           注意：最后一层的输出维度由倒数第二层的维度决定
    activation：隐藏层激活函数
    output_activation：已弃用，因为最后一层固定为X^T * X
    """
    layers = []
    # 构建隐藏层（不包含最后一层）
    for j in range(len(sizes) - 2):  # 只循环到倒数第二层
        layers += [nn.Linear(sizes[j], sizes[j + 1]), activation()]
    
    # 添加倒数第二层（最后一个线性层）
    layers += [nn.Linear(sizes[-2], sizes[-1]), activation()]
    
    # 添加自定义的输出层：X^T * X
    layers += [TransposeMultiply()]
    
    return nn.Sequential(*layers)




def mlp(sizes, activation, output_activation=nn.Identity):
    """
    根据隐藏层数+隐藏层的大小、隐藏层激活函数、输出层激活函数形成 mlp 网络
    sizes：每一层输入的维度，其形式例如：[obs_dim + act_dim, 64, 64, 1]（包含了层数及层的大小）
    activation：隐藏层激活函数的类型（激活函数的类型统一），取值例如：nn.Relu
    output_activation：输出层激活函数的类型，取值例如：nn.Identity（恒等函数，不对最后一层的输出做改变直接输出）
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    # * 是解包操作，将列表layers中的参数逐一解包，传递给 nn.Sequential() 构造函数
    return nn.Sequential(*layers)

#################### 值函数类型：动作值函数、状态值函数等 ####################
class ActionValue(nn.Module,):
    """
    原来的代码中，还将 Action_Distribution 作为基类继承下来
    但是目前不清楚这个基类的具体作用
    """

    def __init__(self, **kwargs):
        # kwargs 本质上来源于 utils.common_utils 中的 get_apprfunc_dict 函数
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        # 构造 q 函数的网络
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )

    def forward(self, obs, act):
        # 将状态、动作拼接起来，直接输入到 q 网络中产生输出
        q = self.q(torch.cat([obs, act], dim=-1))
        # 将 2 维的张量转化为 1 维张量输出
        return torch.squeeze(q, -1)
    

class StateValue(nn.Module,):
    """
    状态值函数的类
    输入状态数据，直接输出状态值
    """
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.v = mlp(
            [obs_dim] + list(hidden_sizes) + [1],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )

    def forward(self, obs):
        v = self.v(obs)
        return torch.squeeze(v, -1)
    

class ActionValueDistri(nn.Module,):
    """
    （连续）动作值分布类
    网络表示的是 q 值的分布，不同于 q 网络，输出的是 q 函数的均值与标准差
    """
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [2],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        # if "min_log_std"  in kwargs or "max_log_std" in kwargs:
        #     warnings.warn("min_log_std and max_log_std are deprecated in ActionValueDistri.")

    def forward(self, obs, act):
        logits = self.q(torch.cat([obs, act], dim=-1))
        value_mean, value_std = torch.chunk(logits, chunks=2, dim=-1)
        # 对标准差处理，确保标准差一定为正。softplus(x)=log(1+e^x)，是Relu的一种平滑，取值严格大于0
        # softplus 相当于对分布 q 网络输出结果的最后加了一层激活函数
        value_std = torch.nn.functional.softplus(value_std) 
        
        # 返回时将均值和标准差拼接起来，因此shape=[batch_size, 2*action_dim]
        return torch.cat((value_mean, value_std), dim=-1)
    
#################### 策略函数类型 ####################
# 随机策略
class StochaPolicy(nn.Module, Action_Distribution_Cls):
    """
    随机策略类，根据策略的均值与标准差随机生成动作
    暂时没有继承除 nn.Module 外的其他类（GOPS中继承了其他的类）
    """
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]

        # 暂时不考虑标准差的类型（原来的 get_apprfunc_dict 函数里默认其为"mlp_shared"）
        # 即策略的均值和标准差由同一个mlp网络输出。输出的维度：act_dim * 2
        # self.std_type = kwargs["std_type"] = "mlp_shared"
        self.policy = mlp(
            [obs_dim] + list(hidden_sizes) + [act_dim * 2],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )

        self.min_log_std = kwargs["min_log_std"]
        self.max_log_std = kwargs["max_log_std"]

        # register_buffer：将张量（动作空间的上下限（act_high_lim 和 act_low_lim））注册为模块缓冲区（buffer）
        # 由 nn.Parameter 包装的张量，会被自动添加到模型的 parameters() 迭代器中，参与梯度计算与优化器更新
        # 由 register_buffer 注册的张量，不会被视为模型参数，但会作为模型状态的一部分被保存和加载
        # 使用 model.state_dict() 保存模型时，缓冲区会被自动包含在内。在加载模型时，动作空间的上下限也会被正确恢复
        # 这里的 act_high_lim 和 act_low_lim 仍然是 StochaPolicy 类自身的属性
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))

        # 动作分布类，本质上来源于act_distribution_type.py
        # 目前包含：TanhGaussDistribution，GaussDistribution，DiracDistribution 这 3 个类
        # self.action_distribution_cls 直接是对其中某个类的引用
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        logits = self.policy(obs)
        # 输入状态，输出动作的均值和对数标准差
        action_mean, action_log_std = torch.chunk(
            logits, chunks=2, dim=-1
        ) 
        # 将对数标准差转化为标准差
        action_std = torch.clamp(
            action_log_std, self.min_log_std, self.max_log_std
        ).exp()

        # 返回动作均值和标准差拼接后的结果
        # 目前动作上下限 "act_high_lim" 和 "act_low_lim" 的作用还没有体现出来
        return torch.cat((action_mean, action_std), dim=-1)


# 确定性策略
class DetermPolicy(nn.Module, Action_Distribution_Cls):
    """
    确定性策略
    输入状态，直接输出动作
    """
    def __init__(self, **kwargs):
        super().__init__()
        obs_dim = kwargs["obs_dim"]
        act_dim = kwargs["act_dim"]
        hidden_sizes = kwargs["hidden_sizes"]

        self.pi = mlp(
            [obs_dim] + list(hidden_sizes) + [act_dim],
            get_activation_func(kwargs["hidden_activation"]),
            get_activation_func(kwargs["output_activation"]),
        )
        self.register_buffer("act_high_lim", torch.from_numpy(kwargs["act_high_lim"]))
        self.register_buffer("act_low_lim", torch.from_numpy(kwargs["act_low_lim"]))

        # 动作分布类，对于确定性策略，动作分布类只能是 DiracDistribution？
        self.action_distribution_cls = kwargs["action_distribution_cls"]

    def forward(self, obs):
        # 生成的是【压缩】动作，使用 tanh 函数将动作限制在了[self.act_low_lim, self.act_high_lim]之间
        action = (
            (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(self.pi(obs)) 
            + (self.act_high_lim + self.act_low_lim) / 2
        )
        return action