"""
存储了动作分布的类型
被 common_utils.py 中的 get_apprfunc_dict 函数调用并存储到字典中
具体为：var["action_distribution_cls"] = GaussDistribution
"""

import torch
EPS = 1e-6

#################### 获取动作分布类 ####################
class Action_Distribution_Cls:
    """
    在 sac.py 中，需要获取 policy 网络的动作分布类，也就是本文件下面给出的一些类
    下面的这些类在 sac.py 中的 get_apprfunc_dict 函数被引入了，并且存放在了字典中
    在 sac.py 构造策略网络的函数 create_apprfunc 中，策略网络被 mlp.py 中对应的类构成成一个实例
    其中的action_distribution_cls属性：self.action_distribution_cls = kwargs["action_distribution_cls"]
    就是下面给出的动作分布类。即 policy.action_distribution_cls = 下面的动作分布类的引用
    这个引用的源头还是在于 common_utils.py 中的 get_apprfunc_dict 函数

    记住，policy 网络会继承这个类
    因此需要在定义 policy 网络时导入本模块（即这个文件）中的 Action_Distribution_Cls 类
    """
    def __init__(self):
        super().__init__()

    def get_act_dist_cls(self, logits):
        # 获取 policy 网络自身的动作分布类，具体的值其实是一个动作分布类的引用
        action_distribution_cls = getattr(self, "action_distribution_cls")

        # 判断 policy 网络中是否有动作的上下限属性
        # 但需要注意的是，这里的 act_high_lim 和 act_low_lim 是环境中实际的动作上下限
        # 但是下面的动作分布类 TanhGaussDistribution 中，动作的上下限确是 1 和 -1，和实际的动作上下限似乎没有直接关系
        has_act_lim = hasattr(self, "act_high_lim") and hasattr(self, "act_low_lim")

        # 根据输入的 logits 创建出动作分布类的实例对象
        # 随机policy：logits = policy 网络生成的均值 + 方差
        # 确定policy：logits = policy 网络直接生成的动作
        act_dist_cls = action_distribution_cls(logits)

        # 这里就根据实际环境的动作上下限，将其赋值给了动作分布类
        if has_act_lim:
            act_dist_cls.act_high_lim = getattr(self, "act_high_lim")
            act_dist_cls.act_low_lim = getattr(self, "act_low_lim")

        # 返回创建好的动作分布类实例
        # 总结：策略网络输出 logits（例如均值和方差），然后动作分布类网络根据这些 logits 生成相关的动作
        # 对于随机策略，输出logits（均值+方差） --> 根据 logits 输出实际的动作，是 2 步完成的
        return act_dist_cls


#################### 具体的动作分布类 ####################
class TanhGaussDistribution:
    def __init__(self, logits):
        # logits 表示策略网络输出的张量，最后一个维度的大小为 2 * action_dim，因此后面要用chunks截断
        # logits 的 shape：[batch_size, 2 * action_dim]
        self.logits = logits
        # 沿着最后一个维度（第一个维度可能表示批量数据）平均分割成2个子张量，分别表示各动作维度上的均值和方差
        # 经过截断后，self.maen和self.std的shape都是：[batch_size, action_dim]
        self.mean, self.std = torch.chunk(logits, chunks=2, dim=-1)

        # self.gauss_distribution 是一个多变量正态分布（维度为 action_dim），且每个维度独立
        # 具有批量维度（表示数据的批量batch_size）和事件维度（表示分布的 “事件空间”：动作的维度）
        # batch_shape: (batch_size,), event_shape: (action_dim,)
        # gauss_distribution 的 batch_shape 和 event_shape 影响概率的计算（log_prob和熵）
        # gauss_distribution.log_prob(action_sample)：计算输入的动作样本的对数概率
        # gauss_distribution.entropy()：计算gauss_distribution本身每个分布对应的熵值
        self.gauss_distribution = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(self.mean, self.std),
            reinterpreted_batch_ndims=1,
        )

        # 动作的上下限，这个固定（每个动作的维度）分别为 1.0 和 -1.0？
        # 为什么步直接赋值为环境给出的动作上下限（传入参数赋值）？
        # 根据 act 的上下 lim，这里的 act 应该是归一化的结果，后面是否会像 CleanRL 那样，
        # 对策略网络生成的 act 进行缩放？
        # 和上面的类对比分析，这 2 行可以认为是初始化的操作
        self.act_high_lim = torch.tensor([1.0])
        self.act_low_lim = torch.tensor([-1.0])

    def sample(self):
        # 根据 self.gauss_distribution 服从的分布采样（生成的样本数=batch_size，即构造函数中传入的logits的第1个维度）
        action = self.gauss_distribution.sample()
        # TanhGaussDistribution 类实现了一种受限高斯分布
        # 原始高斯分布self.gauss_distribution）采样的范围是无界的
        # 实际的环境中，动作有上下限约束（例如这里的[-1,1]），因此通过torch.tanh函数将生成的实际动作映射到[-1,1]区间
        # (self.act_high_lim - self.act_low_lim) / 2 是 action_scale
        # (self.act_high_lim + self.act_low_lim) / 2 是 action_bias
        action_limited = (
            (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(action) 
            + (self.act_high_lim + self.act_low_lim) / 2
        )
        # a'=torch.tanh(a)*scale+bias，因此a'的PDF会改变，那么计算a'的对数概率也就会发生改变
        # PDF之间的关系：p(a') = p(a) * |da/da'|
        # log p(a') = log p(a) - log|da/da'|，|da/da'|是Jacobi行列式的绝对值
        # tanh(x) 关于 x 求导的结果为：1 - tanh^2(x)
        # self.gauss_distribution.log_prob(action)：计算多元高斯分布在action处的对数概率密度值（子动作维度的对数概率相加）
        log_prob = (
            self.gauss_distribution.log_prob(action)  # 原始高斯分布的对数概率
            - torch.log(1 + EPS - torch.pow(torch.tanh(action), 2)).sum(-1)  # tanh 变换的雅可比行列式
            - torch.log((self.act_high_lim - self.act_low_lim) / 2).sum(-1)  # 缩放变换的雅可比行列式
        )
        return action_limited, log_prob

    def rsample(self):
        # 相较于上面 sample() 函数的构造，这里只是使用了 rsample
        # sample 是普通的采样，不支持梯度传播
        # rsample()：重参数化采样，支持梯度传播。action = mu + sigma * epsl(~N(0,1))
        # 此时 action 与mu和sigma形成明确的函数关系，因此梯度可从 action 反向传播到mu和sigma。
        # mu 和 sigma就是网络输出的结果（logits），其中包含网络梯度信息，继而转到了 self.gauss_distribution 上
        action = self.gauss_distribution.rsample()
        action_limited = (
            (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(action) 
            + (self.act_high_lim + self.act_low_lim) / 2
        )
        log_prob = (
            self.gauss_distribution.log_prob(action)
            - torch.log(1 + EPS - torch.pow(torch.tanh(action), 2)).sum(-1)
            - torch.log((self.act_high_lim - self.act_low_lim) / 2).sum(-1)
        )
        return action_limited, log_prob

    def log_prob(self, action_limited) -> torch.Tensor:
        # 根据约束后的动作action_limited来反推出采样得到的action，然后根据action来计算出action_limited的对数概率
        # 实际上上面 2 个函数的返回结果中已经包含了限制在 [-1,1] 上的动作 action_limited 及其对应的对数概率
        action = torch.atanh(
            (1 - EPS)
            * (2 * action_limited - (self.act_high_lim + self.act_low_lim))
            / (self.act_high_lim - self.act_low_lim)
        )
        log_prob = self.gauss_distribution.log_prob(action) - torch.log(
            (self.act_high_lim - self.act_low_lim) / 2
            * (1 + EPS - torch.pow(torch.tanh(action), 2))
        ).sum(-1)
        return log_prob

    def entropy(self):
        # 熵的大小仅与策略网络输出的均值与方差有关，与采样的实际样本无关
        return self.gauss_distribution.entropy()

    def mode(self):
        # 求众数（mode），在PDF中，就是峰值对应的自变量的取值，即均值
        # 当然，这里的求动作的均值也是限制在 [act_low_lim, act_high_lim] 之间的
        return (
            (self.act_high_lim - self.act_low_lim) / 2 * torch.tanh(self.mean) 
            + (self.act_high_lim + self.act_low_lim) / 2
        )

    def kl_divergence(self, other: "GaussDistribution") -> torch.Tensor:
        # 求 2 个 Gaussian 分布之间的 KL 散度
        return torch.distributions.kl.kl_divergence(
            self.gauss_distribution, other.gauss_distribution
        )


class GaussDistribution:
    """
    相较于上面的 tanhGaussDistribution 类，这个是纯的 Gaussian 分布
    生成的动作范围是无界的，不考虑动作的限制
    """
    def __init__(self, logits):
        self.logits = logits
        self.mean, self.std = torch.chunk(logits, chunks=2, dim=-1)
        self.gauss_distribution = torch.distributions.Independent(
            base_distribution=torch.distributions.Normal(self.mean, self.std),
            reinterpreted_batch_ndims=1,
        )
        self.act_high_lim = torch.tensor([1.0])
        self.act_low_lim = torch.tensor([-1.0])

    def sample(self):
        action = self.gauss_distribution.sample()
        log_prob = self.gauss_distribution.log_prob(action)
        return action, log_prob

    def rsample(self):
        action = self.gauss_distribution.rsample()
        log_prob = self.gauss_distribution.log_prob(action)
        return action, log_prob

    def log_prob(self, action) -> torch.Tensor:
        log_prob = self.gauss_distribution.log_prob(action)
        return log_prob

    def entropy(self):
        return self.gauss_distribution.entropy()

    def mode(self):
        return torch.clamp(self.mean, self.act_low_lim, self.act_high_lim)

    def kl_divergence(self, other: "GaussDistribution") -> torch.Tensor:
        return torch.distributions.kl.kl_divergence(
            self.gauss_distribution, other.gauss_distribution
        )
    

class DiracDistribution:
    """
    确定性策略生成动作服从的分布
    策略网络输出的 logits 就是实际的动作，不需要再做额外的变换
    """
    def __init__(self, logits):
        # logits 的shape：[batch_size, action_dim]
        self.logits = logits

    def sample(self):
        return self.logits, torch.zeros_like(self.logits).sum(-1)

    def mode(self):
        return self.logits