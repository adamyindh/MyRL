import numpy as np
from typing import Optional
import random
import torch
import sys
import torch.nn as nn

# 获取策略生成的动作服从的分布类型，例如GaussDistribution、TanhGaussDistribution等
# 从 Multi_RL.utils.act_distribution_cls 模块中导入所有的动作分布类
from Multi_RL.utils.act_distribution_cls import *


# 获取环境的类型（type，例如gym就是一种类型）以及环境的id（例如：HalfCheetah-v4）
def get_env_type(env_name):
    """
    获取环境的类型
    """
    index = env_name.find("_")
    if index == -1:
        return env_name
    return env_name[:index]

def get_env_id(env_name):
    """
    获取环境的id，只有知道了id，才能对应地创建环境
    """
    index = env_name.find("_")
    if index == -1 or index == len(env_name) - 1:
        return ""
    return env_name[index + 1:]

def seed_everything(seed: Optional[int] = None) -> int:
    """
    设置全局随机种子，确保代码中的所有随机数生成过程都具有确定性。让结果可复现
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        seed = random.randint(min_seed_value, max_seed_value)
    elif not isinstance(seed, int):
        # 如果提供的种子不是整数类型（如浮点数、字符串），则尝试将其转换为整数。
        seed = int(seed)

    # 如果对 random 以及 np.random 设置了相同的随机种子，那么指定随机种子后，生成的随机数序列及值相同
    random.seed(seed)
    np.random.seed(seed)

    # 设置 PyTorch CPU 随机数生成器的种子。
    torch.manual_seed(seed)

    # 设置 PyTorch GPU 随机数生成器的种子（所有 GPU）
    torch.cuda.manual_seed_all(seed)
    # # 强制 CuDNN 使用确定性算法。但启用确定性可能会降低性能，因为 CuDNN 无法选择最优（但可能非确定性）的算法。
    # torch.backends.cudnn.deterministic = True
    # # 禁用 CuDNN 的自动调优机制。禁用 benchmark 可能会使模型运行稍慢，因为无法利用特定硬件上的最优算法。
    # torch.backends.cudnn.benchmark = False

    return seed

def change_type(obj):
    """
    这个函数在init_args.py中的【保存配置信息到 JSON 文件】步骤中用到
    将当前训练配置参数保存到一个名为 config.json 的文件中，方便后续查阅和复现实验

    递归地将复杂数据结构中的 NumPy 类型转换为 Python 原生类型，同时处理字典和列表等嵌套结构
    1、NumPy 数值类型 → Python 原生类型：
    将 NumPy 整数（如 np.int32）转换为 Python 的 int。
    将 NumPy 浮点数（如 np.float32）转换为 Python 的 float。
    2、NumPy 数组 → Python 列表：
    将 np.ndarray 转换为 Python 原生列表（通过 obj.tolist()）。
    3、类型对象 → 字符串：
    将类型对象（如 int、np.ndarray）转换为其字符串表示（如 "<class 'int'>"）。
    4、递归处理嵌套结构：
    遍历字典的键值对，递归转换值的类型。
    遍历列表的元素，递归转换每个元素的类型。
    5、其他类型保持不变：
    对于非 NumPy 类型、非字典、非列表的对象（如字符串、布尔值），直接返回原对象。

    在实际应用中，这种转换通常用于：
    数据序列化：
    例如，JSON 序列化要求使用 Python 原生类型。NumPy 数组或特殊数值类型无法直接序列化为 JSON，需要先转换为列表或原生数值。
    """
    if isinstance(
        obj,
        (
            np.int_,
            np.intc,
            np.intp,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    ):
        return int(obj)
    elif isinstance(obj, type):
        return str(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = change_type(v)
        return obj
    elif isinstance(obj, list):
        for i, o in enumerate(obj):
            obj[i] = change_type(o)
        return obj
    else:
        return obj
    

def get_apprfunc_dict(key: str, **kwargs):
    """
    这个函数在算法模块（例如sac.py）中定义估计函数类ApproxContainer时用到
    主要是将脚本文件中关于值函数网络以及策略网络的参数存放到字典中，然后返回，用在create_apprfunc函数中创建网络
    """
    var = dict()
    var["apprfunc"] = kwargs[key + "_func_type"]
    var["name"] = kwargs[key + "_func_name"]
    var["obs_dim"] = kwargs["obsv_dim"]
    var["min_log_std"] = kwargs.get(key + "_min_log_std", float("-20"))
    var["max_log_std"] = kwargs.get(key + "_max_log_std", float("0.5"))

    # key的取值包括：value，policy
    # 如果在脚本中没有初始化网络的输出层，那就默认为是"linear"
    # 例如在 DDPG 中就没有指定policy网络输出层的激活函数是哪个，于是这里就默认为 linear
    if key + "_output_activation" not in kwargs.keys():
        kwargs[key + "_output_activation"] = "linear"

    # 确定估计函数（包括：值（value）函数、策略（policy）函数）的类型，目前只有"mlp"
    apprfunc_type = kwargs[key + "_func_type"]
    if apprfunc_type == "MLP":
        var["hidden_sizes"] = kwargs[key + "_hidden_sizes"]
        var["hidden_activation"] = kwargs[key + "_hidden_activation"]
        var["output_activation"] = kwargs[key + "_output_activation"]
    else:
        raise NotImplementedError
    
    # 确定策略网络输出动作的类型，当前项目只支持连续的动作
    if kwargs["action_type"] == "continu":
        var["act_dim"] = kwargs["action_dim"]
        var["act_high_lim"] = np.array(kwargs["action_high_limit"])
        var["act_low_lim"] = np.array(kwargs["action_low_limit"])
    else:
        print("Only continuous action space is supported")
        raise NotImplementedError
    
    # 对于策略动作的分布，主要是随机策略（例如sac算法中使用的policy）需要使用
    # 对于随机策略输出的动作分布，常见的就是：TanhGaussDistribution，包括CleanRL中sac算法使用的也是这个
    # 但是TanhGaussDistribution作为动作分布的优势体现在哪？特别地为什么要Tanh？这一点还不清楚
    if kwargs["policy_act_distribution"] == "default":
        # 如果没有指定动作分布，那就根据 "policy_func_name"（随机 or 确定）给出动作分布类action_distribution_cls
        if kwargs["policy_func_name"] == "StochaPolicy":  # todo: add TanhGauss
            # var["action_distribution_cls"] 直接获取指定的类（存储的是类）
            var["action_distribution_cls"] = GaussDistribution
        elif kwargs["policy_func_name"] == "DetermPolicy":
            var["action_distribution_cls"] = DiracDistribution
    else:
        # 如果已经指定了动作分布的类型，那就根据指定的类型获取对应的类，并存储在var["action_distribution_cls"]中
        # 例如指定 kwargs["policy_act_distribution"]="TanhGaussDistribution"
        # sys.modules[__name__]表示当前模块，意思是从当前模块获得指定名称的属性或对象
        var["action_distribution_cls"] = getattr(
            sys.modules[__name__], kwargs["policy_act_distribution"]
        )

    return var


def get_activation_func(key: str):
    """
    获取激活函数（包括隐藏层和输出层的激活函数）
    """
    assert isinstance(key, str)

    activation_func = None
    if key == "relu":
        activation_func = nn.ReLU

    elif key == "elu":
        activation_func = nn.ELU

    elif key == "gelu":
        activation_func = nn.GELU

    elif key == "selu":
        activation_func = nn.SELU

    elif key == "sigmoid":
        activation_func = nn.Sigmoid

    elif key == "tanh":
        activation_func = nn.Tanh

    elif key == "linear":
        activation_func = nn.Identity

    if activation_func is None:
        print("Can not identify activation name:" + key)
        raise RuntimeError

    return activation_func


class ModuleOnDevice:
    """
    1、初始化（__init__ 方法）：
    记录传入的模块（module）和目标设备（device）
    获取模块当前所在的设备（prev_device）
    判断目标设备与当前设备是否不同（different_device）
    2、进入上下文（__enter__ 方法）：
    当进入 with 语句块时，如果目标设备与当前设备不同，就将模块转移到目标设备上
    3、退出上下文（__exit__ 方法）：
    当退出 with 语句块时，如果之前进行过设备转移，就将模块迁回原来的设备（prev_device）

    这种机制的典型应用场景是：需要临时将模型转移到特定设备（如 GPU）执行某些操作（如推理、计算），
    操作完成后自动回到原来的设备，避免手动管理设备切换可能带来的疏漏。
    """
    def __init__(self, module, device):
        self.module = module
        self.prev_device = next(module.parameters()).device.type
        self.new_device = device
        self.different_device = self.prev_device != self.new_device

    def __enter__(self):
        if self.different_device:
            self.module.to(self.new_device)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.different_device:
            self.module.to(self.prev_device)
