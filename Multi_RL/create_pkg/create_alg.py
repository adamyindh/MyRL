import os
from typing import Callable, Dict
from dataclasses import dataclass, field
import importlib

from Multi_RL.utils.MyRL_path import algorithm_path

# 创建包含所有算法类信息（在Multi_RL.algorithm文件夹下）的注册表：一个全局字典变量
@dataclass
class Spec:
    """
    Spec是 "Specification"（规格、规范）的缩写，表示 “描述某个实体的核心信息或规范定义”。
    Spec类是一个数据类，用于封装算法的关键信息（所以其名称为Spec），包含以下几个关键属性：
    algorithm：算法的名称，例如 sac
    entry_point：算法类，这是一个可调用对象，像 gops.algorithm.sac.SAC 这样的类。
    approx_container_cls：近似网络容器类，同样是可调用对象，例如 gops.algorithm.sac.ApproxContainer。
    kwargs：环境参数，是一个字典。类型提示为 dict
    """
    algorithm: str
    # Callable 表示可调用对象
    entry_point: Callable
    approx_container_cls: Callable

    # Environment arguments
    # field的含义：为 Spec 类的 kwargs 字段设置一个安全的默认值
    # 确保每次创建 Spec 实例时，kwargs 都是一个新的空字典（而非所有实例共享同一个字典）
    kwargs: dict = field(default_factory=dict)

# registry 注册表
# 这是一个全局字典（初始化为空字典），其键为算法名称（字符串），值是对应的 Spec 对象。
# Dict[str, Spec] 是类型提示，registry的主要功能是对所有已注册的算法进行索引。
registry: Dict[str, Spec] = {}

# 定义 register 函数，下面的循环中，不断将信息记录到 registry 的过程中需要使用
def register(
    algorithm: str, 
    entry_point: Callable, 
    approx_container_cls: Callable, 
    **kwargs,
):
    # 放在函数里，在函数内部声明 “当前函数要修改的是函数外部的全局变量”
    global registry

    # 新定义的 Spec 类对象
    new_spec = Spec(
        algorithm=algorithm, 
        entry_point=entry_point, 
        approx_container_cls=approx_container_cls, 
        kwargs=kwargs
        )

    # 注册表 registry 的键为 new_spec.algorithm，例如"sac"
    # 这个键的值是新建的 Spec 类对象，即 new_spec，包含：算法名称、算法类、函数近似器类、参数
    registry[new_spec.algorithm] = new_spec


# 获取 algorithm_path 中的所有算法模块，例如sac.py
alg_file_list = os.listdir(algorithm_path)

# 将 alg_file_list 中的所有算法模块存储到注册表 registry 中
for alg_file in alg_file_list:
    if alg_file[-3:] == ".py" and alg_file[0] != "_" and alg_file != "base.py":
        # 一个简单的例子就是："sac"
        alg_name = alg_file[:-3]

        # mdl 就是各种算法组成的模块，例如 mdl=Multi_RL.algorithm.sac.py
        mdl = importlib.import_module("Multi_RL.algorithm." + alg_name)

        # 以 SAC 算法为例，这一行的命令为：
        # register(
        # algorithm = "sac", 
        # entry_point = <class 'gops.algorithm.sac.SAC'>, 
        # approx_container_cls = <class 'gops.algorithm.sac.ApproxContainer'>
        # )
        # 也就是说，获得了 SAC 算法的算法类，以及算法中使用的近似网络类
        # 但是没有将相关的字典存入 register 函数中的 new_spec 对象，因此registry["sac"]是一个kwargs为空字典的Spec对象
        register(
            algorithm=alg_name,
            # sac.py 中算法类的字母都大写，例如 class SAC()
            entry_point=getattr(mdl, alg_name.upper()),
            # 算法中定义的函数近似器类的名称统一为：ApproxContainer
            approx_container_cls=getattr(mdl, "ApproxContainer"),
        )


def create_alg(**kwargs) -> object:
    """
    object 是 Python 中所有类的基类
    最后返回的是某一个类（继承自 object 类）的实例对象
    """
    # 获取算法的名称，小写，例如"sac"
    algorithm = kwargs["algorithm"]

    # algo_spec 就是一个 Spec 类，里面包含：算法名称、算法类、函数近似器类、参数
    algo_spec = registry.get(algorithm)

    if algo_spec is None:
        raise KeyError(f"No registered algorithm with id: {algorithm}")
    
    # algo_spec的字典，即kwargs应该是空的。因此algo_spec_kwargs是一个空字典
    algo_spec_kwargs = algo_spec.kwargs.copy()
    # 将脚本中的参数传入给 algo_spec_kwargs
    algo_spec_kwargs.update(kwargs)

    # callable 是 Python 内置函数，用于检查一个对象是否可以被调用
    if callable(algo_spec.entry_point):
        # 获取 sac.py 中的算法类，即 class SAC
        algorithm_creator = algo_spec.entry_point
    else:
        raise RuntimeError(f"{algo_spec.algorithm} registered but entry_point is not specified")
    
    # 获取传入参数中的 trainer_name
    trainer_name = algo_spec_kwargs.get("trainer", None)
    if (
        trainer_name is None
        or trainer_name.startswith("off_serial")
        or trainer_name.startswith("on_serial")
        or trainer_name.startswith("on_sync")
    ):
        # 调用算法类（例如 class SAC）的构造函数，创建算法类的实例对象（传入的参数是脚本文件中的参数）
        algo = algorithm_creator(**algo_spec_kwargs)
    else:
        raise RuntimeError(f"trainer {trainer_name} can not recognized")
    
    print(algorithm, "算法创建成功！")
    
    # 最后，返回创建的算法类实例对象
    return algo


def create_approx_contrainer(algorithm: str, **kwargs,) -> object:
    # algo_spec 就是一个 Spec 类，里面包含：算法名称、算法类、函数近似器类、参数
    algo_spec = registry.get(algorithm)

    if algo_spec is None:
        raise KeyError(f"No registered algorithm with id: {algorithm}")

    algo_spec_kwargs = algo_spec.kwargs.copy()
    algo_spec_kwargs.update(kwargs)

    if callable(algo_spec.approx_container_cls):
        # 实际上就是（例如）sac.py 中的 "ApproxContainer" 类
        approx_contrainer = algo_spec.approx_container_cls(**algo_spec_kwargs)
    else:
        raise RuntimeError(f"{algo_spec.algorithm} registered but approx_container_cls is not specified")
    
    # 返回估计函数类（包含了多种nn形式的估计函数）
    return approx_contrainer
