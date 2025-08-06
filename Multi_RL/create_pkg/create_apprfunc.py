import importlib
import os
from dataclasses import dataclass, field
from typing import Callable, Dict

from Multi_RL.utils.MyRL_path import apprfunc_path

# 创建包含所有估计函数（apprfunc）信息（在Multi_RL.apprfunc文件夹下）的注册表：一个全局字典变量
@dataclass
class Spec:
    """
    Spec是 "Specification"（规格、规范）的缩写，表示 “描述某个实体的核心信息或规范定义”。
    Spec类是一个数据类，用于封装算法的关键信息（所以其名称为Spec），包含以下几个关键属性：
    apprfunc：估计函数的名称，例如 mlp
    name：mlp.py 文件夹内包含的所有类的名称
    entry_point：可调用对象，像 gops.apprfunc.mlp.StochaPolicy 这样的类（随机策略类）
    kwargs：环境参数，是一个字典。类型提示为 dict
    """
    apprfunc: str
    name: str

    # Callable 表示可调用对象
    entry_point: Callable

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
    apprfunc: str, 
    name: str,
    entry_point: Callable, 
    # **kwargs 是一个可变关键字参数，允许接受任意数量的关键字参数
    # 当调用函数时没有传入 **kwargs 对应的参数时，kwargs 在函数内部会被当作一个空字典 {}
    **kwargs,
):
    # 放在函数里，在函数内部声明 “当前函数要修改的是函数外部的全局变量”
    global registry

    # 新定义的 Spec 类对象
    new_spec = Spec(
        apprfunc=apprfunc, 
        name=name, 
        entry_point=entry_point,
        kwargs=kwargs
        )

    # 注册表 registry 的键为 new_spec.apprfunc + "_" + new_spec.name，例如"mlp_StochaPolicy"
    # 这个键的值是新建的 Spec 类对象，即 new_spec，包含：
    # 估计函数（网络）类型、由网络类型构成的类、这个类对应的可调用对象、环境参数（一个空字典）
    registry[new_spec.apprfunc + "_" + new_spec.name] = new_spec


# 获取 apprfunc_path 中的所有函数估计器（网络结构）模块，例如mlp.py（可能还有其他的模块）
apprfunc_file_list = os.listdir(apprfunc_path)

# 将 alg_file_list 中的所有算法模块存储到注册表 registry 中
for apprfunc_file in apprfunc_file_list:
    if apprfunc_file[-3:] == ".py" and apprfunc_file[0] != "_" and apprfunc_file != "base.py":
        # 一个简单的例子就是："mlp"
        apprfunc_name = apprfunc_file[:-3]
        # mdl 是 mlp.py 这个模块中所有类构成的列表
        mdl = importlib.import_module("Multi_RL.apprfunc." + apprfunc_name)
        # 导入模块mdl中的所有类
        for name in mdl.__all__:
            register(
                apprfunc=apprfunc_name, 
                name=name, 
                entry_point=getattr(mdl, name)
            )


def create_apprfunc(**kwargs) -> object:
    """
    根据函数估计器的参数（字典形式传入）创建出函数估计器的实例对象，然后返回
    """
    # 脚本以及 init_args.py 里都没有定义"apprfunc"这个键，因此其来自于get_apprfunc_dict函数
    # 获取特定的函数估计器（value、policy）对应的网络类型，如 "mlp"
    # value、policy 的函数估计器的名称分别是（例如）：ActionValue、StochaPolicy
    apprfunc = kwargs["apprfunc"].lower()
    name = kwargs["name"]

    # 这样，获得【mlp+StochaPolicy】对应的spec对象，里面的 entry_point 就是指定的类
    # 类的名称就是：name = kwargs["name"] 的结果，例如 ActionValue、StochaPolicy 等
    # 因此，脚本中的 “ActionValue、StochaPolicy” 与 mlp.py 里定义的类的名称是对应的
    # registry 是一个字典，apprfunc + "_" + name是键，对应的值就是 spec 对象
    apprfunc_spec = registry.get(apprfunc + "_" + name)

    if apprfunc_spec is None:
        raise KeyError(f"No registered apprfunc with id: {apprfunc}_{name}")

    apprfunc_spec_kwargs = apprfunc_spec.kwargs.copy()
    # apprfunc_spec_kwargs 是个字典，存入的参数就是传入的参数（本质上来源于sac.py中的估计函数的参数（字典形式））
    apprfunc_spec_kwargs.update(kwargs)

    # 调用apprfunc（例如mlp，cnn等）中的类，也就是name对应的ActionValue、StochaPolicy等
    if callable(apprfunc_spec.entry_point):
        apprfunc_creator = apprfunc_spec.entry_point
    else:
        raise RuntimeError(f"{apprfunc_spec.apprfunc}-{apprfunc_spec.name} registered but entry_point is not specified")

    # 创建值函数网络或policy网络时，使用的参数还是get_apprfunc_dict函数返回的参数
    # 例如，apprfunc_creator 是 mlp.py 中的 ActionValue 类
    apprfunc = apprfunc_creator(**apprfunc_spec_kwargs)

    return apprfunc