from dataclasses import dataclass, field
from typing import Callable, Dict, Union

import os
from Multi_RL.utils.MyRL_path import sampler_path, underline2camel
import importlib

# 设计注册表，存储 sampler 文件夹中可导入的模块，具体包括：on_sampler 和 off_sampler
@dataclass
class Spec:
    sampler_name: str
    entry_point: Callable

    # Environment arguments
    kwargs: dict = field(default_factory=dict)

registry: Dict[str, Spec] = {}

def register(
    sampler_name: str, 
    entry_point: Union[Callable, str], 
    **kwargs,
):
    global registry
    new_spec = Spec(
        sampler_name=sampler_name, 
        entry_point=entry_point, 
        kwargs=kwargs
    )
    registry[new_spec.sampler_name] = new_spec

# 将 sampler 的相关信息放到注册表 registry 中
sampler_file_list = os.listdir(sampler_path)

for sampler_file in sampler_file_list:
    if sampler_file[-3:] == ".py" and sampler_file[0] != "_" and sampler_file != "base.py":
        # 例如：sampler_name = "off_sampler"
        sampler_name = sampler_file[:-3]
        # mdl 是 off_sampler 模块中所有类构成的列表
        mdl = importlib.import_module("Multi_RL.trainer.sampler." + sampler_name)

        # underline2camel(sampler_name=off_sampler) 作用后的结果为 OffSampler
        # 如果 first_upper=True，那就是 OFFSampler
        register(
            sampler_name=sampler_name, 
            entry_point=getattr(mdl, underline2camel(sampler_name))
        )

def create_sampler(**kwargs):
    """
    根据传入的参数，创建 sampler
    sampler 的具体作用目前未知
    """
    # 在 SAC 算法中，kwargs["sampler_name"]="off_sampler"
    sampler_name = kwargs["sampler_name"]

    # 从注册表 registry 获得与 sampler_name 对应的 spec 对象
    sampler_spec = registry.get(sampler_name)

    if sampler_spec is None:
        raise KeyError(f"No registered sampler with id: {sampler_name}")
    
    sampler_spec_kwargs = sampler_spec.kwargs.copy()
    # 从脚本中传入的参数（有关于sampler的参数）存到了sampler_spec_kwargs中
    sampler_spec_kwargs.update(kwargs)

    # sampler_spec.entry_point（例如）就是 off_sampler 模块中定义的类：OffSampler
    if callable(sampler_spec.entry_point):
        # 获得与 sampler_name 对应的创建 sampler 的类（或理解为类的引用）
        sampler_creator = sampler_spec.entry_point
        sampler = sampler_creator(**sampler_spec_kwargs)
    else:
        raise RuntimeError(f"{sampler_spec.sampler_name} registered but entry_point is not specified")
    
    # 获得 trainer 的名字，注意 sampler 是 trainer 的一个组成部分
    # 在当前项目中，仅包含 on_serial_trainer 和 off_serial_trainer
    # trainer_name = kwargs.get("trainer", None)

    print(sampler_name, "创建成功！")

    return sampler

