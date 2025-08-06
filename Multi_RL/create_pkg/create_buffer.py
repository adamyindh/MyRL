from dataclasses import dataclass, field
from typing import Callable, Dict, Union

import os
from Multi_RL.utils.MyRL_path import buffer_path, underline2camel
import importlib

@dataclass
class Spec:
    buffer_name: str
    entry_point: Callable

    # Environment arguments
    kwargs: dict = field(default_factory=dict)


# 初始化注册表
registry: Dict[str, Spec] = {}


def register(
    buffer_name: str, 
    entry_point: Callable, 
    **kwargs,
):
    global registry

    new_spec = Spec(
        buffer_name=buffer_name, 
        entry_point=entry_point, 
        kwargs=kwargs
    )

    registry[new_spec.buffer_name] = new_spec

# 获取 buffer 文件夹内的所有模块组成的列表
buffer_file_list = os.listdir(buffer_path)


for buffer_file in buffer_file_list:
    if buffer_file[-3:] == ".py" and buffer_file[0] != "_" and buffer_file != "base.py":
        # 一个简单的例子："replay_buffer"
        buffer_name = buffer_file[:-3]
        # 例如，导入 buffer 文件夹内的 replay_buffer.py 模块
        mdl = importlib.import_module("Multi_RL.trainer.buffer." + buffer_name)
        register(
            buffer_name=buffer_name, 
            entry_point=getattr(mdl, underline2camel(buffer_name))
        )

def create_buffer(**kwargs) -> object:
    # buffer_name 的选项有：replay_buffer/prioritized_replay_buffer
    buffer_name = kwargs.get("buffer_name", None)

    buffer_spec = registry.get(buffer_name)

    if buffer_spec is None:
        raise KeyError(f"No registered buffer with id: {buffer_name}")
    
    # 更新buffer_spec的参数字典为传入的参数字典kwargs
    buffer_spec_kwargs = buffer_spec.kwargs.copy()
    buffer_spec_kwargs.update(kwargs)

    # 导入相关模块中的创建 buffer 的类
    if callable(buffer_spec.entry_point):
        buffer_creator = buffer_spec.entry_point
    else:
        raise RuntimeError(f"{buffer_spec.buffer_name} registered but entry_point is not specified")

    # 根据 trainer 的类型确定 buffer 的类型
    trainer_name = buffer_spec_kwargs.get("trainer", None)
    if trainer_name is None or trainer_name.startswith("on"):
        # 如果 trainer 是 on-policy 的，那么就不需要 buffer
        buffer = None
        print("No buffer for on-policy trainer! return None")
    else:
        # 现在只考虑 on-policy 和 off-policy 2种训练类型，且都是 serial 的
        # 如果是 off-policy + serial，那么就直接创建 replay_buffer（暂时不考虑prioritized_replay_buffer）
        buffer = buffer_creator(**buffer_spec_kwargs) 
        print(buffer_name, "创建成功")

    return buffer