import os
from typing import Callable, Dict, Union
from dataclasses import dataclass, field
import importlib

from Multi_RL.utils.MyRL_path import trainer_path, underline2camel

@dataclass
class Spec:
    trainer: str
    entry_point: Callable

    # Environment arguments
    kwargs: dict = field(default_factory=dict)

registry: Dict[str, Spec] = {}

def register(
    trainer: str, 
    entry_point: Union[Callable, str], 
    **kwargs,
):
    global registry
    new_spec = Spec(
        trainer=trainer, 
        entry_point=entry_point, 
        kwargs=kwargs
    )
    registry[new_spec.trainer] = new_spec

# 获取 trainer_path 里所有文件的名称
trainer_file_list = os.listdir(trainer_path)

for trainer_file in trainer_file_list:
    if trainer_file.endswith("trainer.py"):
        trainer_name = trainer_file[:-3]
        mdl = importlib.import_module("Multi_RL.trainer." + trainer_name)
        register(trainer=trainer_name, entry_point=getattr(mdl, underline2camel(trainer_name)))

def create_trainer(alg, sampler, buffer, evaluator, **kwargs,) -> object:
    trainer_name = kwargs["trainer"]
    trainer_spec = registry.get(trainer_name)

    if trainer_spec is None:
        raise KeyError(f"No registered trainer with id: {trainer_name}")

    if callable(trainer_spec.entry_point):
        trainer_creator = trainer_spec.entry_point
    else:
        raise RuntimeError(f"{trainer_spec.trainer} registered but entry_point is not specified")

    # 根据 on-policy 和 off-policy 的情况分析是否需要设置 replay buffer
    if trainer_spec.trainer.startswith("off"):
        trainer = trainer_creator(alg, sampler, buffer, evaluator, **kwargs)
    elif trainer_spec.trainer.startswith("on"):
        trainer = trainer_creator(alg, sampler, evaluator, **kwargs)
    else:
        raise RuntimeError(f"trainer {trainer_spec.trainer} not recognized")
    
    print(trainer_name, "创建成功！")

    return trainer