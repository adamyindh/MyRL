import os
from typing import Callable, Dict, Union
from dataclasses import dataclass, field

from Multi_RL.trainer.evaluator import Evaluator

@dataclass
class Spec:
    evaluator_name: str
    entry_point: Callable

    # Environment arguments
    kwargs: dict = field(default_factory=dict)

registry: Dict[str, Spec] = {}

def register(
    evaluator_name: str, 
    entry_point: Union[Callable, str], 
    **kwargs,
):
    global registry
    new_spec = Spec(
        evaluator_name=evaluator_name, 
        entry_point=entry_point, 
        kwargs=kwargs
    )
    registry[new_spec.evaluator_name] = new_spec


# 调用 register 函数，让 registry 里存储可以调用的 Evaluator 类
register(
    evaluator_name="evaluator", 
    entry_point=Evaluator
)


# 修改 create_evaluator 函数，不使用 ray
def create_evaluator(evaluator_name: str, **kwargs) -> object:
    evaluator_spec = registry.get(evaluator_name)

    if evaluator_spec is None:
        raise KeyError(f"No registered evaluator with id: {evaluator_name}")

    # 获取 evaluator_spec 对象的参数字典，一开始是空字典
    evaluator_spec_kwargs = evaluator_spec.kwargs.copy()
    # 字典里的参数更新为传入的参数
    evaluator_spec_kwargs.update(kwargs)
    
    if callable(evaluator_spec.entry_point):
        # 由上面的分析，evaluator_spec.entry_point 是 Evaluator 这个类
        evaluator_creator = evaluator_spec.entry_point
        # 统一返回 Evluator 类的实例对象用于“顺序”评估，不使用 ray
        evaluator = evaluator_creator(**evaluator_spec_kwargs)
    else:
        raise RuntimeError(f"{evaluator_spec.evaluator_name} registered but entry_point is not specified")

    print(evaluator_name, "创建成功！")
    
    # 模仿 create_trainer 中返回的结果
    return evaluator

