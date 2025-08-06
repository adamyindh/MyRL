"""
在 off_serial_trainer.py 中用到，在采样的过程中定义
"""

from typing import Sequence, Union

class LogData:
    """
    这个 LogData 类主要用于计算和存储数据的平均值。这个类典型的使用场景是：
    在多次迭代中收集同一指标的多个值，自动计算它们的平均值，最后通过 pop() 方法获取结果并重置，以便进行下一轮统计。
    """
    def __init__(self):
        self.data = {}
        self.counter = {}

    def add_average(self, d: Union[dict, Sequence[dict]]):
        """
        功能：用于累加数据并计算平均值，支持传入单个字典或字典的序列（如列表）
        工作原理：
        对于传入的每个键值对 (k, v)，如果是第一次出现该键 k，则直接存储值 v 并将计数器设为 1
        如果该键已存在，则更新平均值：新平均值 = (旧平均值 × 计数次数 + 新值) / (计数次数 + 1)，同时计数器加 1
        支持批量传入多个字典（通过 Sequence 类型），会逐个处理其中的每个字典
        """
        def _add_average(d: dict):
            for k, v in d.items():
                if k not in self.data.keys():
                    self.data[k] = v
                    self.counter[k] = 1
                else:
                    self.data[k] = (self.data[k] * self.counter[k] + v) / (self.counter[k] + 1)
                    self.counter[k] = self.counter[k] + 1

        if isinstance(d, dict):
            _add_average(d)
        elif isinstance(d, Sequence):
            for di in d:
                _add_average(di)
        else:
            raise TypeError(f'Unsupported type {type(d)} for add_average!')

    def pop(self) -> dict:
        """
        功能：获取当前存储的所有平均值数据，并重置内部状态
        工作原理：
        先复制当前存储的所有平均值数据（self.data）并返回
        然后清空 self.data（存储平均值的字典）和 self.counter（存储计数的字典）
        相当于 "取出" 当前积累的所有平均值结果，并准备开始新一轮的数据积累
        """
        data = self.data.copy()
        self.data = {}
        self.counter = {}
        return data
