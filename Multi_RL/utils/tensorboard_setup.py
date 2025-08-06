"""
设置 Tensorboard 日志文件的相关信息
TAR: total average reward
"""
import os
import numpy as np
import pandas as pd

tb_tags = {
    # 评估时记录的数据
    "TAR of RL iteration": "Evaluation/1. TAR-RL iter",
    "TAR of total time": "Evaluation/2. TAR-Total time [s]",
    "TAR of collected samples": "Evaluation/3. TAR-Collected samples",
    "TAR of replay samples": "Evaluation/4. TAR-Replay samples",

    # 关于 Buffer 的内存大小
    "Buffer RAM of RL iteration": "RAM/RAM [MB]-RL iter",

    # 训练时使用的数据，关于 Actor 和 Critic 的损失函数
    "loss_actor": "Loss/Actor loss-RL iter",
    "loss_critic": "Loss/Critic loss-RL iter",
    
    # 算法使用的时间
    "alg_time": "Time/Algorithm time [ms]-RL iter",

    # 采样时间（或者说：与环境交互活动数据的时间）
    "sampler_time": "Time/Sampler time [ms]-RL iter",
}

def add_scalars(tb_info, writer, step):
    for key, value in tb_info.items():
        writer.add_scalar(key, value, step)

def read_tensorboard(path):
    """
    调用 read_tensorboard(path) 函数，解析指定路径（path）下的 TensorBoard 日志文件
    获取日志中所有标量数据（如损失、准确率等），返回一个包含所有标量数据的字典 data_dict
    """
    import tensorboard
    from tensorboard.backend.event_processing import event_accumulator

    # tensorboard.backend.application.logger.setLevel("ERROR")
    # 新增：导入logging模块
    # 修正：通过logging获取tensorboard的日志器并设置级别
    import logging  
    logging.getLogger("tensorboard").setLevel(logging.ERROR)

    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    valid_key_list = ea.scalars.Keys()

    output_dict = dict()
    for key in valid_key_list:
        event_list = ea.scalars.Items(key)
        x, y = [], []
        for e in event_list:
            x.append(e.step)
            y.append(e.value)

        data_dict = {"x": np.array(x), "y": np.array(y)}
        output_dict[key] = data_dict
    return output_dict

def save_csv(path, step, value):
    """
    保存 2 列数据
    """
    df = pd.DataFrame({"Step": step, "Value": value})
    df.to_csv(path, index=False, sep=",")

def save_tb_to_csv(path):
    """
    将 TensorBoard 日志文件中的标量数据解析并保存为 CSV 文件。
    保存为 csv 的过程中：
    1、遍历 data_dict 中的每个标量数据（以数据名称为键）
    2、对数据名称进行格式化处理（将路径分隔符 / 或 \ 替换为 _，避免创建子目录）
    3、在原日志路径下创建 data 子目录（若不存在则自动创建）
    4、调用 save_csv 函数，将每个标量数据的「步骤（step）」和「值（value）」以两列 CSV 格式保存到 data 子目录中，
    文件名以格式化后的标量名称命名
    """
    data_dict = read_tensorboard(path)
    for data_name in data_dict.keys():
        data_name_format = data_name.replace("\\", "/").replace("/", "_")
        csv_dir = os.path.join(path, "data")
        os.makedirs(csv_dir, exist_ok=True)
        save_csv(
            os.path.join(csv_dir, "{}.csv".format(data_name_format)),
            step=data_dict[data_name]["x"],
            value=data_dict[data_name]["y"],
        )