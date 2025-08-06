# 用于根据传入的参数创建环境
# 当前（2025.7.12）只支持创建单个环境
# 后续可能拓广到多个环境，包括同步或异步的环境（参考 GOPS）

import gymnasium as gym
from datetime import datetime
import numpy as np

from Multi_RL.env.gym_env.make_gym_env import make_gym_env
from Multi_RL.utils.common_utils import get_env_type, get_env_id

# def get_env_type(env_name):
#     index = env_name.find("_")
#     if index == -1:
#         return env_name
#     return env_name[:index]

# def get_env_id(env_name):
#     index = env_name.find("_")
#     if index == -1 or index == len(env_name) - 1:
#         return ""
#     return env_name[index + 1:]

def create_envs(**args):
    """
    使用**args：表示函数可以接受任意数量的关键字参数，这些参数会被收集到字典 args 中
    传入的参数是：**args，args = vars(parser.parse_args())
    在这个函数的参数设置中，相当于：**args = **vars(parser.parse_args())
    因此我们将这个函数的参数args理解为vars(parser.parse_args())，即一个字典
    """
    
    # 从传入的 args 中获得相关的参数，需要用 get 函数才能获取，不能直接用 args.env_name
    env_name = args.get("env_name")

    env_type = get_env_type(env_name)
    env_id = get_env_id(env_name)
    print(f"Creating environment with env_id: {env_id}")

    env_seed = args.get("env_seed")
    capture_video = args.get("capture_video")
    env_num = args.get("env_num")
    run_name = f"{env_id}_{env_seed}_{datetime.now().strftime('%Y/%m/%d/%H:%M')}"

    # 如果传入的是 gym 环境（下划线前面的字符串为"gym"，那就参考 CleanRL 的环境创建过程）
    if env_type == "gym":
        envs = gym.vector.SyncVectorEnv(
        [make_gym_env(env_id, env_seed + i, i, capture_video, run_name) for i in range(env_num)]
    )
        print(env_name, "环境创建成功！")

    return envs