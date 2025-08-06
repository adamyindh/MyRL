import torch
import warnings
import gymnasium as gym
import sys
import os
import datetime
import json
import copy
from Multi_RL.utils.common_utils import seed_everything, change_type

def init_args(envs, **args):
    """
    envs 可能是由多个环境组成的并列环境
    """
    # set torch parallel threads nums in main process
    # 如果 args 字典中存在 "num_threads_main" 键：返回该键对应的值；否则（不存在）返回为 None
    num_threads_main = args.get("num_threads_main", None)
    if num_threads_main is None:
        if "serial" in args["trainer"]:
            num_threads_main = 4
        else:
            num_threads_main = 1
    torch.set_num_threads(num_threads_main)

    # 使用cuda
    if args["enable_cuda"]:
        if torch.cuda.is_available():
            args["use_gpu"] = True
        else:
            warning_msg = "cuda is not available, use CPU instead"
            warnings.warn(warning_msg)
            args["use_gpu"] = False
    else:
        args["use_gpu"] = False

    # 每个 sampler 一次采样的数据量
    # 其实重新写 "batch_size_per_sampler" 这个键不是特别有必要
    args["batch_size_per_sampler"] = args["sample_batch_size"]

    # 关于环境：获取观测维度，使用 single_observation_space，获得单个环境的信息
    if len(envs.single_observation_space.shape) == 1:
        args["obsv_dim"] = envs.single_observation_space.shape[0]
    else:
        args["obsv_dim"] = envs.single_observation_space.shape

    # 对于gym环境，只考虑连续的动作空间
    if isinstance(envs.single_action_space, gym.spaces.Box):
        # get dimension of continuous action
        args["action_type"] = "continu"
        args["action_dim"] = (
            envs.single_action_space.shape[0]
            if len(envs.single_action_space.shape) == 1
            else envs.single_action_space.shape
        )
        args["action_high_limit"] = envs.single_action_space.high.astype("float32")
        args["action_low_limit"] = envs.single_action_space.low.astype("float32")
        # print("环境的状态空间维度：", args["obsv_dim"])
        # print("环境的动作空间类型、维度：", args["action_type"], args["action_dim"])
        # print("环境的动作上下限：", args["action_high_limit"], args["action_low_limit"])
    elif isinstance(envs.single_action_space, gym.spaces.Discrete):
        print("Only continuous action space is supported")
        sys.exit(1)

    # 创建存储相关生成文件的路径
    if args["save_folder"] is None:
        # 获取当前脚本所在目录，__file__ 是 Python 内置变量，表示当前脚本的绝对路径
        # os.path.dirname(path) 返回路径 path 的父目录。调用后返回到 utils
        dir_path = os.path.dirname(__file__)
        # 再调用 2 次，一次返回到 Multi_RL 和 MyRL
        dir_path = os.path.dirname(dir_path)
        dir_path = os.path.dirname(dir_path)

        # 在 MyRL 文件夹内创建新的 results 文件夹，然后在 results 里面创建子文件夹
        # 这个子文件夹的名称为（例如）：gym_HalfCheetah-v4
        # 然后再在 gym_HalfCheetah-v4 这个子文件夹里创建例如 DSACT_250713-134426 这样的文件夹
        args["save_folder"] = os.path.join(
            dir_path + "/results/", args["env_name"],
            args["algorithm"] +'_'+
            datetime.datetime.now().strftime("%y%m%d-%H%M%S"),
            #  + '-Max_Iter=' + str(args["max_iteration"])
        )
    os.makedirs(args["save_folder"], exist_ok=True)
    # 创建存储数据文件夹（例如：DSACT_250713-134426）中的 2 个子文件夹，分别用于存储函数估计器以及评估其
    os.makedirs(args["save_folder"] + "/apprfunc", exist_ok=True)
    os.makedirs(args["save_folder"] + "/evaluator", exist_ok=True)

    # 设置随机种子
    # 传入的 args 字典里没有 seed 这个键，因此 seed 在这里的值为 None
    seed = args.get("seed", None)
    args["seed"] = seed_everything(seed)
    print("Set the global seed: {}".format(args["seed"]))

    # 保存配置信息到 JSON 文件：copy.deepcopy(args) 创建了一个 args 字典的深拷贝
    # 将当前训练配置参数保存到一个名为 config.json 的文件中，方便后续查阅和复现实验。
    with open(args["save_folder"] + "/config.json", "w", encoding="utf-8") as f:
        json.dump(change_type(copy.deepcopy(args)), f, ensure_ascii=False, indent=4)

    # 本项目暂时使用 serial（线性化）的运算，不参考 GOPS 设置 Ray 

    # 最后返回经过初始化的参数
    return args