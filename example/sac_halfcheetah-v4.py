# SAC 算法实现 HalfCheetah-v4 等连续动作的环境

import argparse
import math

from Multi_RL.create_pkg.create_envs import create_envs
from Multi_RL.utils.init_args import init_args
from Multi_RL.create_pkg.create_alg import create_alg
from Multi_RL.create_pkg.create_sampler import create_sampler
from Multi_RL.create_pkg.create_buffer import create_buffer
from Multi_RL.create_pkg.create_evaluator import create_evaluator
from Multi_RL.create_pkg.create_trainer import create_trainer
from Multi_RL.utils.tensorboard_setup import save_tb_to_csv

if __name__ == "__main__":
    # Parameters Setup
    # parser 是 argparse.ArgumentParser 类的一个实例。paeser 是一个参数解析器对象
    # 后续可以使用 parser.add_argument() 方法来定义各种命令行参数
    parser = argparse.ArgumentParser()

    ################################################
    # 测试的环境 + 使用的算法 + 是否使用CUDA加速
    parser.add_argument("--env_name", type=str, default="gym_HalfCheetah-v4", help="id of environment")
    # 算法名称小写，与定义的算法文件（例如 sac.py ）对应上，能根据名称直接搜索到对应的文件
    parser.add_argument("--algorithm", type=str, default="sac", help="RL algorithm")
    parser.add_argument("--enable_cuda", default=True, help="Enable CUDA")
    
    ################################################
    # 1. Parameters for environment
    parser.add_argument("--env_num", type=int, default=1, help="num of environment")
    # 设置环境动作空间的随机种子（确保随机采样动作时可以复现）
    # 这个不同于 common_utils.seed_everything 函数（设置 random、cuda等的随机数） 
    parser.add_argument("--env_seed", type=int, default=1, help="seed of action space")
    parser.add_argument("--reward_scale", type=float, default=1.0, help="reward scale factor")
    parser.add_argument("--capture_vedio", type=bool, default=False, help="Draw environment animation")
    # 是否进行对抗训练
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training") 
    # 是否对环境渲染
    parser.add_argument("--is_render", type=bool, default=False, help="Env Render") 

    # 环境期望达到的目标值
    parser.add_argument("--target_value", type=float, default=2.0, help="Target value") 

    ################################################
    # 2. Parameters of value function (state/action value) approximate
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="ActionValue",
        help="Options: StateValue/ActionValue/ActionValueDistri",
    )
    parser.add_argument(
        "--value_func_type", 
        type=str, 
        default="MLP", 
        help="Options: MLP",
    )
    value_func_type = parser.parse_known_args()[0].value_func_type
    parser.add_argument("-value_hidden_sizes", type=list, default=[256, 256])
    parser.add_argument(
        "--value_hidden_activation", 
        type=str, 
        default="relu", 
        help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--value_output_activation", type=str, default="linear", help="Options: linear/tanh")

    ################################################
    # 3. Parameters of policy function (derive action) approximate 
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="StochaPolicy",
        help="Options: DetermPolicy/StochaPolicy",
    )
    parser.add_argument(
        "--policy_func_type", 
        type=str, 
        default="MLP", 
        help="Options: MLP"
    )
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="TanhGaussDistribution",
        help="Options: default/TanhGaussDistribution/GaussDistribution",
    )
    if policy_func_type == "MLP":
        parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256])
        parser.add_argument(
            "--policy_hidden_activation",
            type=str,
            default="relu", 
            help="Options: relu/gelu/elu/selu/sigmoid/tanh"
        )
    parser.add_argument("--policy_min_log_std", type=int, default=-20)
    parser.add_argument("--policy_max_log_std", type=int, default=0.5)

    # 发现：对于策略函数，暂时没有给定输出层。因为这需要在 common_utils.py 模块中的 get_apprfunc_dict 函数中指定
    # 对于mlp类型的网络，在get_apprfunc_dict函数里最后的输出层设置为线性层"linear"
    # parser.add_argument("--policy_output_activation", type=str, default="linear", help="Options: linear/tanh")

    ################################################
    # 4. Parameters for RL algorithm
    parser.add_argument("--q_learning_rate", type=float, default=1e-3)
    parser.add_argument("--policy_learning_rate", type=float, default=3e-4)
    parser.add_argument("--alpha_learning_rate", type=float, default=1e-3)

    # special parameter
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=math.e)
    # auto_alpha 同时决定是否自动调整 alpha 和 beta
    parser.add_argument("--auto_alpha", type=bool, default=True)
    parser.add_argument("--delay_update", type=int, default=2)
    parser.add_argument("--bound", default=True)
    # Lyapunov 下降条件中，差分必须小于的值
    parser.add_argument("--constant", type=float, default=0.1)

    ################################################
    # 5. Parameters for trainer
    parser.add_argument(
        "--trainer",
        type=str,
        default="off_serial_trainer",
        help="Options: on_serial_trainer, off_serial_trainer",
    )
    # Maximum iteration number
    parser.add_argument("--max_iteration", type=int, default=10000)
    parser.add_argument(
        "--ini_network_dir",
        type=str,
        default=None
    )
    trainer_type = parser.parse_known_args()[0].trainer

    ################################################
    # 6. Parameters for sampler
    parser.add_argument(
        "--sampler_name", 
        type=str, 
        default="off_sampler", 
        help="Options: on_sampler/off_sampler"
    )
    # Period of sampling
    parser.add_argument("--sample_interval", type=int, default=1)
    # Batch size of sampler for buffer store（一次采样，采样多少个数据用于训练？）
    parser.add_argument("--sample_batch_size", type=int, default=20)
    # Add noise to action for better exploration
    parser.add_argument("--noise_params", type=dict, default=None)

    ################################################
    # 7. Parameters for buffer
    parser.add_argument(
        "--buffer_name", 
        type=str, 
        default="replay_buffer", 
        help="Options:replay_buffer/prioritized_replay_buffer"
    )
    # Size of collected samples before training
    parser.add_argument("--buffer_warm_size", type=int, default=int(5e3))
    # Max size of reply buffer
    parser.add_argument("--buffer_max_size", type=int, default=int(1e6))
    # Batch size of replay samples from buffer
    parser.add_argument("--replay_batch_size", type=int, default=256)

    ################################################
    # 8. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")

    ################################################
    # 9. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    # Save value/policy network every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=50000)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=50000)

    ################################################
    # Get parameter dictionary
    # parse_args() 用于解析命令行传入的参数，返回一个包含这些参数的【命名空间对象】
    # var() 是Python的内置函数，返回parser.parse_args()【命名空间对象】的__dict__ 属性，也就是包含【命名空间对象】所有属性及属性值的字典
    # var() 将【命名空间对象】转换为一个字典，其中键是参数名，值是参数的值。方便将这些参数作为关键字参数传递给其他函数
    args = vars(parser.parse_args())

    # ** 是 Python 中的解包操作符，用于将一个字典解包为关键字参数。将字典的键作为参数名，字典键对应的值作为参数的值
    # 例如：{'env_id': 'gym_pendulum', 'reward_scale': 1.0}，那么：create_env(env_id='gym_pendulum', reward_scale=1.0)
    # 根据传入的参数创建环境（创建环境的方式参考 CleanRL）
    envs = create_envs(**args)

    # 根据已有参数，再初始化参数
    args = init_args(envs, **args)

    ################################################
    # Step1-Step5：执行训练与评估的过程

    # Step 1: create algorithm and approximate function
    print("-------------------- Create algorithm and approximate function!-------------------- ")
    alg = create_alg(**args)

    # Step 2: create sampler in trainer
    print("-------------------- Create sampler in trainer!-------------------- ")
    sampler = create_sampler(**args)

    # Step 3: create buffer in trainer
    print("-------------------- Create buffer in trainer!-------------------- ")
    buffer = create_buffer(**args)

    # # Step 4: create evaluator in trainer
    print("-------------------- Create evaluator in trainer!-------------------- ")
    evaluator = create_evaluator(**args)

    # # Step 5: create trainer
    print("-------------------- Create trainer!-------------------- ")
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    ################################################
    # 开始训练
    print("Start training!")
    trainer.train()
    print("Training is finished!")

    ################################################
    # 将训练好的数据（记录在 Tensorboeard 中）存储到 csv 表格中
    save_tb_to_csv(args["save_folder"])
    print("训练数据已经存储到文件夹：{}".format(args["save_folder"]))