"""
该文件将脚本里创建的 alg, sampler, buffer, evaluator 这4个对象整合在一起
实现算法的整体【训练+评估】的流程
"""

_all__ = ["OffSerialTrainer"]

from cmath import inf
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from Multi_RL.utils.tensorboard_setup import tb_tags, add_scalars
from Multi_RL.utils.log_data import LogData
from Multi_RL.utils.common_utils import ModuleOnDevice

class OffSerialTrainer:
    def __init__(self, alg, sampler, buffer, evaluator, **kwargs):
        self.alg = alg
        self.sampler = sampler
        self.buffer = buffer
        self.evaluator = evaluator

        # 暂时不考虑优先经验回放，因此 self.per_flag=False
        self.per_flag = kwargs["buffer_name"] == "prioritized_replay_buffer"

        # create center network
        self.networks = self.alg.networks
        self.sampler.networks = self.networks

        # 现在不使用 ray，那么把评估器的网络也设置为算法对象 alg 的网络
        self.evaluator.networks = self.networks

        # initialize center network
        # 如果有初始网络，那就对其加载并赋值给 self.networks
        if kwargs["ini_network_dir"] is not None:
            self.networks.load_state_dict(torch.load(kwargs["ini_network_dir"]))

        # 训练时使用的批量数据的数量，默认为 256
        self.replay_batch_size = kwargs["replay_batch_size"]

        # 训练的最大迭代次数
        self.max_iteration = kwargs["max_iteration"]

        # 采样的间隔，在训练时，每多少步收集一次数据
        self.sample_interval = kwargs.get("sample_interval", 1)

        # 存储相关信息的间隔。存储什么？暂时还不清楚
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        
        # 存储相关信息的文件所在的位置
        # 文件路径为（示例）：D:\RL_Projects\MyRL\results\gym_HalfCheetah-v4\sac_250726-232000
        self.save_folder = kwargs["save_folder"]

        # 评估间隔，每隔多少次迭代，对当前训练好的策略网络评估
        self.eval_interval = kwargs["eval_interval"]

        # 截止到目前，最大的奖励是多少，初始设置为 -inf
        self.best_tar = -inf
        
        # 记录迭代的次数，初始设置为 0
        self.iteration = 0

        # flush tensorboard at the beginning
        # tb_tags["alg_time"] = "Time/Algorithm time [ms]-RL iter"
        # tb_tags["sampler_time"] = "Time/Sampler time [ms]-RL iter"
        # 创建一个 SummaryWriter 实例，用于向 TensorBoard 写入日志数据。日志的保存位置是：self.save_folder
        # flush_secs=20：每 20 秒自动将缓存中的日志数据写入到磁盘文件中
        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        add_scalars(
            {tb_tags["alg_time"]: 0, tb_tags["sampler_time"]: 0}, 
            self.writer, 
            0
        )
        # 强制将当前缓存中的所有日志数据立即写入到磁盘文件（由self.save_folder指定）中
        self.writer.flush()

        # pre sampling
        # 预采样阶段，在 off-policy 算法中，需要引入 replay buffer，在训练之前需要存入一定数量的数据
        # 默认 buffer_warm_size = 5e3
        while self.buffer.size < kwargs["buffer_warm_size"]:
            # samples 是一个列表，里面每个元素都是一个 Experience 类实例
            samples, _ = self.sampler.sample()
            self.buffer.add_batch(samples)

        # self.sampler_tb_dict 是 LogData 类的实例对象，用于后续计算某个指标的平均值并取出
        self.sampler_tb_dict = LogData()

        # create evaluation tasks
        # 评估任务的辅助对象，可以添加任务、检查已完成的任务数量、获取已完成的任务信息等。
        # GOPS 里使用 ray 进行远程评估时需要，我们这里修正，不远程评估，采用【训练+评估】的线性流程
        # self.evluate_tasks = TaskPool()
        # self.last_eval_iteration = 0

        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            self.networks.cuda()

        # 记录训练的开始时间
        self.start_time = time.time()


    def step(self):
        # sampling
        if self.iteration % self.sample_interval == 0:
            with ModuleOnDevice(self.networks, "cpu"):
                # sample 返回的 sampler_tb_dict 是一个字典，里面存储的是本次采样的时间
                # sampler_tb_dict = {Time/Sampler time [ms]-RL iter: 10000(单位是ms)}
                sampler_samples, sampler_tb_dict = self.sampler.sample()
            self.buffer.add_batch(sampler_samples)
            # 将本次采样的时间放入 self.sampler_tb_dict 中，后续用于计算采样的平均时间
            self.sampler_tb_dict.add_average(sampler_tb_dict)

        # replay
        # 从 replay buffer 中一次性只选取 self.replay_batch_size 个小批量的样本参与训练
        # 从下面的代码可知，replay_samples 是一个字典，每一个键对应一种类型的数据
        replay_samples = self.buffer.sample_batch(self.replay_batch_size)

        # learning
        if self.use_gpu:
            for k, v in replay_samples.items():
                replay_samples[k] = v.cuda()

        # 调用 ApproxContainer 实例（即 self.networks）从 nn.Module 继承的 train() 方法
        # 将模型设置为训练模式：启用某些层（如 Dropout、BatchNorm）在训练时的特定行为
        # Dropout：随机丢弃一部分神经元，防止过拟合。
        # BatchNorm：使用当前批次数据的均值和方差进行归一化。
        # 当模型包含 Dropout、BatchNorm、LayerNorm 等依赖训练 / 评估模式的层时，需要使用 train() 和 eval()
        # 当模型仅包含纯计算层（如 Linear、Conv2d、ReLU 等）时，train() 和 eval() 不会影响模型行为。
        # 根据是否优先经验回放（当前代码没有设置优先经验回放）来更新网络（Q网络、策略网络等），也就是训练
        self.networks.train()
        if self.per_flag:
            alg_tb_dict, idx, new_priority = self.alg.model_update(
                replay_samples, self.iteration
            )
            self.buffer.update_batch(idx, new_priority)
        else:
            # 返回训练过程中的相关数据，例如 critic 的值、策略的熵等
            alg_tb_dict = self.alg.model_update(replay_samples)

        # 将模型设置为评估模式，关闭 Dropout、BatchNorm 等在训练时使用的特殊机制
        self.networks.eval()

        # log
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            # 使用 pop() 函数得到采样的平均时间，并且写入 self.writer 中
            add_scalars(self.sampler_tb_dict.pop(), self.writer, step=self.iteration)

        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            # 调用 save_apprfunc() 函数，存储当前的 self.networks 模型参数
            self.save_apprfunc()

        # evaluate（基于 GOPS 中的 ray 形式重写）
        if self.iteration % self.eval_interval == 0 and self.iteration > 0:
            with ModuleOnDevice(self.networks, "cpu"):
                total_avg_return = self.evaluator.run_evaluation(self.iteration)

            # 判断条件：当前评估回报是否优于历史最佳回报 self.best_tar + 训练是否已进入中后期（如超过 20% 的总迭代次数）
            if (
                total_avg_return >= self.best_tar
                and self.iteration >= self.max_iteration / 5
            ):
                # 记录当前的最佳回报，并且打印提示信息，便于训练的监控
                self.best_tar = total_avg_return
                print("Eval_Iter: {}, Highest total average return = {}!".format(
                    str(self.iteration), str(self.best_tar)
                    )
                )

                # 删除旧的优化模型文件
                """
                删除保存路径下所有以 _opt.pkl 结尾的旧模型文件，避免磁盘占用过多。
                仅保留最新的最佳模型，确保模型文件的整洁性。
                """
                for filename in os.listdir(self.save_folder + "/apprfunc/"):
                    if filename.endswith("_opt.pkl"):
                        os.remove(self.save_folder + "/apprfunc/" + filename)

                # 保存当前的最佳模型。最佳模型以 _opt 为结尾
                torch.save(
                    # 保存 self.networks 中含有网络参数
                    self.networks.state_dict(),  
                    # 新的最优（当前的回报最大）的模型中包含迭代次数（如 apprfunc_1000_opt.pkl），便于追溯不同阶段的模型。
                    self.save_folder + "/apprfunc/apprfunc_{}_opt.pkl".format(self.iteration), 
                )

            # 写入缓冲区内存使用
            self.writer.add_scalar(
                # Buffer RAM：经验回放缓冲区占用的内存大小，监控内存使用效率。备注：RAM 就是内存的意思
                tb_tags["Buffer RAM of RL iteration"],  # 值为："RAM/RAM [MB]-RL iter"
                self.buffer.__get_RAM__(),
                self.iteration,
            )
            # 1、写入评估回报（以迭代次数为横坐标）
            self.writer.add_scalar(
                tb_tags["TAR of RL iteration"], 
                total_avg_return, 
                self.iteration
            )
            # 2、写入评估回报（以总时间为横坐标）
            self.writer.add_scalar(
                tb_tags["TAR of total time"],
                total_avg_return,
                int(time.time() - self.start_time),
            )
            # 3、写入评估回报（以累积采样数为横坐标，这里的累积采样数指的是使用 sampler 与环境交互获得的样本数）
            self.writer.add_scalar(
                tb_tags["TAR of collected samples"],
                total_avg_return,
                self.sampler.get_total_sample_number(),
            )
            # 4、写入评估回报（以采样总数为横坐标，训练时一次采样的数量 self.replay_batch_size = 256）
            self.writer.add_scalar(
                tb_tags["TAR of replay samples"],
                total_avg_return,
                self.iteration * self.replay_batch_size,
            )

    def train(self):
        while self.iteration <= self.max_iteration:
            self.step()
            self.iteration += 1

        # 最后再存储当前的 self.networks 模型参数
        self.save_apprfunc()
        self.writer.flush()

    def save_apprfunc(self):
        torch.save(
            self.networks.state_dict(),
            self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
        )