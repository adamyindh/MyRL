import torch
import numpy as np

from Multi_RL.create_pkg.create_envs import create_envs
from Multi_RL.create_pkg.create_alg import create_approx_contrainer
from Multi_RL.utils.common_utils import get_env_type, get_env_id

class Evaluator:
    """
    这个类用于对当前训练好的策略进行评估，因此需要创建相应的环境
    同时，也需要训练好的策略网络
    """
    def __init__(self, index=0, **kwargs):
        # 设置环境的随机种子
        # 注意，在我们创建的环境里，是根据 reset 来创建随机种子的，不是直接使用env.seed(seed)（GOPS的方式）
        self.seed = kwargs["seed"]

        # 环境 id
        self.env_id = get_env_id(kwargs["env_name"])
        self.reward_scale = kwargs["reward_scale"]

        # 当然任务的目标值
        # 目标状态（目标值）
        self.target_value = kwargs["target_value"]

        # 创建环境以供后面策略的评估
        print("开始策略评估！尝试创建评估环境...")
        # 评估策略时只需要 1 个环境
        kwargs["env_num"] = 1
        self.env = create_envs(**kwargs)

        # 创建评估过程中的函数估计器，包括Q函数网络和策略网络等
        # 后续需要将 self.networks 加载为已经训练好的 networks
        print("创建评估过程中使用的函数估计器")
        self.networks = create_approx_contrainer(**kwargs)
        self.render = kwargs["is_render"]

        self.num_eval_episode = kwargs["num_eval_episode"]
        self.action_type = kwargs["action_type"]
        self.policy_func_name = kwargs["policy_func_name"]
        self.save_folder = kwargs["save_folder"]
  
        # 评估结果直接记录在 tensorboard 日志文件中即可，不需要单独记录，因此这里默认为 False
        self.eval_save = kwargs.get("eval_save", False)

        self.print_time = 0
        self.print_iteration = -1


    def load_state_dict(self, state_dict):
        """
        加载模型参数
        主要作用是将预训练好的模型参数（权重和偏置等）加载到当前模型中
        可能用于加载当前已经训练好的 networks 网络
        """
        self.networks.load_state_dict(state_dict)


    def run_an_episode(self, iteration, render=False):
        # self.print_iteration 用于指示当前迭代（第几次评估）的次数
        if self.print_iteration != iteration:
            self.print_iteration = iteration
            self.print_time = 0
        else:
            self.print_time += 1

        # 定义一些存储状态、动作、奖励、代价等的列表
        obs_list = []
        action_list = []
        # 只需要存储奖励即可，因为 cost 可直接由 -reward 得到
        reward_list = []
        # cost_list = []

        # 环境初始化，在这里设置随机种子
        # 调用 reset 函数时，参数格式为：seed=self.seed（CleanRL中的格式）
        obs, _ = self.env.reset(seed=self.seed)
        # 转换 obs 的数据类型
        obs = np.float32(obs)
        done = False

        # 只要与环境交互没有达到最大终止步数，或者没有达到终止态，就一直交互
        while not done:
            # 与环境交互的这段代码与 off_sampler 的 base 类 _step 函数中的代码很像
            obs_tensor = torch.from_numpy(obs)
            logits = self.networks.policy(obs_tensor)
            action_distribution = self.networks.create_action_distributions(logits)
            
            # 在评估时不使用随机采样的动作，而使用【众数】的动作，即最有可能采样到的动作
            action = action_distribution.mode()
            # action 还是需要和 obs 一样，保留第一个维度，指示环境的数量的维度，例如：1,2...
            # 在存储动作数据时可以把第一个维度拿掉
            action = action.detach().numpy().astype("float32")

            next_obs, reward, termination, truncation, info = self.env.step(action)
            # 将数据类型转化为 float32
            next_obs = np.float32(next_obs)
            reward = np.float32(reward)

            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncation):
                if trunc:
                    real_next_obs[idx] = info["final_observation"][idx]

            # 重新计算 reward
            if self.env_id == "HalfCheetah-v4":
                cost = ((real_next_obs[:, 8] - self.target_value) ** 2) * self.reward_scale
                # 注意：costs 和 rewards 之间始终保持“负值”关系，因此后续仅可通过 rewards 来确定 costs
                reward = -cost
            else:
                # 对于其他环境，先不重新设置 rewards
                reward = reward * self.reward_scale

            # 存储状态、动作和奖励，将第 1 个维度去除
            obs_list.append(obs[0])
            action_list.append(action[0])
            reward_list.append(reward[0])

            # 更新状态
            obs = next_obs

            # 当前评估的 episode 是否结束
            done = np.logical_or(termination, truncation)[0]

            # Draw environment animation
            # 这段在循环（与环境交互）中可以先不要
            # if render:
            #     self.env.render()
            
        eval_dict = {
            "obs_list": obs_list,
            "action_list": action_list,
            "reward_list": reward_list,
        }

        # 存储评估时的数据
        # self.print_time 指的是评估的 episode 的序号，从 0 开始计数
        if self.eval_save:
            np.save(
                self.save_folder
                + "/evaluator/iter{}_ep{}".format(iteration, self.print_time),
                eval_dict,
            )
        
        # 计算当前评估回合的总回报
        episode_return = sum(reward_list)
        return episode_return

        # cost_list = -reward_list
        # episode_cost = sum(cost_list)
        # return episode_cost

    def run_n_episodes(self, n, iteration):
        # iteration 指的是训练过程中的当前迭代次数
        episode_return_list = []
        for _ in range(n):
            # 评估时的 self.render 默认为 False
            episode_return_list.append(self.run_an_episode(iteration, self.render))
        return np.mean(episode_return_list)

    def run_evaluation(self, iteration):
        return self.run_n_episodes(self.num_eval_episode, iteration)

