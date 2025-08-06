import gymnasium as gym
import numpy as np

# class ConvertToFloat32(gym.Wrapper):
#     """
#     参考豆包给出的代码，目的是对环境进行包装，转化数据的类型
#     对于环境对象envs，暂时就使用了 reset 和 step 这2个函数，因此在这里重新定义
#     """
#     def __init__(self, env):
#         super().__init__(env)
#         # 更新观测空间的 dtype 描述
#         self.observation_space.dtype = np.float32

#         # 如果环境对象 envs 有 reward_range 属性，则将其以元组(min,max)的形式存储起来
#         # if hasattr(self, 'reward_range'):
#         #     self.reward_range = (
#         #         np.float32(self.reward_range[0]),
#         #         np.float32(self.reward_range[1])
#         #     )

#     def reset(self, **kwargs):
#         """
#         ** kwargs：将所有关键字参数打包成一个字典，方便在函数内部统一处理
#         例如调用时的形式为：self.env.reset(seed=1)，那么传入到这个函数中时，kwargs={"seed": 1}
#         reset() 方法通常会处理一些标准参数（如 seed、return_info 等）
#         关键字参数：“参数名 = 值” 的形式明确指定的参数，它的核心特点是通过参数名称来标识参数，而不是依赖参数的位置。
#         """
#         observation, info = self.env.reset(**kwargs)
#         # 重置时也转换观测为float32
#         observation = observation.astype(np.float32)
#         return observation, info

#     def step(self, action):
#         observation, reward, terminated, truncated, info = self.env.step(action)
#         # 将观测转换为float32
#         observation = observation.astype(np.float32)
#         # 将奖励转换为float32
#         reward = np.float32(reward)
#         return observation, reward, terminated, truncated, info
    

def make_gym_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        # 添加类型转换包装器
        # 但是由于使用 SyncVectorEnv 向量化环境，它会自动收集多个环境的返回值，并将其拼接为一个 np.ndarray。
        # 在这个拼接过程中，它默认会将 rewards 升级为 np.float64 类型
        # 因此最简单的方法就是直接对收集后的数据进行转换，或者在存入 replay buffer 之前，统一对数据进行类型转换
        # env = ConvertToFloat32(env)
            
        # 记录每个episode的统计信息，例如回合的总奖励（return）和回合长度（即步数）
        # 它会在每个回合结束时，将以下信息添加到`info`字典中（在回合的最后一步）：info['episode']是一个字典，包含：
        # - `r`: 该回合的累计奖励（return）
        # - `l`: 该回合的长度（即步数）
        # - `t`: 从回合开始到结束所经过的时间（如果环境提供了时间信息）
        env = gym.wrappers.RecordEpisodeStatistics(env)
        
        # 设置动作空间的随机种子，设置后随机采样的动作会复现。例如设置了：env.action_space.seed(42)
        # action1 = env.action_space.sample()  # 固定结果1
        # action2 = env.action_space.sample()  # 固定结果2
        # 上面 2 次调用会产生相同的动作序列，不影响策略网络的随机输出结果。
        # 由于是给 env.action_space 设置了种子，因此在这个种子下，env.action_space 调用 sample 的结果可复现
        env.action_space.seed(seed)

        return env

    return thunk