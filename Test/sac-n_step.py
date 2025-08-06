# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from collections import deque


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the environment id of the task"""
    total_timesteps: int = 10000
    """total timesteps of the experiments"""

    num_envs: int = 3
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    n_step: int = 5


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        # 使用 gym.wrappers.RecordEpisodeStatistics 对环境包装，会在每个回合结束时，把回合的相关统计信息添加到 info 字典中
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        # 实验跟踪工具
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
    #初始化经验回放缓冲区，为后续存储和采样交互数据做准备
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # 为每个环境初始化双端队列，用于存储 N 步数据
    n_step_buffers = [deque(maxlen=args.n_step) for _ in range(args.num_envs)]

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)

    # total_timesteps 表示训练的步数（循环的总次数）
    # 现在，在每次循环里，还需要添加一个 N 步的循环，用于收集 N 步的回报数据。相当于总的交互步数为：total_timesteps * N
    for global_step in range(args.total_timesteps):
        #################### 收集数据模块 ####################
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        # terminations：指环境达到了最大的终止条件而结束。形状为：(3,)，和下面的 truncations 的形状相同
        # truncations：若达到了最大步数，则 truncations = True（自然截断），而不是 terminations = True
        # infos：一个字典，它包含了【每个环境】在当前步骤中产生的额外信息。以 3 个环境为例：
        # infos = {
        #     'final_info': [None, None, {'episode': {'r': 100, 'l': 200}}],
        #     'final_observation': [None, None, np.array([...])],
        #     'other_key': [val1, val2, val3]  # 其他环境返回的信息
        # }
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # 当 "final_info" in infos 为 True 时，说明至少有一个并行环境在当前步骤中结束了一个回合。
        if "final_info" in infos:
            # infos["final_info"]：一个列表。若环境没结束，对应位置为 None；若结束，对应位置是一个包含该回合详细信息的字典
            for info in infos["final_info"]:
                # 如果不为 None，那就说明这个位置对应的环境已经结束了
                if info is not None:
                    # info["episode"]["r"] 表示该回合的累积奖励
                    # info["episode"]["l"] 表示该回合的长度
                    # 由于经过 gym.wrappers.RecordEpisodeStatistics 包装，info 实际上是一个字典，'episode' 是字典的键
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()

        # 如果由于截断导致环境终止（例如超出某一个范围），则获取准确的“下一个时刻”的状态
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        # 上面都是一步交互的数据（对所有环境），将一步交互的数据转化成n步数据，存储到双端队列中（n_step_buffers[i]是一个双端队列）
        # 注意：多个环境中的 obs, real_next_obs, actions, rewards, terminations, infos 都存在一块，需要将其分解分别存储在 n_step_buffer 中
        for i in range(args.num_envs):
            # done 是在第 i 个环境下计算的，下面的计算都是在第 i 个环境下计算的
            done = terminations[i] or truncations[i]

            # 注意，将一步转移中 real_next_obs 作为真实的下一个时刻状态
            n_step_buffers[i].append((obs[i], actions[i], rewards[i], real_next_obs[i], done))

            # 检查 n_step_buffers 中的每个双端队列是否已满，或者 episode 是否已经结束，从而据此计算n步回报，形成一组有n步回报的数据
            if len(n_step_buffers[i]) == args.n_step or done:
                # 获取双端队列中的初始观测的状态
                first_obs, first_action, _, _, _ = n_step_buffers[i][0]

                # 正常情况下，n步回报 transition 中的下一个状态是最后一组一步 transition 中的下一个状态（中间没有遇到 episode 结束）
                # 后面如果遇到“n步数据中间某步为done”的情况，那就重新对 last_next_obs 赋值
                last_next_obs = n_step_buffers[i][-1][-2]

                # 正常情况下（没遇到 done=True 的情况），n步中最后一步的done是False
                last_done = False

                # 计算 N 步回报
                n_step_return = 0
                for j, (_, _, r, s_, d) in enumerate(n_step_buffers[i]):
                    n_step_return += args.gamma ** j * r
                    # 如果中途遇到了done=True的，那么意味着此时回合已经结束，就不再继续计算后面的回报
                    if d:
                        # 如果计算 n 步回报的中间过程中有已经终止的 transition，那么其中的下一个状态一定作为最终的 next_obs
                        last_next_obs = s_  
                        last_done = d
                        break

                # 将n步transition的数据存储到replay buffer中。与原来的 rb.add() 的存储顺序保持一致
                # rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
                # 对于 1 步 TD，是交互一次就将数据存起来；对于 N 步TD，是当n步临时双端队列满了或者done=True才将数据存到rb中
                # 这里用 np.array() 把状态、动作等包起来，是为了和以前的 obs 形式保持一致，原来的(3,17)-->(1,17)
                # 经过实验，发现多环境（num_envs≠1）下，直接使用下面这个命令不行，因为 rb 在初始化时传入了参数 num_envs
                # 意味着传入的obs, actions的shape必须为：(3,17), (3,6) 
                rb.add(np.array([first_obs]), np.array([last_next_obs]), np.array([first_action]), 
                       np.array([n_step_return]), np.array([last_done]), {})

                # 如果 n_step_buffers[i] 的中间done都不为True，那么当前 episode 没有结束，可以继续添加数据（又形成一个满的双端队列）
                # 如果中间出现了一个done为True，那么意味着回合已经结束了，不能再利用双端队列中已有的数据计算 n 步回报了
                # 因为 done=True 意味着是“干扰数据”，会影响我们获得正常的n步回报，因此自然要把双端队列清除掉
                # done 是循环一开始给出的，如果收集n步数据的过程中出现了done，那么就要计算n步回报了，计算完后也要清除临时的n步buffer
                if done:
                    n_step_buffers[i].clear()
            
        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        # 回到 3 个环境并列的状态，更新当前的状态。注意，这是在 total_timesteps 循环中的 n_step 循环中
        obs = next_obs

        #################### 基于收集的数据训练模块 ####################
        # ALGO LOGIC: training.
        # 从代码来看，先将replay buffer预热，达到一定数据量开始训练。
        # 然后就是每一个训练步都会采集数据，然后从replay buffer 中采样数据进行训练
        # 如果要计算 N 步回报，那么要参考 D4PG-PyTorch 的做法，将 N 步的数据线临时存储起来
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()

    end_time=time.time()
    print("{}算法运行时间：{}".format("SAC", str(end_time-start_time)))
