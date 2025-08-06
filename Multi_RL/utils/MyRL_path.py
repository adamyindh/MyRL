import os

# 首先，获得 Multi_RL 文件夹的路径
Multi_RL_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 获得算法文件（例如sac.py）所在的文件夹
algorithm_path = os.path.join(Multi_RL_path, "algorithm")

# 获得估计函数文件（或者说估计函数的网络结构，例如mlp.py）所在的文件夹
apprfunc_path = os.path.join(Multi_RL_path, "apprfunc")

# 获得 sampler 文件夹（包含采样的模块，例如on_sampler、off_sampler）
# 由于 sampler 和 buffer 都是 trainer 对象训练时需要使用的，因此将这些文件夹放在 trainer 的文件夹内
trainer_path = os.path.join(Multi_RL_path, "trainer")
sampler_path = os.path.join(trainer_path, "sampler")
# 获取 buffer 文件所在的路径（trainer文件夹的子文件夹）
buffer_path = os.path.join(trainer_path, "buffer")

def underline2camel(s: str, first_upper: bool = False) -> str:
    """
    将下划线命名法（snake_case）的字符串转换为驼峰命名法（CamelCase）的字符串
    默认首字母小写，例如：gym_pendulum --> GymPendulum
    """
    arr = s.split("_")
    if first_upper:
        res = arr.pop(0).upper()
    else:
        res = ""
    for a in arr:
        res = res + a[0].upper() + a[1:]
    return res