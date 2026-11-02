from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from alpha_zero.envs.gomoku import GomokuEnv

# 创建一个单一的环境实例
env = DummyVecEnv([lambda: GomokuEnv(board_size=15)])

# 使用PPO算法来训练模型
model = PPO("MlpPolicy", env, verbose=1)

# 训练模型
model.learn(total_timesteps=100000)

# 保存训练好的模型
model.save("ppo_gomoku")

# 之后可以加载训练好的模型进行使用
# model = PPO.load("ppo_gomoku")

# 测试模型
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()  # 如果你的环境支持渲染，可以使用这个方法查看棋盘
