# sanity_check_env.py

from envs.Pendulum_gym import MiniArmPendulumEnv

env = MiniArmPendulumEnv()
obs, info = env.reset()
print("obs shape:", obs.shape)

for i in range(10):
    action = env.action_space.sample()  # 随机动作
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step {i}: reward={reward:.3f}, term={terminated}, trunc={truncated}")
