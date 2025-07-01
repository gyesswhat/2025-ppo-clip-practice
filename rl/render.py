import gymnasium as gym
from stable_baselines3 import PPO
import time

# CartPole 환경 생성 (화면 렌더링을 위해 render_mode="human" 사용)
env = gym.make("CartPole-v1", render_mode="human")

# 저장된 모델 불러오기
model = PPO.load("ppo_cartpole_model")

n_episodes = 5

for episode in range(n_episodes):
    obs, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        time.sleep(0.02)  # 너무 빠르지 않게 (20ms 간격)

    print(f"Episode {episode + 1} reward: {episode_reward}")

env.close()
