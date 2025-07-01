from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import gymnasium as gym
import os

# CartPole-v1 환경을 하나 생성. n_envs=1은 병렬 환경 없이 단일로 사용
env = make_vec_env("CartPole-v1", n_envs=1)

# MLP 신경망 구조를 가진 PPO 모델을 생성. 로그는 ../logs/ppo_cartpole_tensorboard/에 저장
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="../logs/ppo_cartpole_tensorboard/")

from stable_baselines3.common.callbacks import EvalCallback

eval_callback = EvalCallback(
    env,
    best_model_save_path='../logs/best_model/', # 최고 성능 모델 저장
    log_path='../logs/ppo_cartpole_tensorboard/', # 평가 결과 로그 기록
    eval_freq=5000, # 5000 timestep마다 평가
    n_eval_episodes=100, # 에피소드 100개 돌려서 평균 보상 측정
    deterministic=True, # 예측에 랜덤성 없이 평가 -> 그냥 가장 확률 높은 행동만 선택. 즉, 평가 시 정확히 모델이 가장 자신 있어 하는 전략을 그대로 쓰게 됨
    render=False # 일단 렌더링 빼고
)

model.learn(
    total_timesteps=100_000,
    tb_log_name="cartpole_eval",
    callback=eval_callback  # 자동으로 평가하고, 가장 잘된 모델을 저장하고, 기준 넘으면 알려주는 방식
)

# 학습한 모델을 로컬에 저장. 나중에 불러와서 다시 사용할 수 있음
model.save("ppo_cartpole_model")