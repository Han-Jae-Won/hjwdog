from maze_env import MazeEnv
from stable_baselines3 import DQN

env = MazeEnv(size=7)
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2000000)
model.save("saved_model/maze_dqn")

# 평가 (에이전트가 미로 풀기 시연)
obs = env.reset()
done = False
step = 0
while not done and step < 50:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    env.render()
    step += 1

if all(env.agent_pos == env.goal_pos):
    print("AI가 미로를 클리어했습니다! 🎉")
else:
    print("실패 또는 타임아웃")
