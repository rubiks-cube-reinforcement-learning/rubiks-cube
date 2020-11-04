import numpy as np
import gym
from gym import spaces
from gym.spaces import Box

from utils import KociembaCube

MAX_STEPS = 15
MAX_SOLUTION_LENGTH = 15

class RubiksEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(RubiksEnv, self).__init__()
        self.action_space = spaces.Discrete(len(KociembaCube.MOVES))
        self.observation_space = Box(0, 6, [1, len(KociembaCube.SOLVED_STATE)], dtype=np.uint8)
        self.cube = None
        self.current_step = None
        self.last_reward = None
        self.current_solution = None

    def step(self, action):
        self.current_step += 1

        previous_solution = self.cube.get_solution()
        move = self.cube.get_move_by_idx(action)
        self.cube.move(move)
        self.current_solution = self.cube.get_solution()
        diff = len(previous_solution) - len(self.current_solution)
        reward = 1 if diff > 0 else 0
        done = self.current_step > MAX_STEPS or len(self.current_solution) > MAX_SOLUTION_LENGTH or self.cube.is_solved()
        obs = self._next_observation()
        self.last_reward = reward
        return obs, reward, done, {}

    def reset(self):
        self.cube = KociembaCube()
        self.cube.scramble(2)
        self.current_step = 0
        self.last_reward = 0
        self.current_solution = self.cube.get_solution()

        return self._next_observation()

    def _next_observation(self):
        return np.array(self.cube.as_vector())

    def render(self, mode='human', close=False):
        print(f'Steps: {self.current_step}  |  Reward: {self.last_reward}  |  Distance to solution: {len(self.current_solution)}  |  State: {self.cube.__str__()}')

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: RubiksEnv()])
model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000)
obs = env.reset()
for i in range(10):
  action, _states = model.predict(obs)
  obs, rewards, done, info = env.step(action)
  env.render()


# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 8))
# plt.plot(rewards, label='Rewards')
# plt.xlabel('Episode', fontsize=20)
# plt.ylabel('Reward', fontsize=20)
# plt.hlines(REWARD_THRESHOLD, 0, len(test_rewards), color='r')
# plt.legend(loc='lower right')
# plt.grid()
# plt.savefig('plot.png')
