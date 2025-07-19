import gym
from gym import spaces
import numpy as np

class MazeEnv(gym.Env):
    def __init__(self, size=7):
        super(MazeEnv, self).__init__()
        self.size = size
        self.observation_space = spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32)
        self.action_space = spaces.Discrete(4)  # 0:위, 1:아래, 2:좌, 3:우
        self.maze = None # Initialize to None
        self.start_pos = None
        self.goal_pos = None
        self.agent_pos = None

    def _make_maze(self):
        # 간단한 무작위 미로 생성 (DFS 알고리즘)
        self.maze = np.ones((self.size, self.size), dtype=np.int8)
        stack = [(1, 1)]
        self.maze[1, 1] = 0
        dirs = [(-2,0),(2,0),(0,-2),(0,2)]
        while stack:
            y, x = stack[-1]
            np.random.shuffle(dirs)
            for dy, dx in dirs:
                ny, nx = y+dy, x+dx
                if 0 < ny < self.size-1 and 0 < nx < self.size-1 and self.maze[ny, nx] == 1:
                    self.maze[ny, nx] = 0
                    self.maze[y+dy//2, x+dx//2] = 0
                    stack.append((ny, nx))
                    break
            else:
                stack.pop()
        self.start_pos = np.array([1, 1])
        self.goal_pos = np.array([self.size-2, self.size-2])

    def set_maze_and_pos(self, maze_array, start_pos, goal_pos):
        self.maze = maze_array
        self.start_pos = np.array(start_pos)
        self.goal_pos = np.array(goal_pos)
        self.agent_pos = self.start_pos.copy()
        return self.agent_pos.copy()

    def reset(self):
        # If maze is not set, generate a default one
        if self.maze is None:
            self._make_maze()
        self.agent_pos = self.start_pos.copy()
        return self.agent_pos.copy()

    def step(self, action):
        y, x = self.agent_pos
        move = [(-1,0),(1,0),(0,-1),(0,1)][action]
        ny, nx = y + move[0], x + move[1]
        reward = -1
        done = False
        # 미로 내부 & 벽이 아닐 때만 이동
        if 0 <= ny < self.size and 0 <= nx < self.size and self.maze[ny, nx] == 0:
            self.agent_pos = np.array([ny, nx])
        # 목표 도달 보상
        if np.all(self.agent_pos == self.goal_pos):
            reward = 100
            done = True
        return self.agent_pos.copy(), reward, done, {}

    def render(self, mode="human"):
        grid = np.full((self.size, self.size), '⬜', dtype='<U1')
        for y in range(self.size):
            for x in range(self.size):
                if self.maze[y, x] == 1:
                    grid[y, x] = '⬛'
        y, x = self.agent_pos
        gy, gx = self.goal_pos
        grid[gy, gx] = '🏁'
        grid[y, x] = '🐶'
        print('\n'.join(' '.join(row) for row in grid))
        print('-'*20)
