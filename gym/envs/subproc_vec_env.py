import multiprocessing as mp

import gymnasium as gym
import numpy as np


class EnvWorker(mp.Process):
    def __init__(self, remote, env_fn, queue, lock):
        super(EnvWorker, self).__init__()
        self.remote = remote
        self.env = env_fn()
        self.queue = queue
        self.lock = lock
        self.task_id = None
        self.terminated = False
        self.truncated = False

    def empty_step(self):
        observation = np.zeros(self.env.observation_space.shape,
                               dtype=np.float64)
        reward, terminated, truncated = 0.0, True, True
        return observation, reward, terminated, truncated, {}

    def try_reset(self, seed=None):
        observation, info = self.env.reset(seed=seed)
        return observation, info

    def run(self):
        while True:
            command, data = self.remote.recv()
            if command == 'step':
                observation, reward, terminated, truncated, info = self.env.step(data)
                if (terminated or truncated) and (not (self.terminated or truncated)):
                    observation, info = self.try_reset()
                self.remote.send((observation, reward, terminated, truncated, self.task_id, info))
            elif command == 'reset':
                observation, info = self.try_reset(data)
                self.remote.send((observation, info, self.task_id))
            elif command == 'set_task':
                self.env.unwrapped.set_task(data)
                self.remote.send(True)
            elif command == 'close':
                self.remote.close()
                break
            elif command == 'get_spaces':
                self.remote.send((self.env.observation_space,
                                  self.env.action_space))
            else:
                raise NotImplementedError()


class SubprocVecEnv(gym.Env):
    def __init__(self, env_factory, queue):
        self.lock = mp.Lock()
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in env_factory])
        self.workers = [EnvWorker(remote, env_fn, queue, self.lock)
                        for (remote, env_fn) in zip(self.work_remotes, env_factory)]
        for worker in self.workers:
            worker.daemon = True
            worker.start()
        for remote in self.work_remotes:
            remote.close()
        self.waiting = False
        self.closed = False

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        observations, rewards, terminated, truncated, task_ids, infos = zip(*results)
        return np.stack(observations), np.stack(rewards), np.stack(terminated), np.stack(truncated), task_ids, infos

    def reset(self, seed=None):
        for remote in self.remotes:
            remote.send(('reset', seed))
        results = [remote.recv() for remote in self.remotes]
        observations, infos, task_ids = zip(*results)
        return np.stack(observations), infos, task_ids

    def set_task(self, tasks):
        for remote, task in zip(self.remotes, tasks):
            remote.send(('set_task', task))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for worker in self.workers:
            worker.join()
        self.closed = True
