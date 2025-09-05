#!/usr/bin/env python3

import multiprocessing as mp

from .envs.subproc_vec_env import SubprocVecEnv

class AsyncVectorEnv(SubprocVecEnv):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/gym/async_vec_env.py)

    **Description**

    Asynchronous vectorized environment for working with MetaEnvs.
    Allows multiple environments to be run as separate processes.

    **Credit**

    Adapted from OpenAI and Tristan Deleu's implementations.

    """
    def __init__(self, env_fns, env=None):
        self.num_envs = len(env_fns)
        self.queue = mp.Queue()
        super(AsyncVectorEnv, self).__init__(env_fns, queue=self.queue)
        if env is None:
            env = env_fns[0]()
        self._env = env
        # self.reset()

    def set_task(self, task):
        tasks = [task for _ in range(self.num_envs)]
        reset = super(AsyncVectorEnv, self).set_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks):
        tasks = self._env.unwrapped.sample_tasks(num_tasks)
        return tasks

    def step(self, actions):
        obs, rews, terminateds, truncateds, task_ids, infos = super(AsyncVectorEnv, self).step(actions)
        return obs, rews, terminateds, truncateds, task_ids, infos

    def reset(self, seed=None):
        for i in range(self.num_envs):
            self.queue.put(i)
        for i in range(self.num_envs):
            self.queue.put(None)
        obs, infos, ids = super(AsyncVectorEnv, self).reset(seed=seed)
        return obs, infos

    def render(self, *args, **kwargs):
        self._env.render(*args, **kwargs)
