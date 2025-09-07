import random
from metaworld import ML1
from custom_gym.envs.meta_env import MetaEnv


class MetaWorldML1(ML1, MetaEnv):
    """
    **Description**

    The ML1 Benchmark of Meta-World is focused on solving just one task on different object / goal
    configurations.This task can be either one of the following: 'reach', 'push' and 'pick-and-place'.
    The meta-training is performed on a set of 50 randomly chosen once initial object and goal positions.
    The meta-testing is performed on a held-out set of 10 new different configurations. The starting state
    of the robot arm is always fixed. The goal positions are not provided in the observation space, forcing
    the Sawyer robot arm to explore and adapt to the new goal through trial-and-error. This is considered
    a relatively easy problem for a meta-learning algorithm to solve and acts as a sanity check to a

    """

    def __init__(self, env_name, seed=None):
        super(MetaWorldML1, self).__init__(env_name, seed=seed)

    def sample_tasks(self, num_tasks):
        tasks = random.sample(super(MetaWorldML1, self).train_tasks, k=num_tasks)
        return tasks
