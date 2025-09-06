import random
import torch
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from custom_gym.async_vec_env import AsyncVectorEnv
from ch.envs import ActionSpaceScaler
from ch.torch_wrapper import Torch
from policies import DiagNormalPolicy
from ch.models.robotics import LinearValue
from ch.runner_wrapper import Runner
import custom_gym


def main(env_name = 'HalfCheetahForwardBackward-v5', seed = 42, num_workers = 10, num_iterations = 5, meta_bsz = 3, cuda = 0):
    cuda = bool(cuda)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_name = 'cpu'
    if cuda:
        torch.cuda.manual_seed(seed)
        device_name = f'cuda:{5}'
    device = torch.device(device_name)

    def make_env():
        env = gym.make(env_name)
        env = ActionSpaceScaler(env)
        return env

    env = AsyncVectorEnv([make_env for _ in range(num_workers)])
    env.reset(seed)
    env.set_task(env.sample_tasks(1)[0])
    env = Torch(env)
    policy = DiagNormalPolicy(env.state_size, env.action_size, device=device)
    if cuda:
        policy = policy.to(device)
    baseline = LinearValue(env.state_size, env.action_size)
    all_rewards = []

    for iteration in range(num_iterations):
        iteration_reward = 0.0
        iteration_replays = []
        iteration_policies = []

        for task_config in tqdm(env.sample_tasks(meta_bsz), leave=False, desc='Data'):  # Samples a new config
            clone = deepcopy(policy)
            env.set_task(task_config)
            env.reset()
            task = Runner(env)
            task_replay = []
        
if __name__ == '__main__':
    main()