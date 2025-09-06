import random
import torch
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch import autograd
from custom_gym.async_vec_env import AsyncVectorEnv
from ch.envs import ActionSpaceScaler
from ch.torch_wrapper import Torch
from policies import DiagNormalPolicy
from ch.models.robotics import LinearValue
from ch.runner_wrapper import Runner
from ch._torch import normalize
from ch.td import discount
from ch.pg import generalized_advantage
from ch.algorithms.a2c import policy_loss
from algorithms.maml import maml_update
import custom_gym

def compute_advantages(baseline, tau, gamma, rewards, terminated, truncated, states, next_states):
    # Update baseline
    dones = torch.logical_or(terminated, truncated)
    returns = discount(gamma, rewards, dones)
    baseline.fit(states, returns)
    values = baseline(states)
    next_values = baseline(next_states)
    bootstraps = values * (~dones) + next_values * dones
    next_value = torch.zeros(1, device=values.device)
    return generalized_advantage(tau=tau,
                                gamma=gamma,
                                rewards=rewards,
                                dones=dones,
                                values=bootstraps,
                                next_value=next_value)


def maml_a2c_loss(train_episodes, learner, baseline, gamma, tau):
    # Update policy and baseline
    states = train_episodes.state()
    actions = train_episodes.action()
    rewards = train_episodes.reward()
    terminated = train_episodes.terminated()
    truncated = train_episodes.truncated()
    next_states = train_episodes.next_state()
    log_probs = learner.log_prob(states, actions)
    advantages = compute_advantages(baseline, tau, gamma, rewards,
                                    terminated, truncated, states, next_states)
    advantages = normalize(advantages).detach()
    return policy_loss(log_probs, advantages)


def fast_adapt_a2c(clone, train_episodes, adapt_lr, baseline, gamma, tau, first_order=False):
    second_order = not first_order
    loss = maml_a2c_loss(train_episodes, clone, baseline, gamma, tau)
    gradients = autograd.grad(loss,
                              clone.parameters(),
                              retain_graph=second_order,
                              create_graph=second_order)
    return maml_update(clone, adapt_lr, gradients)



def main(env_name = 'HalfCheetahForwardBackward-v5', seed = 42, num_workers = 10, num_iterations = 5, meta_bsz = 3, adapt_steps = 1, adapt_bsz = 10,
         adapt_lr= 0.05, gamma=0.99, tau=0.95, cuda = 0):
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

            # Fast Adapt
            for step in range(adapt_steps):
                train_episodes = task.run(clone, episodes=adapt_bsz)
                if cuda:
                    train_episodes = train_episodes.to(device, non_blocking=True)
                clone = fast_adapt_a2c(clone, train_episodes, adapt_lr,
                                    baseline, gamma, tau, first_order=True)
                task_replay.append(train_episodes)

            # Compute Validation Loss
            valid_episodes = task.run(clone, episodes=adapt_bsz)
            task_replay.append(valid_episodes)
            iteration_reward += valid_episodes.reward().sum().item() / adapt_bsz
            iteration_replays.append(task_replay)
            iteration_policies.append(clone)

if __name__ == '__main__':
    main()