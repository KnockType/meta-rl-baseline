import random
import torch
import wandb
import yaml
import multiprocessing
import gymnasium as gym
import numpy as np
import algorithms.trpo as trpo
from tqdm import tqdm
from copy import deepcopy
from torch import autograd
from random import choice
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from custom_gym.async_vec_env import AsyncVectorEnv
from ch.envs import ActionSpaceScaler
from ch.torch_wrapper import Torch
from policies import DiagNormalPolicy
from ch.models.robotics import LinearValue
from ch.runner_wrapper import Runner
from ch._torch import normalize
from ch.td import discount
from ch.pg import generalized_advantage
from ch.algorithms.a2c import policy_loss as a2c_policy_loss
from algorithms.trpo import policy_loss as trpo_policy_loss
from algorithms.maml import maml_update
from utils import clone_module
from custom_gym.envs.metaworld.ML1 import MetaWorldML1


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
    return a2c_policy_loss(log_probs, advantages)


def fast_adapt_a2c(clone, train_episodes, adapt_lr, baseline, gamma, tau, first_order=False):
    second_order = not first_order
    loss = maml_a2c_loss(train_episodes, clone, baseline, gamma, tau)
    gradients = autograd.grad(loss,
                              clone.parameters(),
                              retain_graph=second_order,
                              create_graph=second_order)
    return maml_update(clone, adapt_lr, gradients)

def meta_surrogate_loss(iteration_replays, iteration_policies, policy, baseline, tau, gamma, adapt_lr):
    mean_loss = 0.0
    mean_kl = 0.0
    for task_replays, old_policy in tqdm(zip(iteration_replays, iteration_policies),
                                         total=len(iteration_replays),
                                         desc='Surrogate Loss',
                                         leave=False):
        train_replays = task_replays[:-1]
        valid_episodes = task_replays[-1]
        new_policy = clone_module(policy)

        # Fast Adapt
        for train_episodes in train_replays:
            new_policy = fast_adapt_a2c(new_policy, train_episodes, adapt_lr,
                                        baseline, gamma, tau, first_order=False)

        # Useful values
        states = valid_episodes.state()
        actions = valid_episodes.action()
        next_states = valid_episodes.next_state()
        rewards = valid_episodes.reward()
        terminated = valid_episodes.terminated()
        truncated = valid_episodes.truncated()

        # Compute KL
        old_densities = old_policy.density(states)
        new_densities = new_policy.density(states)
        kl = kl_divergence(new_densities, old_densities).mean()
        mean_kl += kl

        # Compute Surrogate Loss
        advantages = compute_advantages(baseline, tau, gamma, rewards, terminated, truncated, states, next_states)
        advantages = normalize(advantages).detach()
        old_log_probs = old_densities.log_prob(actions).mean(dim=1, keepdim=True).detach()
        new_log_probs = new_densities.log_prob(actions).mean(dim=1, keepdim=True)
        mean_loss += trpo_policy_loss(new_log_probs, old_log_probs, advantages)
    mean_kl /= len(iteration_replays)
    mean_loss /= len(iteration_replays)
    return mean_loss, mean_kl

def make_env(benchmark, seed, num_workers, test=False):
    def init_env():
        if test:
            env = benchmark.test_classes['push-v3']()
        else:
            env = benchmark.train_classes['push-v3']()
        env = ActionSpaceScaler(env)
        return env
    
    env = AsyncVectorEnv([init_env for _ in range(num_workers)])
    if test:
        task = choice([task for task in benchmark.test_tasks if task.env_name == 'push-v3'])
    else:
        task = choice([task for task in benchmark.train_tasks if task.env_name == 'push-v3'])
    env.set_task(task)
    env.reset(seed)
    env = Torch(env)
    return env

def main(benchmark = MetaWorldML1(env_name='push-v3'), gpu_id=5):
    wandb.init()
    config = wandb.config
    env = make_env(benchmark, config.seed, config.num_workers)
    
    cuda = bool(config.cuda)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device_name = 'cpu'
    if cuda:
        torch.cuda.manual_seed(config.seed)
        device_name = f'cuda:{gpu_id}'
    device = torch.device(device_name)

    policy = DiagNormalPolicy(env.state_size, env.action_size, device=device)
    if cuda:
        policy = policy.to(device)
    baseline = LinearValue(env.state_size, env.action_size)
    all_rewards = []

    for iteration in range(config.num_iterations):
        iteration_reward = 0.0
        iteration_replays = []
        iteration_policies = []

        for task_config in tqdm(benchmark.sample_tasks(config.meta_bsz), leave=False, desc='Data'):  # Samples a new config
            clone = deepcopy(policy)
            env.set_task(task_config)
            env.reset()
            task = Runner(env)
            task_replay = []

            # Fast Adapt
            for step in range(config.adapt_steps):
                train_episodes = task.run(clone, episodes=config.adapt_bsz)
                if cuda:
                    train_episodes = train_episodes.to(device, non_blocking=True)
                clone = fast_adapt_a2c(clone, train_episodes, config.adapt_lr,
                                    baseline, config.gamma, config.tau, first_order=True)
                task_replay.append(train_episodes)

            # Compute Validation Loss
            valid_episodes = task.run(clone, episodes=config.adapt_bsz)
            task_replay.append(valid_episodes)
            iteration_reward += valid_episodes.reward().sum().item() / config.adapt_bsz
            iteration_replays.append(task_replay)
            iteration_policies.append(clone)
        
        # Print statistics
        print('\nIteration', iteration)
        adaptation_reward = iteration_reward / config.meta_bsz
        print('adaptation_reward', adaptation_reward)

        all_rewards.append(adaptation_reward)
        wandb.log({"adaptation_reward": adaptation_reward}) 

        # TRPO meta-optimization
        backtrack_factor = 0.5
        ls_max_steps = 15
        max_kl = 0.01
        if cuda:
            policy = policy.to(device, non_blocking=True)
            baseline = baseline.to(device, non_blocking=True)
            iteration_replays = [[r.to(device, non_blocking=True) for r in task_replays] for task_replays in
                                iteration_replays]


        # Compute CG step direction
        old_loss, old_kl = meta_surrogate_loss(iteration_replays, iteration_policies, policy, baseline, config.tau, config.gamma, config.adapt_lr)
        grad = autograd.grad(old_loss,
                            policy.parameters(),
                            retain_graph=True)
        grad = parameters_to_vector([g.detach() for g in grad])
        Fvp = trpo.hessian_vector_product(old_kl, policy.parameters())
        step = trpo.conjugate_gradient(Fvp, grad)
        shs = 0.5 * torch.dot(step, Fvp(step))
        lagrange_multiplier = torch.sqrt(shs / max_kl)
        step = step / lagrange_multiplier
        step_ = [torch.zeros_like(p.data) for p in policy.parameters()]
        vector_to_parameters(step, step_)
        step = step_
        del old_kl, Fvp, grad
        old_loss.detach_()

        # Line-search
        for ls_step in range(ls_max_steps):
            stepsize = backtrack_factor ** ls_step * config.meta_lr
            clone = deepcopy(policy)
            for p, u in zip(clone.parameters(), step):
                p.data.add_(u.data, alpha=-stepsize)
            new_loss, kl = meta_surrogate_loss(iteration_replays, iteration_policies, clone, baseline, config.tau, config.gamma, config.adapt_lr)
            if new_loss < old_loss and kl < max_kl:
                for p, u in zip(policy.parameters(), step):
                    p.data.add_(u.data, alpha=-stepsize)
                break
    
    # Evaluate on a set of unseen tasks
    #evaluate(benchmark, policy, baseline, config.adapt_lr, config.gamma, config.tau, config.num_workers, config.seed, cuda, config.cuda_device)
    wandb.finish()

def evaluate(benchmark, policy, baseline, adapt_lr, gamma, tau, n_workers, seed, cuda, cuda_device):
    device_name = 'cpu'
    if cuda:
        device_name = f'cuda:{cuda_device}'
    device = torch.device(device_name)

    # Parameters
    adapt_steps = 3
    adapt_bsz = 10
    n_eval_tasks = 10

    tasks_reward = 0.

    env = make_env(benchmark, seed, n_workers, test=True)
    eval_task_list = benchmark.sample_tasks(n_eval_tasks)

    for i, task in enumerate(eval_task_list):
        clone = deepcopy(policy)
        env.set_task(task)
        env.reset()
        task = Runner(env)

        # Adapt
        for step in range(adapt_steps):
            adapt_episodes = task.run(clone, episodes=adapt_bsz)
            if cuda:
                adapt_episodes = adapt_episodes.to(device, non_blocking=True)
            clone = fast_adapt_a2c(clone, adapt_episodes, adapt_lr, baseline, gamma, tau, first_order=True)

        eval_episodes = task.run(clone, episodes=adapt_bsz)

        task_reward = eval_episodes.reward().sum().item() / adapt_bsz
        print(f"Reward for task {i} : {task_reward}")
        tasks_reward += task_reward

    final_eval_reward = tasks_reward / n_eval_tasks

    print(f"Average reward over {n_eval_tasks} test tasks: {final_eval_reward}")
    
    return final_eval_reward


def run_agent(sweep_id, count, gpu_id):
    wandb.agent(sweep_id, function=lambda: main(gpu_id=gpu_id), count=count)

with open("sweep_metaworld.yaml") as file:
    sweep_config = yaml.safe_load(file)
    sweep_id = wandb.sweep(sweep=sweep_config, project="maml-trpo-metaworld")
    NUM_AGENTS = 3
    COUNT = 5
    processes = []
    for i in range(NUM_AGENTS):
        p = multiprocessing.Process(target=run_agent, args=(sweep_id, COUNT, 4))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()