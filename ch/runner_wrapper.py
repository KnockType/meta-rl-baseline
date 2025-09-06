#!/usr/bin/env python3
import torch
from ch._utils import _min_size, _istensorable
from .env.utils import is_vectorized
from .base_wrapper import Wrapper
from .experience_replay import ExperienceReplay

def flatten_episodes(replay, episodes, num_workers):
    #  TODO: This implementation is not efficient.

    #  NOTE: Additional info (other than a transition's default fields) is simply copied.
    #  To know from which worker the data was gathered, you can access sars.runner_id
    #  TODO: This is not great. What is the best behaviour with infos here ?
    flat_replay = ExperienceReplay()
    worker_replays = [ExperienceReplay() for w in range(num_workers)]
    flat_episodes = 0
    for sars in replay:
        state = sars.state.view(_min_size(sars.state))
        action = sars.action.view(_min_size(sars.action))
        reward = sars.reward.view(_min_size(sars.reward))
        next_state = sars.next_state.view(_min_size(sars.next_state))
        terminated = sars.terminated.view(_min_size(sars.terminated))
        truncated = sars.truncated.view(_min_size(sars.truncated))
        fields = set(sars._fields) - {'state', 'action', 'reward', 'next_state', 'terminated', 'truncated'}
        infos = {f: getattr(sars, f) for f in fields}
        for worker in range(num_workers):
            infos['runner_id'] = worker
            # The following attemps to split additional infos. (WIP. Remove ?)
            # infos = {}
            # for f in fields:
            #     value = getattr(sars, f)
            #     if isinstance(value, Iterable) and len(value) == num_workers:
            #         value = value[worker]
            #     elif _istensorable(value):
            #         tvalue = ch.totensor(value)
            #         tvalue = tvalue.view(_min_size(tvalue))
            #         if tvalue.size(0) == num_workers:
            #             value = tvalue[worker]
            #     infos[f] = value
            worker_replays[worker].append(state[worker],
                                          action[worker],
                                          reward[worker],
                                          next_state[worker],
                                          terminated[worker],
                                          truncated[worker],
                                          **infos,
                                          )
            if bool(terminated[worker] or truncated[worker]):
                flat_replay += worker_replays[worker]
                worker_replays[worker] = ExperienceReplay()
                flat_episodes += 1
            if flat_episodes >= episodes:
                break
        if flat_episodes >= episodes:
            break
    return flat_replay


class Runner(Wrapper):

    """
    <a href="" class="source-link">[Source]</a>

    ## Description

    Helps collect transitions, given a `get_action` function.

    ## Example

    ~~~python
    env = MyEnv()
    env = Runner(env)
    replay = env.run(lambda x: policy(x), steps=100)
    # or
    replay = env.run(lambda x: policy(x), episodes=5)
    ~~~

    """
    #  TODO: When is_vectorized and using episodes=n, use the parallel
    #  environmnents to sample n episodes, and stack them inside a flat replay.

    def __init__(self, env):
        super(Runner, self).__init__(env)
        self.env = env
        self._needs_reset = True
        self._current_state = None

    def reset(self, *args, **kwargs):
        self._current_state = self.env.reset(*args, **kwargs)
        self._needs_reset = False
        return self._current_state

    def step(self, action, *args, **kwargs):
        # TODO: Implement it to be compatible with .run()
        raise NotImplementedError('Runner does not currently support step.')

    def run(self,
            get_action,
            steps=None,
            episodes=None,
            render=False):
        """
        ## Description

        Runner wrapper's run method.

        !!! info
            Either use the `steps` OR the `episodes` argument.

        ## Arguments

        * `get_action` (function) - Given a state, returns the action to be taken.
        * `steps` (int, *optional*, default=None) - The number of steps to be collected.
        * `episodes` (int, *optional*, default=None) - The number of episodes to be collected.
        """

        if steps is None:
            steps = float('inf')
            if self.is_vectorized:
                self._needs_reset = True
        elif episodes is None:
            episodes = float('inf')
        else:
            msg = 'Either steps or episodes should be set.'
            raise Exception(msg)

        replay = ExperienceReplay(vectorized=self.is_vectorized)
        collected_episodes = 0
        collected_steps = 0
        while True:
            if collected_steps >= steps or collected_episodes >= episodes:
                if self.is_vectorized and collected_episodes >= episodes:
                    replay = flatten_episodes(replay, episodes, self.num_envs)
                    self._needs_reset = True
                return replay
            if self._needs_reset:
                self.reset()
            info = {}
            action = get_action(self._current_state)
            if isinstance(action, tuple):
                skip_unpack = False
                if self.is_vectorized:
                    if len(action) > 2:
                        skip_unpack = True
                    elif len(action) == 2 and \
                            self.env.num_envs == 2 and \
                            not isinstance(action[1], dict):
                        # action[1] is not info but an action
                        action = (action, )

                if not skip_unpack:
                    if len(action) == 2:
                        info = action[1]
                        action = action[0]
                    elif len(action) == 1:
                        action = action[0]
                    else:
                        msg = 'get_action should return 1 or 2 values.'
                        raise NotImplementedError(msg)
            old_state = self._current_state
            state, reward, terminated, truncated, _ = self.env.step(action)
            if not self.is_vectorized and (terminated or truncated):
                collected_episodes += 1
                self._needs_reset = True
            elif self.is_vectorized:
                collected_episodes += sum(torch.logical_or(terminated, truncated))
            replay.append(old_state, action, reward, state, terminated, truncated, **info)
            self._current_state = state
            if render:
                self.env.render()
            collected_steps += 1