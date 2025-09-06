import torch as th
from torch import nn

class LinearValue(nn.Module):

    """
    <a href="https://github.com/seba-1511/cherry/blob/master/cherry/models/robotics.py" class="source-link">[Source]</a>

    ## Description

    A linear state-value function, whose parameters are found by minimizing
    least-squares.

    ## Credit

    Adapted from Tristan Deleu's implementation.

    ## References

    1. Duan et al. 2016. “Benchmarking Deep Reinforcement Learning for Continuous Control.”
    2. [https://github.com/tristandeleu/pytorch-maml-rl](https://github.com/tristandeleu/pytorch-maml-rl)

    ## Example

    ~~~python
    states = replay.state()
    rewards = replay.reward()
    dones = replay.done()
    returns = ch.td.discount(gamma, rewards, dones)
    baseline = LinearValue(input_size)
    baseline.fit(states, returns)
    next_values = baseline(replay.next_states())
    ~~~
    """

    def __init__(self, input_size, reg=1e-5):
        """
        ## Arguments

        * `inputs_size` (int) - Size of input.
        * `reg` (float, *optional*, default=1e-5) - Regularization coefficient.
        """
        super(LinearValue, self).__init__()
        self.linear = nn.Linear(2 * input_size + 4, 1, bias=False)
        self.reg = reg

    def _features(self, states):
        length = states.size(0)
        ones = th.ones(length, 1).to(states.device)
        al = th.arange(length, dtype=th.float32, device=states.device).view(-1, 1) / 100.0
        return th.cat([states, states**2, al, al**2, al**3, ones], dim=1)

    def fit(self, states, returns):
        """
        ## Description

        Fits the parameters of the linear model by the method of least-squares.

        ## Arguments

        * `states` (tensor) - States collected with the policy to evaluate.
        * `returns` (tensor) - Returns associated with those states (ie, discounted rewards).
        """
        features = self._features(states)
        reg = self.reg * th.eye(features.size(1))
        reg = reg.to(states.device)
        A = features.t() @ features + reg
        b = features.t() @ returns
        if hasattr(th, 'linalg') and hasattr(th.linalg, 'lstsq'):
            coeffs = th.linalg.lstsq(A, b).solution
        elif hasattr(th, 'lstsq'):  # Required for torch < 1.3.0
            coeffs, _ = th.lstsq(b, A)
        else:
            coeffs, _ = th.gels(b, A)
        self.linear.weight.data = coeffs.data.t()

    def forward(self, state):
        """
        ## Description

        Computes the value of a state using the linear function approximator.

        ## Arguments

        * `state` (Tensor) - The state to evaluate.
        """
        features = self._features(state)
        return self.linear(features)
