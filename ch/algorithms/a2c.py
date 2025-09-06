import torch

def policy_loss(log_probs, q_curr, alpha=1.0):
        """
        ## Description

        The policy loss of the Soft Actor-Critic.

        New actions are sampled from the target policy, and those are used to compute the Q-values.
        While we should back-propagate through the Q-values to the policy parameters, we shouldn't
        use that gradient to optimize the Q parameters.
        This is often avoided by either using a target Q function, or by zero-ing out the gradients
        of the Q function parameters.

        ## Arguments

        * `log_probs` (tensor) - Log-density of the selected actions.
        * `q_curr` (tensor) - Q-values of state-action pairs.
        * `alpha` (float, *optional*, default=1.0) - Entropy weight.

        ## Returns

        * (tensor) - The policy loss for the given arguments.

        ## Example

        ~~~python
        densities = policy(batch.state())
        actions = densities.sample()
        log_probs = densities.log_prob(actions)
        q_curr = q_function(batch.state(), actions)
        loss = policy_loss(log_probs, q_curr, alpha=0.1)
        ~~~

        """
        msg = 'log_probs and q_curr must have equal size.'
        assert log_probs.size() == q_curr.size(), msg
        return torch.mean(alpha * log_probs - q_curr)
