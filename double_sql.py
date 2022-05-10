import torch

from pfrl.agents import dqn
from pfrl.utils import evaluating
from pfrl.utils.recurrent import pack_and_forward


class DoubleSQL(dqn.DQN):
    """Double Soft Q-Learning.
    See: https://arxiv.org/pdf/1702.08165.
    Based on pfrl.agents.double_dqn and pfrl.agents.dpp.
    """
    
    def __init__(self, *args, **kwargs):
        self.alpha = kwargs.pop("alpha", 1.0)
        super().__init__(*args, **kwargs)

    def _l_operator(self, qout):
        return self.alpha * torch.logsumexp((1 / self.alpha) * qout.q_values, dim=1)

    def _compute_target_values(self, exp_batch):

        batch_next_state = exp_batch["next_state"]

        with evaluating(self.model):
            if self.recurrent:
                next_qout, _ = pack_and_forward(
                    self.model,
                    batch_next_state,
                    exp_batch["next_recurrent_state"],
                )
            else:
                next_qout = self.model(batch_next_state)

        if self.recurrent:
            target_next_qout, _ = pack_and_forward(
                self.target_model,
                batch_next_state,
                exp_batch["next_recurrent_state"],
            )
        else:
            target_next_qout = self.target_model(batch_next_state)

        next_q_expect = self._l_operator(target_next_qout)

        batch_rewards = exp_batch["reward"]
        batch_terminal = exp_batch["is_state_terminal"]
        discount = exp_batch["discount"]

        return batch_rewards + discount * (1.0 - batch_terminal) * next_q_expect
