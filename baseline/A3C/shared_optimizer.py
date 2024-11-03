"""
Shared optimizer, the parameters in the optimizer will shared in the multiprocessors.
"""

import torch
class SharedOptimizer(torch.optim.RMSprop):
    def __init__(self, params, lr=1e-3,centered=False):
        super(SharedOptimizer, self).__init__(params)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['square_avg'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['square_avg'].share_memory_()
