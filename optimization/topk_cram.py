"""
Code adapted from: https://github.com/davda54/sam/blob/main/sam.py

"""
import numpy as np

import torch
import torch.nn as nn
from utils import percentile

import pdb

class TopkCrAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, sparsities=[0.5], grad_norm=False, plus_version=False, 
                 unif_prune=False, sparse_grad=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, sparsities=sparsities, grad_norm=grad_norm, plus_version=plus_version, 
                        unif_prune=unif_prune, sparse_grad=sparse_grad, **kwargs)
        super(TopkCrAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)


    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = 1.
        if self.param_groups[0]["grad_norm"]:
            grad_norm = self._grad_norm()

        for group in self.param_groups:
            if group["grad_norm"]:
                grad_norm += 1e-12
            scale = group["rho"] / grad_norm 
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                if group["plus_version"]:
                    self.state[p]["clean_grad"] = p.grad.clone()
                e_w = p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        
        sparsities = self.param_groups[0]["sparsities"]
        k = np.random.choice(sparsities)

        # now compute C(w + e(w))
        if self.param_groups[0]["unif_prune"]:
            self._sparsify_weights_unif(k)
        else:
            self._sparsify_weights(k)

        if zero_grad: self.zero_grad()
        

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None: continue
                # mask the gradients in case we want them to be sparse
                if group['sparse_grad'] and (len(p.shape)>1):
                    mask = 1. - (p.data==0.).float()
                    p.grad.mul_(mask)
                p.data = self.state[p]["old_p"]  # get back to "w" from "C(w + e(w))"
                # add the gradient of the dense model in case of CrAM+
                if group["plus_version"]:
                    p.grad.add_(self.state[p]["clean_grad"])
        self.base_optimizer.step()  # do the actual "compression-aware" update
        
        self.update_state_dict()
        
        if zero_grad: self.zero_grad()


    def update_state_dict(self):
        # update the momentum buffer in the main optimizer state_dict
        # this is only to make sure the momentum is included in case of restarting from checkpoint
        for group in self.param_groups:
            for p in group["params"]:
                self.state[p].update(self.base_optimizer.state[p])



    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(torch.stack([p.grad.norm(p=2).to(shared_device) for group in self.param_groups for p in group["params"] if p.grad is not None]), p=2)
        return norm


    @torch.no_grad()
    def _sparsify_weights(self, k):
        for i, group in enumerate(self.param_groups):
            if i>0: 
                break
            params_stats = None
            for p in group["params"]:
                if len(p.shape)>1:
                    p_stats = p.data.abs().view(-1)
                    if params_stats is None:
                        params_stats = p_stats
                    else:
                        params_stats = torch.cat((params_stats, p_stats))
            threshold = percentile(params_stats, k)
            for p in group["params"]:
                if len(p.shape)>1:
                    mask = (p.data.abs()>threshold).float()
                    p.mul_(mask)


    @torch.no_grad()
    def _sparsify_weights_unif(self, k):
        for j, group in enumerate(self.param_groups):
            if j>0:
                break
            params = list(group["params"])
            for i in range(1, len(params)-2):
                if len(params[i].shape)>1:
                    p_stats = params[i].data.abs().view(-1)
                    threshold = percentile(p_stats, k)
                    mask = (params[i].data.abs() > threshold).float() 
                    params[i].mul_(mask)


    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


    def get_sparsity(self):
        for group in self.param_groups:
            total_zeros = 0.
            total_params = 0.
            for p in group["params"]:
                total_zeros_p = (p.data==0.).float().sum().item()
                total_zeros += total_zeros_p
                total_params += p.data.numel()
            print('sparsity model: ', total_zeros/total_params)
