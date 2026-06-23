import math
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim import AdamW

from llava.train.stiefel import (
    euclidean2riemannian_U,
    euclidean2riemannian_Vt,
    exp_map,
    polar_retraction,
    polar_retraction_U,
    polar_retraction_Vt,
    tangent_project_U,
    tangent_project_Vt,
)



class RiemannianAdamW(AdamW):
    def __init__(self, params, *,
                 lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, gradient_prev_task=None, grad_scaling=None):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.grad_scaling = grad_scaling
        self._load_F(gradient_prev_task)



    def _initialize_adamw_state(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def _load_F(self, grad_prev_task):
        if grad_prev_task is None:
            return

        # calculate the maximum column number of all F to align
        max_p = 0
        for F in grad_prev_task.values():
            if F.shape[1] > max_p:
                max_p = F.shape[1]
        
        max_p = ((max_p + 7) // 8) * 8

        for p, F in grad_prev_task.items():
            if not isinstance(p, torch.nn.Parameter):
                raise ValueError(
                    f"[ERROR] gradient_prev_task keys must be torch.nn.Parameter, but got {type(p)}"
                )
            if p not in self.state:
                self.state[p] = {}

            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(p)
            if "exp_avg_sq" not in state:
                state["exp_avg_sq"] = torch.zeros_like(p)

            if F.shape[1] < max_p:
                F_padded = torch.nn.functional.pad(F, (0, max_p - F.shape[1]))
                state["F"] = F_padded.to(p.device, p.dtype).detach()
            else:
                state["F"] = F.to(p.device, p.dtype).detach()
                
        print(f"F_load with alignment to rank {max_p}!!!")


    @torch.no_grad()
    def pre_step(self):
        self._cache = {}
        stiefel_groups = defaultdict(list)
        
        for group in self.param_groups:
            tag = group.get("lora_tag", "")
            if tag not in ["U", "Vt"]: continue
            for p in group['params']:
                if p.grad is None: continue
                # record cache for post_step
                self._cache[p] = p.data.detach().clone()
                
                calc_shape = p.shape if tag == "U" else (p.shape[1], p.shape[0])
                stiefel_groups[(calc_shape, tag)].append(p)

        for (shape, tag), ps in stiefel_groups.items():
            if tag == "U":
                x = torch.stack([p.data for p in ps])
                g = torch.stack([p.grad.data for p in ps])
                g_riem = euclidean2riemannian_U(x, g)
            else:
                # Vt branch: directly stack F after alignment
                x = torch.stack([p.data.mT for p in ps])
                g = torch.stack([p.grad.data.mT for p in ps])
                
                f_list = [self.state[p].get("F", None) for p in ps]
                
                fs = torch.stack(f_list) if f_list[0] is not None else None
                g_riem = euclidean2riemannian_Vt(x, g, fs)

            # batch write back
            for i, p in enumerate(ps):
                p.grad.data.copy_(g_riem[i] if tag == "U" else g_riem[i].mT)
        
        stiefel_groups.clear()

    @torch.no_grad()
    def post_step(self):
        if not self._cache:
            return

        stiefel_groups = defaultdict(list)
        for group in self.param_groups:
            tag = group.get("lora_tag", "")
            if tag not in ["U", "Vt"]: continue
            for p in group['params']:
                if p in self._cache:
                    calc_shape = p.shape if tag == "U" else (p.shape[1], p.shape[0])
                    stiefel_groups[(calc_shape, tag)].append(p)

        c = self.grad_scaling if isinstance(self.grad_scaling, float) else None

        for (shape, tag), ps in stiefel_groups.items():
            pre_ps_list = [self._cache[p] for p in ps]
            if tag == "U":
                x_pre = torch.stack(pre_ps_list)
                x_now = torch.stack([p.data for p in ps])
            else:
                x_pre = torch.stack([p.mT for p in pre_ps_list])
                x_now = torch.stack([p.data.mT for p in ps])

            delta = x_now - x_pre
            if c is not None and shape[1] != c:
                delta.div_(math.sqrt(shape[1] / c))

            if tag == "U":
                delta_proj = tangent_project_U(x_pre, delta)
                x_new = polar_retraction_U(x_pre, delta_proj)
            else:
                # batch get F after alignment
                f_list = [self.state[p].get("F", None) for p in ps]
                fs = torch.stack(f_list) if f_list[0] is not None else None
                
                delta_proj = tangent_project_Vt(x_pre, delta, fs)
                x_new = polar_retraction_Vt(x_pre, delta_proj, fs)

            for i, p in enumerate(ps):
                p.data.copy_(x_new[i] if tag == "U" else x_new[i].mT)

        self._cache.clear()
        stiefel_groups.clear()


    def step(self, closure=None):
        self.pre_step()
        loss = super().step(closure)
        self.post_step()
        return loss
  