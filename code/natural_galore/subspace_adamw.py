import torch
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Tuple, Union, Iterable, Callable
from .galore_projector_tensor import GaLoreProjectorTensor

torch_compile_options = {
    "epilogue_fusion": False,
    "max_autotune": False,
    "shape_padding": True,
    "trace.enabled": False,  # Output Triton kernel outputs!
    "triton.cudagraphs": True,
}

class SubSpaceAdamW(Optimizer):
    def __init__(
        self,
        params: Union[Iterable[nn.parameter.Parameter], Iterable[dict]],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        natural_history: int = 20,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        correct_bias: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            natural_history=natural_history,
            correct_bias=correct_bias,
            amsgrad=amsgrad,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)
        self.init_lr = lr

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            for p in group["params"]:
                state = self.state[p]
                if len(state) != 0 and not torch.is_tensor(state["step"]):
                    state["step"] = torch.tensor(float(state["step"]))

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            max_exp_avg_sqs: List[Tensor] = []
            state_steps: List[Tensor] = []
            projectors: List[Optional[GaLoreProjectorTensor]] = []

            amsgrad = group["amsgrad"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                # Initialize state step
                if "step" not in state:
                    state["step"] = (
                        torch.zeros((), dtype=torch.float32, device=p.device)
                        if group["capturable"]
                        else torch.tensor(0.0)
                    )

                # GaLore Projection
                if "rank" in group:
                    if "projector" not in state:
                        state["projector"] = GaLoreProjectorTensor(
                            rank=group["rank"],
                            update_proj_gap=group["update_proj_gap"],
                            scale=group["scale"],
                            proj_type=group["proj_type"],
                            natural_history=group["natural_history"],
                        )
                    grad = state["projector"].project(
                        grad,
                        int(state["step"].item()),
                        lr_ratio=group["lr"] / self.init_lr,
                    )
                else:
                    state["projector"] = None

                params_with_grad.append(p)
                grads.append(grad)
                projectors.append(state["projector"])

                # State initialization
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(
                        grad, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        grad, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            grad, memory_format=torch.preserve_format
                        )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                if amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                state_steps.append(state["step"])

            # Perform optimization step
            adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                projectors,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                correct_bias=group["correct_bias"],
            )

        return loss

def adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    projectors: List[Optional[GaLoreProjectorTensor]],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    correct_bias: bool,
):
    """Functional API that performs AdamW algorithm computation with projection.

    Args:
        params (List[Tensor]): List of parameters.
        grads (List[Tensor]): List of gradients.
        exp_avgs (List[Tensor]): Exponential moving averages of gradients.
        exp_avg_sqs (List[Tensor]): Exponential moving averages of squared gradients.
        max_exp_avg_sqs (List[Tensor]): Max exponential moving averages of squared gradients.
        state_steps (List[Tensor]): List of step counts.
        projectors (List[Optional[GaLoreProjector]]): List of projectors per parameter.
        amsgrad (bool): Whether to use AMSGrad variant.
        beta1 (float): Coefficient for computing running averages of gradient.
        beta2 (float): Coefficient for computing running averages of squared gradient.
        lr (float): Learning rate.
        weight_decay (float): Weight decay coefficient.
        eps (float): Term added to the denominator to improve numerical stability.
        maximize (bool): Maximize the params based on the objective, instead of minimizing.
        capturable (bool): Whether to use capturable version.
        differentiable (bool): Whether to use differentiable version.
    """
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # Update step
        step_t += 1

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        # Compute denominator
        denom = exp_avg_sq.sqrt().add_(eps)

        # Compute step size
        step = step_t.item()
        step_size = lr
        if correct_bias:
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            step_size = step_size * (bias_correction2**0.5) / bias_correction1

        # Compute normalized gradient
        norm_grad = exp_avg / denom

        # GaLore Projection Back
        if projectors[i] is not None:
            norm_grad = projectors[i].project_back(norm_grad)

        # Update parameter
        param.add_(norm_grad, alpha=-step_size)

        # Perform step weight decay
        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want to decay the weights in a manner that doesn't interact
        # with the m/v parameters. This is equivalent to adding the square
        # of the weights to the loss with plain (non-momentum) SGD.
        # Add weight decay at the end (fixed version)
        if weight_decay > 0.0:
            param.add_(param, alpha=(-lr * weight_decay))
