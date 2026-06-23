import torch
from torch import Tensor
from typing import Optional


def symm(A: Tensor) -> Tensor:
    return 0.5 * (A + A.transpose(-2, -1))


@torch.jit.script
def euclidean2riemannian(x: Tensor, grad: Tensor) -> Tensor:
    return grad - x @ (grad.transpose(-2, -1) @ x)


def euclidean2riemannian_U(x: Tensor, grad: Tensor) -> Tensor:
    return grad - x @ (grad.transpose(-2, -1) @ x)


def euclidean2riemannian_Vt(x: Tensor, grad: Tensor, F: Optional[Tensor]) -> Tensor:
    if F is None:
        return euclidean2riemannian_U(x, grad)
    # adapt batch: first do F constraint projection, then do Stiefel projection
    # use associative law F @ (F.T @ grad) to improve batch calculation efficiency
    grad = grad - F @ (F.transpose(-2, -1) @ grad)
    return grad - x @ (grad.transpose(-2, -1) @ x)


@torch.jit.script
def tangent_project(x: Tensor, grad: Tensor) -> Tensor:
    return grad - x @ (0.5 * (x.transpose(-2, -1) @ grad + grad.transpose(-2, -1) @ x))


def tangent_project_U(x: Tensor, grad: Tensor) -> Tensor:
    return grad - x @ (grad.transpose(-2, -1) @ x)


def tangent_project_Vt(x: Tensor, grad: Tensor, F: Optional[Tensor]) -> Tensor:
    if F is None:
        return tangent_project_U(x, grad)
    # batch tangent projection with F constraint
    grad = grad - F @ (F.transpose(-2, -1) @ grad)
    return grad - x @ (grad.transpose(-2, -1) @ x)


@torch.jit.script
def exp_map(X: Tensor, grad: Tensor) -> Tensor:
    # batch exponential mapping
    xTgrad = X.transpose(-2, -1) @ grad
    Q, R = torch.linalg.qr(grad - X @ xTgrad)
    Z = torch.zeros_like(R)
    
    # adapt batch unit matrix extension
    Id = torch.eye(Z.shape[-2], device=Z.device).expand(Z.shape[0], -1, -1)
    
    top_row = torch.cat([xTgrad, -R.transpose(-2, -1)], dim=-1)
    bottom_row = torch.cat([R, Z], dim=-1)
    matrix = torch.cat([top_row, bottom_row], dim=-2)
    
    exp_mat = torch.linalg.matrix_exp(matrix)
    IZ = torch.cat([Id, Z], dim=-2)
    MN = exp_mat @ IZ
    XQ = torch.cat([X, Q], dim=-1)
    out = XQ @ MN
    return out


def polar_uf(m: Tensor) -> Tensor:
    # torch.linalg.svd natively supports batch [B, N, P]
    with torch.no_grad():
        U, _, Vt = torch.linalg.svd(m, full_matrices=False)
        return U @ Vt



@torch.jit.script
def polar_retraction(X: Tensor, grad: Tensor) -> Tensor:
    return polar_uf(X + grad)


def polar_retraction_U(X: Tensor, grad: Tensor) -> Tensor:
    return polar_uf(X + grad)


def polar_retraction_Vt(X: Tensor, grad: Tensor, F: Optional[Tensor]) -> Tensor:
    if F is None:
        return polar_retraction_U(X, grad)
    new_V = X + grad
    # batch F constraint reflection
    new_V = new_V - F @ (F.transpose(-2, -1) @ new_V)
    return polar_uf(new_V)



# import torch
# from torch import Tensor


# def symm(A):
#     return 0.5 * (A + A.t())


# @torch.jit.script
# def euclidean2riemannian(x: Tensor, grad: Tensor) -> Tensor:
#     return grad - x @ (grad.t() @ x)

# def euclidean2riemannian_U(x: Tensor, grad: Tensor) -> Tensor:
#     return grad - x @ (grad.t() @ x)

# def euclidean2riemannian_Vt(x: Tensor, grad: Tensor, F: Tensor) -> Tensor:
#     # x: d*r, grad: d*r, F: d*p
#     # output: d*r
#     if F is None:
#         return euclidean2riemannian_U(x, grad)
#     grad = grad - F @ F.t() @ grad
#     return grad - x @ (grad.t() @ x)

# @torch.jit.script
# def tangent_project(x: Tensor, grad: Tensor) -> Tensor:
#     return grad - x @ symm(x.t() @ grad)

# def tangent_project_U(x: Tensor, grad: Tensor) -> Tensor:
#     return grad - x @ (grad.t() @ x)

# def tangent_project_Vt(x: Tensor, grad: Tensor, F: Tensor) -> Tensor:
#     if F is None:
#         return tangent_project_U(x, grad)
#     grad = grad - F @ F.t() @ grad
#     return grad - x @ (grad.t() @ x)



# @torch.jit.script
# def exp_map(X: Tensor, grad: Tensor) -> Tensor:
#     xTgrad = X.t() @ grad
#     Q, R = torch.linalg.qr(grad - X @ xTgrad)
#     Z = torch.zeros_like(R)
#     Id = torch.eye(Z.shape[-2], device=Z.device)[None].expand(Z.shape[0], -1, -1)
#     top_row = torch.cat([xTgrad, -R.t()], dim=-1)
#     bottom_row = torch.cat([R, Z], dim=-1)
#     matrix = torch.cat([top_row, bottom_row], dim=-2)
#     exp_mat = torch.linalg.matrix_exp(matrix)
#     IZ = torch.cat([Id, Z], dim=-2)
#     MN = exp_mat @ IZ
#     XQ = torch.cat([X, Q], dim=-1)
#     out = XQ @ MN
#     return out


# def polar_uf(m: Tensor) -> Tensor:
#     U, _, Vt = torch.linalg.svd(m, full_matrices=False, driver='gesvda')
#     uf = U @ Vt
#     return uf


# @torch.jit.script
# def polar_retraction(X: Tensor, grad: Tensor) -> Tensor:
#     return polar_uf(X + grad)

# def polar_retraction_U(X: Tensor, grad: Tensor) -> Tensor:
#     return polar_uf(X + grad)

# def polar_retraction_Vt(X: Tensor, grad: Tensor, F: Tensor) -> Tensor:
#     if F is None:
#         return polar_retraction_U(X, grad)
#     new_V = X + grad
#     new_V = new_V - F @ F.t() @ new_V
#     return polar_uf(new_V)
