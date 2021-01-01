"""
Network definitions from https://github.com/ferrine/hyrnn
"""

import geoopt
import geoopt.manifolds.stereographic.math as gmath
import numpy as np
import torch.nn
import torch.nn.functional
from torch.cuda.amp import autocast


def mobius_linear(
        input,
        weight,
        bias=None,
        hyperbolic_input=True,
        hyperbolic_bias=True,
        nonlin=None,
        k=-1.0,
):
    k = torch.tensor(k)
    if hyperbolic_input:
        output = mobius_matvec(weight, input, k=k)
    else:
        output = torch.nn.functional.linear(input, weight)
        output = gmath.expmap0(output, k=k)
    if bias is not None:
        if not hyperbolic_bias:
            bias = gmath.expmap0(bias, k=k)
        output = gmath.mobius_add(output, bias.unsqueeze(0).expand_as(output), k=k)
    if nonlin is not None:
        output = gmath.mobius_fn_apply(nonlin, output, k=k)
    output = gmath.project(output, k=k)
    return output


def mobius_matvec(m: torch.Tensor, x: torch.Tensor, *, k: torch.Tensor, dim=-1):
    return _mobius_matvec(m, x, k, dim=dim)


def _mobius_matvec(m: torch.Tensor, x: torch.Tensor, k: torch.Tensor, dim: int = -1):
    if m.dim() > 2 and dim != -1:
        raise RuntimeError(
            "broadcasted Möbius matvec is supported for the last dim only"
        )
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    if dim != -1 or m.dim() == 2:
        # mx = torch.tensordot(x, m, [dim], [1])
        mx = torch.matmul(m, x.transpose(1, 0)).transpose(1, 0)
    else:
        mx = torch.matmul(m, x.unsqueeze(-1)).squeeze(-1)
    mx_norm = mx.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    res_c = gmath.tan_k(mx_norm / x_norm * gmath.artan_k(x_norm, k), k) * (mx / mx_norm)
    cond = (mx == 0).prod(dim=dim, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return res


def one_rnn_transform(W, h, U, x, b, k):
    W_otimes_h = gmath.mobius_matvec(W, h, k=k)
    U_otimes_x = gmath.mobius_matvec(U, x, k=k)
    Wh_plus_Ux = gmath.mobius_add(W_otimes_h, U_otimes_x, k=k)
    return gmath.mobius_add(Wh_plus_Ux, b, k=k)


def mobius_gru_cell(
        input: torch.Tensor,
        hx: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias: torch.Tensor,
        k: torch.Tensor,
        nonlin=None,
):
    W_ir, W_ih, W_iz = weight_ih.chunk(3)
    b_r, b_h, b_z = bias
    W_hr, W_hh, W_hz = weight_hh.chunk(3)

    z_t = gmath.logmap0(one_rnn_transform(W_hz, hx, W_iz, input, b_z, k), k=k).sigmoid()
    r_t = gmath.logmap0(one_rnn_transform(W_hr, hx, W_ir, input, b_r, k), k=k).sigmoid()

    rh_t = gmath.mobius_pointwise_mul(r_t, hx, k=k)
    h_tilde = one_rnn_transform(W_hh, rh_t, W_ih, input, b_h, k)

    if nonlin is not None:
        h_tilde = gmath.mobius_fn_apply(nonlin, h_tilde, k=k)
    delta_h = gmath.mobius_add(-hx, h_tilde, k=k)
    h_out = gmath.mobius_add(hx, gmath.mobius_pointwise_mul(z_t, delta_h, k=k), k=k)
    return h_out


def mobius_gru_loop(
        input: torch.Tensor,
        h0: torch.Tensor,
        weight_ih: torch.Tensor,
        weight_hh: torch.Tensor,
        bias: torch.Tensor,
        k: torch.Tensor,
        batch_sizes=None,
        hyperbolic_input: bool = False,
        hyperbolic_hidden_state0: bool = False,
        nonlin=None,
):
    if not hyperbolic_hidden_state0:
        hx = gmath.expmap0(h0, k=k)
    else:
        hx = h0
    if not hyperbolic_input:
        input = gmath.expmap0(input, k=k)
    outs = []
    if batch_sizes is None:
        input_unbinded = input.unbind(0)
        for t in range(input.size(0)):
            hx = mobius_gru_cell(
                input=input_unbinded[t],
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin,
                k=k,
            )
            outs.append(hx)
        outs = torch.stack(outs)
        h_last = hx
    else:
        h_last = []
        T = len(batch_sizes) - 1
        for i, t in enumerate(range(batch_sizes.size(0))):
            ix, input = input[: batch_sizes[t]], input[batch_sizes[t]:]
            hx = mobius_gru_cell(
                input=ix,
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin,
                k=k,
            )
            outs.append(hx)
            if t < T:
                hx, ht = hx[: batch_sizes[t + 1]], hx[batch_sizes[t + 1]:]
                h_last.append(ht)
            else:
                h_last.append(hx)
        h_last.reverse()
        h_last = torch.cat(h_last)
        outs = torch.cat(outs)
    return outs, h_last


class MobiusLinear(torch.nn.Linear):
    def __init__(
            self,
            *args,
            hyperbolic_input=True,
            hyperbolic_bias=True,
            nonlin=None,
            k=-1.0,
            fp64_hyper=True,
            **kwargs
    ):
        k = torch.tensor(k)
        super().__init__(*args, **kwargs)
        if self.bias is not None:
            if hyperbolic_bias:
                self.ball = manifold = geoopt.PoincareBall(c=k.abs())
                self.bias = geoopt.ManifoldParameter(self.bias, manifold=manifold)
                with torch.no_grad():
                    # self.bias.set_(gmath.expmap0(self.bias.normal_() / 4, k=k))
                    self.bias.set_(gmath.expmap0(self.bias.normal_() / 400, k=k))
        with torch.no_grad():
            # 1e-2 was the original value in the code. The updated one is from HNN++
            std = 1 / np.sqrt(2 * self.weight.shape[0] * self.weight.shape[1])
            # Actually, we divide that by 100 so that it starts really small and far from the border
            std = std / 100
            self.weight.normal_(std=std)
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin
        self.k = k
        self.fp64_hyper = fp64_hyper

    def forward(self, input):
        if self.fp64_hyper:
            input = input.double()
        else:
            input = input.float()
        with autocast(enabled=False):  # Do not use fp16
            return mobius_linear(
                input,
                weight=self.weight,
                bias=self.bias,
                hyperbolic_input=self.hyperbolic_input,
                nonlin=self.nonlin,
                hyperbolic_bias=self.hyperbolic_bias,
                k=self.k,
            )

    def extra_repr(self):
        info = super().extra_repr()
        info += "c={}, hyperbolic_input={}".format(self.ball.c, self.hyperbolic_input)
        if self.bias is not None:
            info = ", hyperbolic_bias={}".format(self.hyperbolic_bias)
        return info


class MobiusDist2Hyperplane(torch.nn.Module):
    def __init__(self, in_features, out_features, k=-1.0, fp64_hyper=True):
        k = torch.tensor(k)
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = ball = geoopt.PoincareBall(c=k.abs())
        self.sphere = sphere = geoopt.manifolds.Sphere()
        self.scale = torch.nn.Parameter(torch.zeros(out_features))
        point = torch.randn(out_features, in_features) / 4
        point = gmath.expmap0(point, k=k)
        tangent = torch.randn(out_features, in_features)
        self.point = geoopt.ManifoldParameter(point, manifold=ball)
        self.fp64_hyper = fp64_hyper
        with torch.no_grad():
            self.tangent = geoopt.ManifoldParameter(tangent, manifold=sphere).proj_()

    def forward(self, input):
        if self.fp64_hyper:
            input = input.double()
        else:
            input = input.float()
        with autocast(enabled=False):  # Do not use fp16
            input = input.unsqueeze(-2)
            distance = gmath.dist2plane(
                x=input, p=self.point, a=self.tangent, k=self.ball.c, signed=True
            )
            return distance * self.scale.exp()

    def extra_repr(self):
        return (
            "in_features={in_features}, out_features={out_features}"
            #             "c={ball.c}".format(
            #                 **self.__dict__
            #             )
        )
