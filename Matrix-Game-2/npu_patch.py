"""Matrix-Game 2.0 昇腾适配:旋转位置编码。

改动来源:MindSpeed-MM examples/self_forcing/npu_adapt/patch.py。
但**不能照搬** —— 官方那份是按 Self-Forcing 写的,它的 grid_sizes 是「每样本一个三元组」,
而 MG2 的 grid_sizes 是「单个三元组、循环走批次」,照搬会报
`TypeError: cannot unpack non-iterable int object`(已实测)。

因此这里保留 MG2 自己的函数体,只加入昇腾真正需要的那一处修改:
把三个频率切片显式转成 complex64。其余一字未改。
"""
import torch
import torch_npu  # noqa: F401
from wan.modules import causal_model, model


def _replace(mod, name):
    def deco(fn):
        setattr(mod, name, fn)
        return fn
    return deco


@_replace(causal_model, 'causal_rope_apply')
def npu_causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    f, h, w = grid_sizes.tolist()          # MG2 口径:单个三元组
    for i in range(len(x)):                # MG2 口径:循环走批次
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1).to(torch.complex64),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1).to(torch.complex64),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1).to(torch.complex64),
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


@_replace(model, 'rope_apply')
def npu_rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    f, h, w = grid_sizes.tolist()
    for i in range(len(x)):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1).to(torch.complex64),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1).to(torch.complex64),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1).to(torch.complex64),
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)
