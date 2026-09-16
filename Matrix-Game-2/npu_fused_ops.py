"""Matrix-Game 2.0 在昇腾上的融合算子替换(由环境变量开关,默认不启用)。

两处替换都必须和上游保持完全相同的数学语义。

1) RMSNorm
   - wan.modules.model.WanRMSNorm(主干 q/k 归一化):上游是「转 float32 算归一化,转回原精度,再乘权重」。
     rms_mode="bf16":照抄昇腾官方 Self-Forcing 补丁,npu_rms_norm(x, weight) 在原精度一次算完。
     rms_mode="fp32":复刻上游顺序,float32 归一化(gamma 全 1),转回原精度后再乘权重。
   - wan.modules.action_module.WanRMSNorm(动作模块):上游前向**不乘权重**(参数定义了但没用)。
     所以 gamma 必须传全 1。传 self.weight 会悄悄改掉语义。

2) 旋转位置编码 causal_rope_apply
   上游把相邻两个元素配成一个复数,乘以频率复数。这等价于 npu_rotary_mul 的 interleave 模式:
   r1 = cos、r2 = sin,各自按相邻两份复制展开到 D。
   约束(jit_compile=False):input 为 BSND,B*N<1000,D<896 且为偶数,r1/r2 必须是 1S1D。
   本模型 B=1、N=12、D=128,满足。
   rope_mode="fp32":在 float32 下算(与当前生产路径实际精度一致);"bf16":直接在原精度算。
   cos/sin 表从模型自己的 freqs 取实部虚部,只缓存最近一条(见 _ROPE 处的说明)。
"""
import weakref

import torch
import torch_npu

_ONES = {}

# cos/sin 只留最近一条。key 里带 start_frame,而 start_frame 每块 +num_frame_per_block
# 单调递增(causal_inference.py 里不回绕,local_attn_size 只裁 KV 缓存),
# 所以跨块必然 miss,留多条毫无收益、只会按块漏显存:
# 本模型 f*h*w=2640、D=128,cos+sin 每条约 2.7 MB,约 370 次按键就是 1 GB。
# 命中只发生在同一块内的 8 次调用(q/k × 去噪步数),单条足够。
# freqs 用 weakref 比对本体,不能用 data_ptr —— 指针会在张量释放后被复用,
# 可能给另一张 freqs 表返回旧缓存。
_ROPE = {"key": None, "freqs": None, "val": None}


def _ones(dim, dtype, device):
    k = (dim, dtype, str(device))
    t = _ONES.get(k)
    if t is None:
        t = torch.ones(dim, dtype=dtype, device=device)
        _ONES[k] = t
    return t


def main_rms_bf16(self, x):
    return torch_npu.npu_rms_norm(x, self.weight.to(x.dtype), epsilon=self.eps)[0]


def main_rms_fp32(self, x):
    y = torch_npu.npu_rms_norm(x.float(), _ones(x.shape[-1], torch.float32, x.device), epsilon=self.eps)[0]
    return y.type_as(x) * self.weight


def action_rms(self, x):
    return torch_npu.npu_rms_norm(x, _ones(x.shape[-1], x.dtype, x.device), epsilon=self.eps)[0]


def _cos_sin(freqs, f, h, w, start_frame, c, device, dtype):
    k = (start_frame, f, h, w, c, str(device), dtype,
         tuple(freqs.shape), freqs.dtype)
    held = _ROPE["freqs"]
    if _ROPE["key"] == k and held is not None and held() is freqs:
        return _ROPE["val"]
    fc = freqs.detach().cpu()
    parts = fc.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    fi = torch.cat([
        parts[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
        parts[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        parts[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, -1)
    s = f * h * w
    cos = fi.real.to(torch.float64).repeat_interleave(2, dim=-1).view(1, s, 1, -1)
    sin = fi.imag.to(torch.float64).repeat_interleave(2, dim=-1).view(1, s, 1, -1)
    hit = (cos.to(dtype).to(device), sin.to(dtype).to(device))
    _ROPE.update(key=k, freqs=weakref.ref(freqs), val=hit)
    return hit


def fused_causal_rope_apply(x, grid_sizes, freqs, start_frame=0, mode="fp32"):
    c = x.size(3) // 2
    f, h, w = grid_sizes.tolist()
    s = f * h * w
    work = torch.float32 if mode == "fp32" else x.dtype
    r1, r2 = _cos_sin(freqs, f, h, w, start_frame, c, x.device, work)
    out = torch_npu.npu_rotary_mul(x[:, :s].to(work).contiguous(), r1, r2, rotary_mode="interleave")
    if s < x.size(1):
        out = torch.cat([out, x[:, s:].to(work)], dim=1)
    return out.type_as(x)


def install(rms_mode=None, rope_mode=None):
    """在模型构造前后调用均可:RMSNorm 改的是类方法,rope 改的是模块级函数。"""
    from wan.modules import model as M, causal_model as CM, action_module as AM
    done = []
    if rms_mode:
        M.WanRMSNorm.forward = main_rms_fp32 if rms_mode == "fp32" else main_rms_bf16
        AM.WanRMSNorm.forward = action_rms
        done.append(f"rms={rms_mode}")
    if rope_mode:
        def _rope(x, grid_sizes, freqs, start_frame=0):
            return fused_causal_rope_apply(x, grid_sizes, freqs, start_frame, rope_mode)
        CM.causal_rope_apply = _rope
        done.append(f"rope={rope_mode}")
    return done
