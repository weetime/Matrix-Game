"""Matrix-Game 2.0 昇腾适配垫片。

替换 `flash_attn.flash_attn_func`。MG2 的三处调用都是裸 (q, k, v),
不传 causal / window_size / 掩码 —— 窗口是靠显式切 KV 缓存实现的,
所以这里是精确替换,不存在丢掩码的语义偏移。

布局:MG2 传入 [B, S, H, D](BSND)。
"""
import math
import torch

try:
    import torch_npu
    _NPU = True
except ImportError:
    _NPU = False


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None,
                    causal=False, window_size=(-1, -1), **kwargs):
    if causal or window_size != (-1, -1):
        raise NotImplementedError(
            "该垫片只覆盖 MG2 实际用到的无掩码路径;调用方传了 causal/window_size,"
            "说明上游改了语义,必须重新核对而不是静默近似。")
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    if _NPU and torch.npu.is_available():
        return torch_npu.npu_fusion_attention(
            q, k, v, q.shape[2],
            pse=None, padding_mask=None, atten_mask=None,
            scale=scale, keep_prob=1 - dropout_p,
            input_layout="BSND",
        )[0]
    # 非昇腾环境走精确 SDPA,便于在 A100 上做同代码路径对照
    out = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        dropout_p=dropout_p, scale=scale)
    return out.transpose(1, 2)
