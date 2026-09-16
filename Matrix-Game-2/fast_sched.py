"""调度重排:让画面不必等「重算 KV 缓存」那一次 DiT 前向。

上游每段的顺序是:4 次去噪 → 1 次用干净上下文重算 KV → VAE 解码 → 出画。
但重算 KV 只为下一段准备状态,与本段画面没有数据依赖(已核代码 Step 3.3 / 3.4)。
把出画提到重算之前,按实测口径可省下约一次前向的时间。

实现方式是运行时挂钩,不重写上游循环:
- 包住 generator.forward,数每段的调用次数;第 n_steps 次(最后一次去噪)算完后,
  就地做 VAE 解码并回调 emit 出画。
- 包住 vae_decoder.forward 做记忆化:上游循环后面那次解码直接返回已算好的结果,不重复解码。
- VAE 缓存由本模块自己持有,与上游传进来的保持同一血缘(上游会把我们返回的缓存存下再传回)。

数学过程不变,因此可以用「与基线逐字节相同」作为验收判据。
"""
import torch

STATE = {
    "calls": 0,          # 本段内 generator 调用次数
    "n_steps": 4,        # 去噪步数,install 时传入
    "pending": None,     # 已提前算好的 (video, cache)
    "vae_cache": None,   # 本模块持有的 VAE 缓存
    "emit": None,        # 出画回调
    "early_ms": 0,       # 提前解码耗时,供计时用
    "emit_chunk": None,  # 逐隐帧出画回调,开了就走流式解码
}


def block_begin():
    """每段开始(取到按键)时调用,重置计数。"""
    STATE["calls"] = 0
    STATE["pending"] = None


def session_begin():
    """每局开始时调用,清掉 VAE 的时间维缓存。

    上游每次 `inference()` 都会把 vae_cache 重新置成全 None,而本模块自己持有一份缓存、
    不理会上游传进来的那份,所以必须在这里跟着清。不清的话第二局会接着上一局的缓存解码:
    第 0 段会吐出 12 张而不是 9 张(因果 VAE 的首个隐帧本该只出 1 张),
    且开局画面混进上一局的残留。
    """
    STATE["vae_cache"] = None
    block_begin()


def _to_frames(video):
    """与上游出画前的转换逐步对齐:换轴 -> (x+1)*127.5 -> 裁剪 -> uint8 -> 连续内存。
    上游在 causal_inference.py 里就是这么做完再交给 process_video 的,这里必须一致,
    否则出的图不是同一个东西(第一次实现漏了这步,PIL 直接报 __array_interface__)。"""
    import numpy as np
    v = video.permute(0, 1, 3, 4, 2)                      # B,T,C,H,W -> B,T,H,W,C
    v = ((v.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)[0]
    return np.ascontiguousarray(v)


def _decode(vae_module, denoised_pred):
    import time
    t0 = time.time()
    z = denoised_pred.transpose(1, 2).half()      # 与上游一致:B,C,F,H,W -> B,F,C,H,W
    cache = STATE["vae_cache"]
    if cache is None:
        from demo_utils.constant import ZERO_VAE_CACHE
        cache = [None] * len(ZERO_VAE_CACHE)
    video, cache = vae_module._orig_forward(z, *cache)
    STATE["vae_cache"] = list(cache)
    STATE["early_ms"] = int((time.time() - t0) * 1000)
    return video, cache


def _decode_stream(vae_module, denoised_pred, emit_chunk):
    """逐隐帧解码:解出一个隐帧就立刻发它那几张画面,让回传与后面的解码重叠。

    切分点就是上游 `VAEDecoderWrapper.forward` 自己的 `for i in range(iter_)` 循环边界:
    每次只喂一个隐帧、把 feat_cache 串下去,调用的仍是上游原函数,所以逐位等价。
    已核两处前提:wrapper 里的 conv2 是 kernel=1 的 CausalConv3d,时间维 padding 为 0,
    帧与帧之间不耦合;缩放、clamp、换轴全是逐元素运算,按帧做与整段做结果相同。
    唯一由本函数补回的是末尾那次 `torch.cat`,只为把整段结果交还给上游循环。
    """
    import time
    import torch
    t0 = time.time()
    z = denoised_pred.transpose(1, 2).half()      # 与上游一致:B,C,F,H,W -> B,F,C,H,W
    cache = STATE["vae_cache"]
    if cache is None:
        from demo_utils.constant import ZERO_VAE_CACHE
        cache = [None] * len(ZERO_VAE_CACHE)
    outs, base, n_lat = [], 0, z.shape[1]
    for i in range(n_lat):
        out_, cache = vae_module._orig_forward(z[:, i:i + 1], *cache)
        cache = list(cache)
        outs.append(out_)
        emit_chunk(_to_frames(out_), base, i == n_lat - 1)
        base += out_.shape[1]
    STATE["vae_cache"] = cache
    STATE["early_ms"] = int((time.time() - t0) * 1000)
    return torch.cat(outs, dim=1), cache


def install(pipeline, emit_fn, n_steps, emit_chunk_fn=None):
    """emit_fn(frames) 由调用方负责发出整段;传了 emit_chunk_fn 就改走逐隐帧流式解码。"""
    STATE["n_steps"] = n_steps
    STATE["emit"] = emit_fn
    STATE["emit_chunk"] = emit_chunk_fn

    gen = pipeline.generator
    vae = pipeline.vae_decoder
    if getattr(gen, "_fast_sched", False):
        return
    gen._orig_forward = gen.forward
    vae._orig_forward = vae.forward

    def gen_forward(*a, **k):
        out = gen._orig_forward(*a, **k)
        STATE["calls"] += 1
        if STATE["calls"] == STATE["n_steps"]:
            # 最后一次去噪刚算完:此时就能出画,不必等下一次「重算 KV」的前向
            denoised = out[1] if isinstance(out, (tuple, list)) else out
            if STATE["emit_chunk"] is not None:
                video, cache = _decode_stream(vae, denoised, STATE["emit_chunk"])
                STATE["pending"] = (video, cache)
            else:
                video, cache = _decode(vae, denoised)
                STATE["pending"] = (video, cache)
                if STATE["emit"] is not None:
                    STATE["emit"](_to_frames(video))
        return out

    def vae_forward(z, *cache):
        if STATE["pending"] is not None:
            video, c = STATE["pending"]
            STATE["pending"] = None
            return video, c          # 上游这次解码直接取已算好的结果
        return vae._orig_forward(z, *cache)

    gen.forward = gen_forward
    vae.forward = vae_forward
    gen._fast_sched = True
    return True
