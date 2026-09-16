#!/usr/bin/env python
"""Matrix-Game 2.0 交互式世界模型 —— 昇腾 NPU 上的单进程可玩服务端。

在装好 CANN 与 torch_npu 的容器里直接运行,浏览器打开就能玩:
模型只加载一次,每按一个键生成约 1 秒画面(12 帧 @ 12fps),逐隐帧推给页面。

    python serve.py --mode templerun --port 8800

上游 `inference_streaming.py` 只能在终端里逐段敲键、结果写成 mp4,没法边玩边看。
这里在运行时替换流水线里的两个函数,不改动上游源码:

- `get_current_action`:改成从命令队列取一个按键。上游是 while + input() + 裸 except,
  管道 EOF 会被吞掉导致空转死循环。
- `process_video`:改成把刚解码出的帧编码成 JPEG 放进内存,由 HTTP 取,不再写 mp4。

三处提速都默认开启,每一处都用「与未开启时逐字节相同」验收过:
融合位置编码、出画不等重算 KV、逐隐帧流式解码。
"""
import argparse
import collections
import io
import json
import os
import queue
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MG2_ROOT", "/workspace/Matrix-Game/Matrix-Game-2")
WEIGHTS = os.environ.get("MG2_WEIGHTS", "/weights/Matrix-Game-2.0")

CONFIGS = {
    "templerun": ("configs/inference_yaml/inference_templerun.yaml",
                  "templerun_distilled_model/templerun_7dim_onlykey.safetensors"),
    "universal": ("configs/inference_yaml/inference_universal.yaml",
                  "base_distilled_model/base_distill.safetensors"),
    "gta_drive": ("configs/inference_yaml/inference_gta_drive.yaml",
                  "gta_distilled_model/gta_keyboard2dim.safetensors"),
}

# 按键映射逐字照抄上游 pipeline/causal_inference.py 的 get_current_action
CAM = 0.1
MAPS = {
    "templerun": {
        "keyboard": {"w": [0, 1, 0, 0, 0, 0, 0], "s": [0, 0, 1, 0, 0, 0, 0],
                     "a": [0, 0, 0, 0, 0, 1, 0], "d": [0, 0, 0, 0, 0, 0, 1],
                     "z": [0, 0, 0, 1, 0, 0, 0], "c": [0, 0, 0, 0, 1, 0, 0],
                     "q": [1, 0, 0, 0, 0, 0, 0]},
        "mouse": None},
    "universal": {
        "keyboard": {"w": [1, 0, 0, 0], "s": [0, 1, 0, 0], "a": [0, 0, 1, 0],
                     "d": [0, 0, 0, 1], "q": [0, 0, 0, 0]},
        "mouse": {"i": [CAM, 0], "k": [-CAM, 0], "j": [0, -CAM], "l": [0, CAM], "u": [0, 0]}},
    "gta_drive": {
        "keyboard": {"w": [1, 0], "s": [0, 1], "q": [0, 0]},
        "mouse": {"a": [0, -CAM], "d": [0, CAM], "q": [0, 0]}},
}
DEFAULT_MOUSE = {"universal": "u", "gta_drive": "q"}

DEMO_IMAGES = {"templerun": "demo_images/temple_run/0000.png",
               "universal": "demo_images/universal/0000.png",
               "gta_drive": "demo_images/gta_drive/0000.png"}


class Hub:
    """进程内的唯一状态:命令队列、帧仓、浏览器订阅者。"""

    def __init__(self, mode):
        self.mode = mode
        self.cmds = queue.Queue()
        self.clients = []
        self.lock = threading.Lock()
        self.frames = collections.OrderedDict()     # (段号, 帧号) -> JPEG 字节
        self.snapshot = {"type": "status", "phase": "loading", "text": "正在加载模型…"}
        self.ready_msg = None
        self.block = 0
        self.t_key = 0.0
        self.key = "?"
        self.chunk = {"enc_ms": 0, "bytes": 0, "n": 0}

    def broadcast(self, msg):
        if msg.get("type") in ("status", "ready", "session", "done", "closed"):
            self.snapshot = msg
        if msg.get("type") == "ready":
            self.ready_msg = msg
        data = json.dumps(msg, ensure_ascii=False)
        for q in list(self.clients):
            q.put(data)

    def put_frames(self, idx, base, jpegs):
        urls = []
        with self.lock:
            for j, b in enumerate(jpegs):
                self.frames[(idx, base + j)] = b
                urls.append("/frame/%d/%d.jpg" % (idx, base + j))
            while len(self.frames) > 8 * 12:          # 只留最近几段,防止内存无限涨
                self.frames.popitem(last=False)
        return urls

    def get_frame(self, idx, j):
        with self.lock:
            return self.frames.get((idx, j))


HUB = None


class ResetSession(Exception):
    pass


def build_action(mode, kb, ms):
    import torch
    m = MAPS[mode]
    act = {"keyboard": torch.tensor(m["keyboard"][kb]).cuda()}
    if m["mouse"] is not None:
        act["mouse"] = torch.tensor(m["mouse"][ms or DEFAULT_MOUSE[mode]]).cuda()
    return act


def get_current_action(mode="templerun"):
    """上游每段开头调用它取动作。这里改成等浏览器按键,而不是等终端输入。"""
    HUB.broadcast({"type": "wait", "idx": HUB.block})
    while True:
        msg = HUB.cmds.get()
        c = msg.get("cmd")
        if c == "RESET":
            raise ResetSession()
        if c == "KEY":
            kb = str(msg.get("k", "")).lower()
            ms = str(msg["m"]).lower() if msg.get("m") else None
            try:
                act = build_action(mode, kb, ms)
            except KeyError:
                HUB.broadcast({"type": "err", "text": "非法按键 %s %s" % (kb, ms or "")})
                continue
            HUB.key = kb if ms is None else "%s+%s" % (kb, ms)
            HUB.t_key = time.time()
            import fast_sched
            fast_sched.block_begin()
            return act
        HUB.broadcast({"type": "err", "text": "此刻只接受 KEY 或 RESET"})


def _encode(frames):
    """一组帧编码成 JPEG 字节列表。"""
    from PIL import Image
    out = []
    for f in frames:
        buf = io.BytesIO()
        Image.fromarray(f).save(buf, format="JPEG", quality=78)
        out.append(buf.getvalue())
    return out


def emit_chunk(frames, base, last):
    """逐隐帧出画:解完一个隐帧就把它那几张推给页面,让页面不必等整段解完。"""
    ready_ms = int((time.time() - HUB.t_key) * 1000)
    t0 = time.time()
    jpegs = _encode(frames)
    enc_ms = int((time.time() - t0) * 1000)
    HUB.chunk["enc_ms"] += enc_ms
    HUB.chunk["bytes"] += sum(len(b) for b in jpegs)
    HUB.chunk["n"] += len(jpegs)
    urls = HUB.put_frames(HUB.block, base, jpegs)
    HUB.broadcast({"type": "chunk", "idx": HUB.block, "base": base,
                   "urls": urls, "ready_ms": ready_ms})
    if last:
        HUB.broadcast({"type": "block", "idx": HUB.block, "gen_ms": ready_ms,
                       "enc_ms": HUB.chunk["enc_ms"], "key": HUB.key,
                       "n": HUB.chunk["n"], "bytes": HUB.chunk["bytes"]})
        HUB.block += 1
        HUB.emitted = True
        HUB.chunk = {"enc_ms": 0, "bytes": 0, "n": 0}


def emit_video(frames):
    """整段一次出画。只在关掉流式解码时走这条。"""
    emit_chunk(frames, 0, True)


def process_video(video, output_path, *args, **kwargs):
    """替换上游的存盘函数:整局结尾那两次全量编码跳过,不写 mp4。"""
    if not str(output_path).endswith("_current.mp4"):
        return
    if getattr(HUB, "emitted", False):
        HUB.emitted = False
        return          # 调度重排已经提前出过画,这里不再重复


def run_session(gi, S, mode, img_path, num_blocks):
    import torch
    import fast_sched
    fast_sched.session_begin()          # 清 VAE 时间维缓存,否则第二局会接着上一局
    max_frames = num_blocks * 3         # 隐空间帧数;配置里 num_frame_per_block = 3
    t0 = time.time()
    image = S.load_image(img_path)
    image = gi._resizecrop(image, 352, 640)
    image = gi.frame_process(image)[None, :, None, :, :].to(dtype=gi.weight_dtype, device=gi.device)
    padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (max_frames - 1), 1, 1)
    img_cond = torch.concat([image, padding_video], dim=2)
    tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
    img_cond = gi.vae.encode(img_cond, device=gi.device, **tiler_kwargs).to(gi.device)
    mask_cond = torch.ones_like(img_cond)
    mask_cond[:, :, 1:] = 0
    cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
    visual_context = gi.vae.clip.encode_video(image)
    sampled_noise = torch.randn([1, 16, max_frames, 44, 80], device=gi.device, dtype=gi.weight_dtype)
    num_frames = (max_frames - 1) * 4 + 1
    cd = {"cond_concat": cond_concat.to(device=gi.device, dtype=gi.weight_dtype),
          "visual_context": visual_context.to(device=gi.device, dtype=gi.weight_dtype)}
    if mode == "universal":
        cond = S.Bench_actions_universal(num_frames)
        cd["mouse_cond"] = cond["mouse_condition"].unsqueeze(0).to(device=gi.device, dtype=gi.weight_dtype)
    elif mode == "gta_drive":
        cond = S.Bench_actions_gta_drive(num_frames)
        cd["mouse_cond"] = cond["mouse_condition"].unsqueeze(0).to(device=gi.device, dtype=gi.weight_dtype)
    else:
        cond = S.Bench_actions_templerun(num_frames)
    cd["keyboard_cond"] = cond["keyboard_condition"].unsqueeze(0).to(device=gi.device, dtype=gi.weight_dtype)
    HUB.broadcast({"type": "session", "blocks": num_blocks,
                   "prep_ms": int((time.time() - t0) * 1000)})
    with torch.no_grad():
        gi.pipeline.inference(noise=sampled_noise, conditional_dict=cd, return_latents=False,
                              output_folder="/tmp/mg2", name="play", mode=mode)
    HUB.broadcast({"type": "done"})


def worker(args):
    """模型加载 + 命令主循环,独占那张 NPU。"""
    t0 = time.time()
    os.makedirs("/tmp/mg2", exist_ok=True)
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.path.insert(1, HERE)
    # 上游逐段的 "Continue?" 提问:补丁在该变量存在时短路
    os.environ["MG2_KEY"] = "q"
    import inference_streaming as S
    from pipeline import causal_inference as CI
    CI.get_current_action = get_current_action
    CI.process_video = process_video

    cfg, ckpt = CONFIGS[args.mode]
    cfg = args.config or cfg
    if args.steps == 3 and args.mode == "templerun":
        cfg = "configs/inference_yaml/inference_templerun_3step.yaml"

    import npu_fused_ops
    done = npu_fused_ops.install(None, "fp32" if args.fused_rope else None)
    print("[serve] 融合算子:%s" % (",".join(done) or "无"), flush=True)

    ns = argparse.Namespace(config_path=cfg, checkpoint_path=os.path.join(WEIGHTS, ckpt),
                            pretrained_model_path=WEIGHTS, max_num_output_frames=45,
                            output_folder="/tmp/mg2", seed=args.seed)
    S.set_seed(args.seed)
    gi = S.InteractiveGameInference(ns)
    mode = gi.config.pop("mode")

    import fast_sched
    n_steps = len(gi.pipeline.denoising_step_list)
    fast_sched.install(gi.pipeline, emit_video, n_steps,
                       emit_chunk_fn=emit_chunk if args.stream_decode else None)
    load_ms = int((time.time() - t0) * 1000)
    print("[serve] 就绪,加载 %.1f 秒,去噪 %d 步" % (load_ms / 1000, n_steps), flush=True)
    HUB.broadcast({"type": "ready", "mode": mode, "load_ms": load_ms})

    while True:
        msg = HUB.cmds.get()
        c = msg.get("cmd")
        if c == "RESET":
            HUB.broadcast({"type": "reset_ok"})
            continue
        if c != "START":
            HUB.broadcast({"type": "err", "text": "请先开局,收到 %s" % c})
            continue
        img = msg.get("img") or DEMO_IMAGES[mode]
        nb = max(2, min(60, int(msg.get("blocks", 15))))
        HUB.block = 0
        try:
            run_session(gi, S, mode, img, nb)
        except ResetSession:
            HUB.broadcast({"type": "reset_ok"})
        except Exception as e:
            traceback.print_exc()
            HUB.broadcast({"type": "err", "text": "%s: %s" % (type(e).__name__, str(e)[:300])})


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = os.path.join(HERE, "index.html")
            return self._send(200, open(page, "rb").read(), "text/html; charset=utf-8")
        if self.path.startswith("/frame/"):
            try:
                blk, rest = self.path[len("/frame/"):].split("/")
                data = HUB.get_frame(int(blk), int(rest.split(".")[0]))
            except Exception:
                data = None
            if data is None:
                return self._send(404, "no frame", "text/plain")
            return self._send(200, data, "image/jpeg")
        if self.path == "/events":
            return self._events()
        self._send(404, "not found", "text/plain")

    def _events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        q = queue.Queue()
        HUB.clients.append(q)
        try:
            # 新开的页面先补发「就绪」,再补发最后一条状态,否则会一直显示加载中
            if HUB.ready_msg is not None and HUB.snapshot is not HUB.ready_msg:
                self.wfile.write(("data: %s\n\n" % json.dumps(HUB.ready_msg, ensure_ascii=False)).encode())
            hello = dict(HUB.snapshot)
            hello["mode"] = HUB.mode
            self.wfile.write(("data: %s\n\n" % json.dumps(hello, ensure_ascii=False)).encode())
            self.wfile.flush()
            while True:
                try:
                    self.wfile.write(("data: %s\n\n" % q.get(timeout=15)).encode())
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            if q in HUB.clients:
                HUB.clients.remove(q)

    def do_POST(self):
        if self.path != "/cmd":
            return self._send(404, "not found", "text/plain")
        n = int(self.headers.get("Content-Length", "0"))
        try:
            msg = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, json.dumps({"ok": False}), "application/json")
        if msg.get("cmd") not in ("START", "KEY", "RESET"):
            return self._send(400, json.dumps({"ok": False, "err": "未知命令"}), "application/json")
        HUB.cmds.put(msg)
        self._send(200, json.dumps({"ok": True}), "application/json")


def main():
    p = argparse.ArgumentParser(description="Matrix-Game 2.0 交互式世界模型服务端")
    p.add_argument("--mode", default=os.environ.get("MG2_MODE", "templerun"),
                   choices=sorted(CONFIGS), help="场景:神庙逃亡 / 通用 / GTA 驾驶")
    p.add_argument("--port", type=int, default=int(os.environ.get("MG2_PORT", "8800")))
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--steps", type=int, default=4, choices=(3, 4), help="去噪步数")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", default=None, help="覆盖推理 yaml 路径")
    p.add_argument("--no-fused-rope", dest="fused_rope", action="store_false",
                   help="关掉融合位置编码(会慢一倍,仅用于对照)")
    p.add_argument("--no-stream-decode", dest="stream_decode", action="store_false",
                   help="关掉逐隐帧流式解码(整段解完再出画,仅用于对照)")
    args = p.parse_args()

    global HUB
    HUB = Hub(args.mode)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print("[serve] 网页已开:http://%s:%d  (场景 %s)" % (args.host, args.port, args.mode), flush=True)
    print("[serve] 正在加载模型,首次约需一到两分钟", flush=True)
    worker(args)


if __name__ == "__main__":
    main()
