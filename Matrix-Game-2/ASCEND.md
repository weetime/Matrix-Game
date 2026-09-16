# Running Matrix-Game 2.0 on Ascend 910B

Two layers sit on top of upstream. The first makes it run; the second makes it playable.
Everything was measured on a single Ascend 910B3 (64 GB), CANN 9.0.1,
torch 2.10.0+cpu with torch_npu 2.10.0.post2, Python 3.12, Ubuntu 22.04 aarch64.

## Layer 1 — the port

Six files changed, +103/-8 lines, plus `npu_patch.py` and `npu_shim.py`.
Model structure, weights and the sampling procedure are untouched; only operator
calls and entry points change. See the `ascend-port` branch for that diff alone.

One trap worth repeating: `torch.cuda.is_available()` cannot gate "are we on Ascend".
`transfer_to_npu` redirects `torch.cuda.*` to `torch.npu.*`, so it returns True and any
CUDA-only branch behind it still executes. Use `torch_npu.npu.is_available()`.

## Layer 2 — making it playable

Right after the port, one key press took 2.33 s of server-side generation. Three changes
bring that to 0.92 s. All three leave the generated frames **byte-identical** to the
unoptimised path (verified over 69 frames), because none of them touch the math.

| change | where | effect |
|---|---|---|
| fused rotary embedding | `npu_fused_ops.py` | per-forward 400 ms → 155 ms |
| emit before the KV recompute | `fast_sched.py` | the 5th forward leaves the critical path |
| per-latent-frame decode | `fast_sched.py` | first frames ship while the rest still decode |

The rotary embedding was the surprise. Upstream expresses the rotation with complex
arithmetic and rebuilds the frequency table on every call; an operator-level
microbenchmark puts that at roughly 245 ms of the 400 ms forward. Swapping in
`torch_npu.npu_rotary_mul` (interleave mode) and caching cos/sin removes it.
Note the fused op needs `r1`/`r2` shaped `1S1D` under a BSND input.

A fourth, code-free win: four Ascend tuning environment variables give about 9% more.
Unlike the three above, this one was **only timed, not checked for bit-exactness**.

```
TASK_QUEUE_ENABLE=2 CPU_AFFINITY_CONF=2 ACLNN_CACHE_LIMIT=1000000 COMBINED_ENABLE=1
```

## Playing it

`serve.py` is a single process: the model loads once, HTTP and inference share it, and
decoded frames go straight to memory for the page to fetch. No mp4 files, no terminal
prompts.

```bash
export MG2_WEIGHTS=/weights/Matrix-Game-2.0
export TASK_QUEUE_ENABLE=2 CPU_AFFINITY_CONF=2 ACLNN_CACHE_LIMIT=1000000 COMBINED_ENABLE=1
python serve.py --mode templerun --port 8800 --host 0.0.0.0
```

Open `http://127.0.0.1:8800`. `--mode universal` and `--mode gta_drive` switch scene packs;
the three packs have incompatible key dimensions, so one process serves one pack.

| mode | movement | camera | bundled first frames |
|---|---|---|---|
| `universal` | W/A/S/D | I/J/K/L | 17 |
| `gta_drive` | W throttle, S brake | A/D steering | 6 |
| `templerun` | Q run, W jump, S slide, A/D strafe, Z/C turn | none | 6 |

Measured on the delivered path: 0.71 s from key press to the first frame in the browser,
0.94 s for all twelve, 1.09 s per cycle when pressing continuously. Twelve frames are
one second of 12 fps video, so a 1.09 s cycle delivers 11.0 fps — 92% of real time.
Model load takes 70 to 125 s; the model occupies 9.1 GB after load and peaks at 16.4 GB.

## Known limits

- Upstream has no multi-GPU inference path, and none was added.
- The byte-identical check covers the first session in a process.
- Cutting denoising from 4 steps to 3 was only compared qualitatively on one scene.
- VAE decode still costs 290 ms, now three tenths of each cycle and the largest single
  remainder. A lighter decoder is the obvious next step and was not attempted.
- Camera motion on heavily stylised first frames can collapse; long runs in the parkour
  scene can drift into the game's own score screen, which the model evidently learned
  along with the world.
