"""Time-budgeted pretraining for any backbone family (GPT-2 baseline, SWA-Lite, JurisNetLM, EncDec hybrid,
or LegalLM itself). Same recipe as legallm/train.py, which trained the submitted LegalLM:

  * DDP over all GPUs (launch with torchrun), fp16 autocast + GradScaler (T4 has no bf16), AdamW
    (betas 0.9/0.95, weight decay on matrices only), grad-clip 1.0, optional torch.compile
  * random 1,024-token windows from the uint16 memmap written by prepare_data.py
  * LR: linear warmup -> constant -> linear decay over the last `decay_fraction` of the TIME budget, so a
    run always ends annealed when the budget runs out; periodic resumable checkpoints (Kaggle sessions)

    torchrun --standalone --nproc_per_node=2 pretrain.py configs/gpt2_baseline.json
    python pretrain.py configs/swa_lite.json --train_hours 0.05      # quick single-process run

Config keys: model (dict with "arch"), train_bin, val_bin, [train_types_bin, val_types_bin] (JurisNetLM),
out_dir, tokenizer, seed, micro_batch, grad_accum, peak_lr, min_lr_ratio, warmup_steps, decay_fraction,
weight_decay, compile, train_hours, eval_every_sec, ckpt_every_sec, log_every, eval_batches.
"""
import argparse
import json
import os
import shutil
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backbones import build_backbone, count_parameters  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("--train_hours", type=float, default=None, help="override the config's time budget")
ap.add_argument("--out_dir", default=None)
args = ap.parse_args()
cfg = json.load(open(args.config))
if args.train_hours is not None:
    cfg["train_hours"] = args.train_hours
if args.out_dir:
    cfg["out_dir"] = args.out_dir
CKPT = os.path.join(cfg["out_dir"], "checkpoint", "ckpt.pt")
FINAL = os.path.join(cfg["out_dir"], "final")

ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
use_cuda = torch.cuda.is_available()
if ddp:
    dist.init_process_group("nccl" if use_cuda else "gloo")
    rank, local, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
else:
    rank, local, world = 0, 0, 1
device = torch.device("cuda", local) if use_cuda else torch.device("cpu")
if use_cuda:
    torch.cuda.set_device(local)
master = rank == 0
log = (lambda *a: print(*a, flush=True)) if master else (lambda *a: None)
if master:
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)

T, B, ACC = cfg["model"].get("max_seq_len", 1024), cfg["micro_batch"], cfg["grad_accum"]
amp = torch.autocast("cuda", dtype=torch.float16) if use_cuda else nullcontext()
mm = lambda p, dt: np.memmap(p, dtype=dt, mode="r")                                    # noqa: E731
train_data, val_data = mm(cfg["train_bin"], np.uint16), mm(cfg["val_bin"], np.uint16)
with_types = bool(cfg.get("train_types_bin"))
if with_types:
    train_types, val_types = mm(cfg["train_types_bin"], np.uint8), mm(cfg["val_types_bin"], np.uint8)
    assert len(train_types) == len(train_data) and len(val_types) == len(val_data), "types not aligned with tokens"
log(f"token pool {len(train_data):,} train / {len(val_data):,} val | GPUs {world} | arch {cfg['model'].get('arch', 'legallm')}")

torch.manual_seed(cfg["seed"])
model = build_backbone(cfg["model"]).to(device)
n_params = count_parameters(model)
log(f"parameters: {n_params / 1e6:.2f}M")
assert 50e6 <= n_params <= 300e6 or cfg.get("allow_any_size"), "capstone rule: 50M-300M parameters"

decay = [p for p in {id(p): p for p in model.parameters()}.values() if p.dim() >= 2]
no_decay = [p for p in {id(p): p for p in model.parameters()}.values() if p.dim() < 2]
opt_kw = dict(lr=cfg["peak_lr"], betas=(0.9, 0.95), eps=1e-8)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                         {"params": no_decay, "weight_decay": 0.0}], **opt_kw, **({"fused": True} if use_cuda else {}))
scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
state = {"step": 0, "train_seconds": 0.0, "tokens": 0, "log": []}
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
    state = ck["state"]
    log(f"resumed at step {state['step']} ({state['tokens']:,} tokens, {state['train_seconds'] / 3600:.2f} h)")
rng = np.random.default_rng([cfg["seed"], rank, state["step"]])


def batch_at(starts, data, types):
    buf = torch.from_numpy(np.stack([data[i:i + T + 1] for i in starts]).astype(np.int64))
    extra = {}
    if with_types:
        tb = torch.from_numpy(np.stack([types[i:i + T + 1] for i in starts]).astype(np.int64))
        extra = {"type_ids": tb[:, :-1].to(device), "type_targets": tb[:, 1:].to(device)}
    buf = buf.to(device, non_blocking=True)
    return buf[:, :-1], buf[:, 1:], extra


def get_batch(r):
    return batch_at(r.integers(0, len(train_data) - T - 1, size=B), train_data, train_types if with_types else None)


run_model = model
if cfg.get("compile") and use_cuda:
    try:
        run_model = torch.compile(model)
        x, y, ex = get_batch(np.random.default_rng(0))
        with amp:
            run_model(x, targets=y, **ex).backward()
        log("torch.compile OK")
    except Exception as e:                                    # fall back to eager mode
        run_model = model
        log("torch.compile failed, eager mode:", repr(e)[:200])
    model.zero_grad(set_to_none=True)
dmodel = DDP(run_model, device_ids=[local] if use_cuda else None) if ddp else run_model


@torch.no_grad()
def evaluate():
    n = B * world * cfg.get("eval_batches", 4)
    starts = np.linspace(0, len(val_data) - T - 2, n).astype(np.int64)[rank::world]
    tot = torch.zeros(2, device=device, dtype=torch.float64)
    run_model.eval()
    for s in range(0, len(starts), B):
        x, y, ex = batch_at(starts[s:s + B], val_data, val_types if with_types else None)
        if cfg["model"].get("arch") == "encdec":
            ex = {"split": T // 2}                             # fixed cut -> comparable validation loss
        with amp:
            tot[0] += run_model(x, targets=y, **ex).double() * len(x)
        tot[1] += len(x)
    run_model.train()
    if ddp:
        dist.all_reduce(tot)
    return (tot[0] / tot[1]).item()


def save_ckpt():
    if master:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "state": state, "config": cfg["model"]}, CKPT + ".tmp")
        os.replace(CKPT + ".tmp", CKPT)
    if ddp:
        dist.barrier()


def lr_at(step, progress):
    warm = min(1.0, (step + 1) / cfg["warmup_steps"])
    start = 1.0 - cfg["decay_fraction"]
    frac = 0.0 if progress <= start else min(1.0, (progress - start) / cfg["decay_fraction"])
    return cfg["peak_lr"] * warm * (1.0 - (1.0 - cfg["min_lr_ratio"]) * frac)


budget = cfg["train_hours"] * 3600
loop_start, base = time.time(), state["train_seconds"]
tok_per_step = B * T * ACC * world
log(f"budget {cfg['train_hours']:.2f} h | {tok_per_step:,} tokens/step")
last_eval = last_ckpt = time.time()
dmodel.train()
while True:
    now = time.time()
    ctrl = torch.tensor([now - loop_start, float(now - last_eval > cfg["eval_every_sec"]),
                         float(now - last_ckpt > cfg["ckpt_every_sec"])], device=device, dtype=torch.float64)
    if ddp:
        dist.broadcast(ctrl, 0)                                # rank 0 decides -> ranks stay in lock-step
    trained = base + ctrl[0].item()
    if trained >= budget:
        break
    lr = lr_at(state["step"], trained / budget)
    for g in opt.param_groups:
        g["lr"] = lr
    loss_sum = torch.zeros((), device=device)
    for micro in range(ACC):
        x, y, ex = get_batch(rng)
        with (dmodel.no_sync() if ddp and micro < ACC - 1 else nullcontext()):
            with amp:
                loss = dmodel(x, targets=y, **ex)
            scaler.scale(loss / ACC).backward()
        loss_sum += loss.detach().float()
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    state["step"] += 1; state["tokens"] += tok_per_step; state["train_seconds"] = trained
    if state["step"] % cfg["log_every"] == 0:
        ml = loss_sum / ACC
        if ddp:
            dist.all_reduce(ml); ml /= world
        state["log"].append({"step": state["step"], "tokens": state["tokens"], "train_loss": ml.item(), "lr": lr,
                             "grad_norm": float(gn), "hours": trained / 3600})
        log(f"step {state['step']:>6} | {state['tokens'] / 1e6:8.1f}M tok | loss {ml.item():.4f} | lr {lr:.2e} | {trained / 3600:.2f} h")
    if ctrl[1].item() > 0:
        vl = evaluate()
        state["log"].append({"step": state["step"], "tokens": state["tokens"], "val_loss": vl})
        log(f"    validation loss {vl:.4f} (ppl {np.exp(vl):.2f})")
        last_eval = time.time()
    if ctrl[2].item() > 0:
        save_ckpt(); last_ckpt = time.time()

vl = evaluate()
state["log"].append({"step": state["step"], "tokens": state["tokens"], "val_loss": vl})
log(f"finished: step {state['step']} | {state['tokens']:,} tokens | validation loss {vl:.4f}")
save_ckpt()
if master:
    os.makedirs(FINAL, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(FINAL, "pytorch_model.bin"))
    json.dump(cfg["model"], open(os.path.join(FINAL, "config.json"), "w"), indent=2)
    json.dump({"step": state["step"], "tokens": state["tokens"], "train_hours": state["train_seconds"] / 3600,
               "final_val_loss": vl, "parameters": n_params}, open(os.path.join(FINAL, "training_state.json"), "w"), indent=2)
    json.dump(state["log"], open(os.path.join(cfg["out_dir"], "train_log.json"), "w"))
    if cfg.get("tokenizer") and os.path.exists(cfg["tokenizer"]):
        shutil.copy(cfg["tokenizer"], os.path.join(FINAL, "tokenizer.json"))
if ddp:
    dist.barrier()
    dist.destroy_process_group()
