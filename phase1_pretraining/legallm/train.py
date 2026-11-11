"""Time-budgeted training for LegalLM. Launched by the notebook with torchrun (one process per GPU).

* DistributedDataParallel across all GPUs (no DataParallel output gathering)
* fp16 autocast + GradScaler (T4 has no bf16), fused AdamW, optional torch.compile
* Random windows from a uint16 token memmap (documents separated by EOS)
* LR: warmup -> constant -> linear decay over the last DECAY_FRACTION of the training time,
  so the run always ends annealed when the time budget runs out
* Periodic checkpoints (resumable across Kaggle sessions)
"""
import json
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LegalLM, ModelConfig, count_parameters  # noqa: E402

cfg = json.load(open(sys.argv[1]))
ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
use_cuda = torch.cuda.is_available()
if ddp:
    dist.init_process_group("nccl" if use_cuda else "gloo")
    rank, local, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
else:
    rank, local, world = 0, 0, 1
if use_cuda:
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
else:
    device = torch.device("cpu")
master = rank == 0


def log(*a):
    if master:
        print(*a, flush=True)


T = cfg["model"]["max_seq_len"]
B = cfg["micro_batch"]
ACC = cfg["grad_accum"]
amp = torch.autocast("cuda", dtype=torch.float16) if use_cuda else nullcontext()

train_data = np.memmap(cfg["train_bin"], dtype=np.uint16, mode="r")
val_data = np.memmap(cfg["val_bin"], dtype=np.uint16, mode="r")
log(f"Token pool: {len(train_data):,} train / {len(val_data):,} val | GPUs: {world}")

torch.manual_seed(cfg["seed"])
model = LegalLM(ModelConfig(**cfg["model"])).to(device)
log(f"Parameters: {count_parameters(model) / 1e6:.2f}M")

decay, no_decay = [], []
for _, p in model.named_parameters():
    (decay if p.dim() >= 2 else no_decay).append(p)
groups = [{"params": decay, "weight_decay": cfg["weight_decay"]},
          {"params": no_decay, "weight_decay": 0.0}]
opt_kwargs = dict(lr=cfg["peak_lr"], betas=(0.9, 0.95), eps=1e-8)
opt = torch.optim.AdamW(groups, fused=True, **opt_kwargs) if use_cuda else torch.optim.AdamW(groups, **opt_kwargs)
scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

state = {"step": 0, "train_seconds": 0.0, "tokens": 0, "log": []}
if not os.path.exists(cfg["ckpt_path"]) and cfg.get("init_from"):
    # Warm start: weights (and optimizer moments, if present) from a previous run, fresh LR schedule.
    ck = torch.load(cfg["init_from"], map_location=device, weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    if "opt" in ck:
        try:
            opt.load_state_dict(ck["opt"])
        except Exception as e:
            log("Optimizer state not reused:", repr(e)[:200])
    prev_tokens = ck.get("state", {}).get("tokens", 0) if isinstance(ck.get("state"), dict) else 0
    del ck
    log(f"Warm start from {cfg['init_from']} (model previously trained on {prev_tokens:,} tokens)")
elif os.path.exists(cfg["ckpt_path"]):
    ck = torch.load(cfg["ckpt_path"], map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    scaler.load_state_dict(ck["scaler"])
    state = ck["state"]
    del ck
    log(f"Resumed at step {state['step']} ({state['tokens']:,} tokens, {state['train_seconds'] / 3600:.2f} h trained)")

rng = np.random.default_rng([cfg["seed"], rank, state["step"]])


def get_batch(r, data=train_data, bsz=B):
    ix = r.integers(0, len(data) - T - 1, size=bsz)
    buf = torch.from_numpy(np.stack([data[i:i + T + 1] for i in ix]).astype(np.int64))
    if use_cuda:
        buf = buf.pin_memory().to(device, non_blocking=True)
    return buf[:, :-1], buf[:, 1:]


# ---- optional torch.compile, tested on one micro-batch before DDP wraps the model
run_model = model
if cfg.get("compile") and use_cuda:
    try:
        t0 = time.time()
        compiled = torch.compile(model)
        x, y = get_batch(np.random.default_rng(0))
        with amp:
            loss = compiled(x, targets=y)
        loss.backward()
        run_model = compiled
        log(f"torch.compile OK ({time.time() - t0:.0f}s)")
    except Exception as e:  # fall back to eager mode
        log("torch.compile failed, using eager mode:", repr(e)[:300])
    model.zero_grad(set_to_none=True)
dmodel = DDP(run_model, device_ids=[local] if use_cuda else None) if ddp else run_model


@torch.no_grad()
def evaluate():
    n_windows = B * world * cfg.get("eval_batches", 4)
    starts = np.linspace(0, len(val_data) - T - 2, n_windows).astype(np.int64)[rank::world]
    total = torch.zeros(2, device=device, dtype=torch.float64)
    run_model.eval()
    for s in range(0, len(starts), B):
        chunk = starts[s:s + B]
        buf = torch.from_numpy(np.stack([val_data[i:i + T + 1] for i in chunk]).astype(np.int64)).to(device)
        with amp:
            loss = run_model(buf[:, :-1], targets=buf[:, 1:])
        total[0] += loss.double() * len(chunk)
        total[1] += len(chunk)
    run_model.train()
    if ddp:
        dist.all_reduce(total)
    return (total[0] / total[1]).item()


def save_ckpt():
    if master:
        tmp = cfg["ckpt_path"] + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "state": state, "config": cfg["model"]}, tmp)
        os.replace(tmp, cfg["ckpt_path"])
        with open(cfg["log_path"], "w") as f:
            json.dump(state["log"], f)
    if ddp:
        dist.barrier()


def lr_at(step, progress):
    warm = min(1.0, (step + 1) / cfg["warmup_steps"])
    start = 1.0 - cfg["decay_fraction"]
    if progress <= start:
        return cfg["peak_lr"] * warm
    frac = min(1.0, (progress - start) / cfg["decay_fraction"])
    return cfg["peak_lr"] * warm * (1.0 - (1.0 - cfg["min_lr_ratio"]) * frac)


loop_start = time.time()
session_seconds = max(0.0, cfg["deadline"] - loop_start)
base_seconds = state["train_seconds"]
total_seconds = cfg.get("total_train_seconds") or (base_seconds + session_seconds)
tokens_per_step = B * T * ACC * world
log(f"Training up to {session_seconds / 3600:.2f} h this session; LR schedule spans "
    f"{total_seconds / 3600:.2f} h in total; {tokens_per_step:,} tokens per step")

last_eval = last_ckpt = t_log = time.time()
tok_log = state["tokens"]
dmodel.train()
while True:
    # rank 0 decides (time, eval, checkpoint) so all processes stay in lock-step
    now = time.time()
    ctrl = torch.tensor([now - loop_start,
                         float(now - last_eval > cfg["eval_every_sec"]),
                         float(now - last_ckpt > cfg["ckpt_every_sec"])], device=device, dtype=torch.float64)
    if ddp:
        dist.broadcast(ctrl, 0)
    elapsed, do_eval, do_ckpt = ctrl[0].item(), ctrl[1].item() > 0, ctrl[2].item() > 0
    trained = base_seconds + elapsed
    if elapsed >= session_seconds or trained >= total_seconds:
        break

    lr = lr_at(state["step"], trained / total_seconds)
    for g in opt.param_groups:
        g["lr"] = lr
    loss_sum = torch.zeros((), device=device)
    for micro in range(ACC):
        x, y = get_batch(rng)
        sync_ctx = dmodel.no_sync() if (ddp and micro < ACC - 1) else nullcontext()
        with sync_ctx:
            with amp:
                loss = dmodel(x, targets=y)
            scaler.scale(loss / ACC).backward()
        loss_sum += loss.detach().float()
    scaler.unscale_(opt)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    state["step"] += 1
    state["tokens"] += tokens_per_step
    state["train_seconds"] = trained

    if state["step"] % cfg["log_every"] == 0:
        mean_loss = loss_sum / ACC
        if ddp:
            dist.all_reduce(mean_loss)
            mean_loss /= world
        t_now = time.time()
        tps = (state["tokens"] - tok_log) / max(t_now - t_log, 1e-9)
        t_log, tok_log = t_now, state["tokens"]
        entry = {"step": state["step"], "tokens": state["tokens"], "train_loss": mean_loss.item(),
                 "lr": lr, "tok_per_s": tps, "hours": trained / 3600, "grad_norm": float(grad_norm)}
        state["log"].append(entry)
        log(f"step {state['step']:>6} | tokens {state['tokens'] / 1e6:8.1f}M | loss {entry['train_loss']:.4f} | "
            f"lr {lr:.2e} | {tps:,.0f} tok/s | {trained / 3600:.2f} h")
    if do_eval:
        vl = evaluate()
        state["log"].append({"step": state["step"], "tokens": state["tokens"], "val_loss": vl, "hours": trained / 3600})
        log(f"    validation loss {vl:.4f} (ppl {np.exp(vl):.2f})")
        last_eval = time.time()
    if do_ckpt:
        save_ckpt()
        last_ckpt = time.time()

state["train_seconds"] = base_seconds + (time.time() - loop_start)
vl = evaluate()
state["log"].append({"step": state["step"], "tokens": state["tokens"], "val_loss": vl,
                     "hours": state["train_seconds"] / 3600})
log(f"Finished: step {state['step']}, {state['tokens']:,} tokens this run total, final validation loss {vl:.4f} "
    f"(ppl {np.exp(vl):.2f})")
save_ckpt()
if master:
    os.makedirs(cfg["final_dir"], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(cfg["final_dir"], "pytorch_model.bin"))
    with open(os.path.join(cfg["final_dir"], "config.json"), "w") as f:
        json.dump(cfg["model"], f, indent=2)
    with open(os.path.join(cfg["final_dir"], "training_state.json"), "w") as f:
        json.dump({"step": state["step"], "tokens": state["tokens"], "train_hours": state["train_seconds"] / 3600,
                   "final_val_loss": vl, "annealed": state["train_seconds"] >= total_seconds - 120}, f, indent=2)
if ddp:
    dist.barrier()
    dist.destroy_process_group()