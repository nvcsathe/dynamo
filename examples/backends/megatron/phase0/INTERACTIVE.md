# Megatron + Dynamo — Phase 0 Interactive Bring-up

For iterating on the integration. Same end-state as `launch_phase0.slurm`, but
you drive each process by hand in tmux panes so you can restart pieces
individually without resubmitting.

Use this when you're changing handler code, the streaming protocol, or the
launcher itself. Use the sbatch (`launch_phase0.slurm`) once the integration
is stable and you just want a hands-off bring-up.

## 0. Allocate

```bash
salloc \
  --job-name=megatron-dynamo-interactive \
  --account=<acct> --partition=<partition> \
  --nodes=1 --gpus-per-node=2 \
  --time=02:00:00
```

After `salloc` returns, you're still on the login node holding the allocation.
Get onto the compute node and start tmux:

```bash
srun --jobid=$SLURM_JOB_ID --overlap --pty bash
tmux                       # then Ctrl-b " or Ctrl-b % to split
```

You want 4 panes: **A** coordinator, **B** worker, **C** frontend, **D** curl.

If `tmux` isn't on the compute image, open multiple
`srun --jobid=$SLURM_JOB_ID --overlap --pty bash` shells from separate
login-node terminals instead.

## 1. NATS + etcd (one-time per session)

Easiest for interactive work: run both on the compute node itself.

```bash
nats-server --jetstream --port 4222 > /tmp/nats.log 2>&1 &
etcd --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://0.0.0.0:2379 > /tmp/etcd.log 2>&1 &

export NATS_SERVER=nats://127.0.0.1:4222
export ETCD_ENDPOINTS=http://127.0.0.1:2379
```

If you already have NATS/etcd on a head node, just `export` those URLs and
skip the launches. JetStream must be enabled on NATS — Dynamo's event plane
relies on it.

## 2. Pane A — Megatron coordinator (TP=2)

```bash
cd $MEGATRON_WORKTREE
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29500

uv run python -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 --node-rank=0 \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT \
  tools/run_dynamic_text_generation_server.py \
    --frontend dynamo \
    --inference-coordinator-port 5555 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --load /shared/models/Llama-3.1-8B-mcore \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model meta-llama/Llama-3.1-8B \
    --bf16 --use-flash-attn \
    2>&1 | tee /tmp/coordinator.log
```

Wait for a line like `MEGATRON_COORDINATOR_ADDR=tcp://<host>:5555`. Grab it
from any other pane:

```bash
grep -m1 -oP 'MEGATRON_COORDINATOR_ADDR=\K.*' /tmp/coordinator.log
```

## 3. Pane B — Dynamo Megatron worker

```bash
cd $DYNAMO_WORKTREE
export COORD_ADDR=$(grep -m1 -oP 'MEGATRON_COORDINATOR_ADDR=\K.*' /tmp/coordinator.log)
export NATS_SERVER ETCD_ENDPOINTS

python -m dynamo.megatron \
  --coordinator-addr "$COORD_ADDR" \
  --model meta-llama/Llama-3.1-8B \
  --served-model-name llama-3.1-8b \
  --context-length 4096 \
  2>&1 | tee /tmp/worker.log
```

Wait for the `register_model` success line. The worker stays alive.

## 4. Pane C — Dynamo frontend

```bash
cd $DYNAMO_WORKTREE
export NATS_SERVER ETCD_ENDPOINTS

python -m dynamo.frontend --http-port 8080 \
  2>&1 | tee /tmp/frontend.log
```

The frontend discovers the worker via etcd. Look for a log line confirming
`llama-3.1-8b` appears in its model list.

## 5. Pane D — smoke test

```bash
# Liveness:
curl -s http://127.0.0.1:8080/v1/models | jq .

# Streaming completion:
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
        "model": "llama-3.1-8b",
        "messages": [{"role":"user","content":"Hello in one short sentence."}],
        "stream": true,
        "max_tokens": 64
      }'
```

You should see SSE chunks — one Megatron `ENGINE_REPLY_PARTIAL` per step,
surfaced through the worker as a `{"token_ids":[...]}` chunk, detokenized by
the frontend into `delta.content` deltas.

## Iteration cost

Only Pane A (the model) is expensive to restart.

| Change | Restart | Cost |
|---|---|---|
| Worker handler / engine_client | Pane B | ~5s |
| Frontend args / router config | Pane C | ~3s |
| Megatron streaming protocol (headers, inference_client, coordinator, engine) | Pane A | ~30-60s for 8B; ~5s for 1B |

When iterating on the Megatron-side protocol, swap to **Llama-3.2-1B**, TP=1
(`--gpus-per-node=1 --tensor-model-parallel-size 1`) to keep the cycle
under 10 seconds. The integration code is identical at 1B and 8B.

## Cleanup

When done: `exit` all tmux panes → `exit` the `srun --pty` → `scancel
$SLURM_JOB_ID` (or just let the time limit expire). Native `nats-server` and
`etcd` processes die with the allocation.

## Common interactive-only gotchas

- **Pane A holds both GPUs.** Pane B has no GPU pin, which is correct (the
  Dynamo worker is CPU-only ZMQ + NATS). Don't pass `--gpus=0` to anything
  in Panes B/C/D — just inherit the allocation.
- **Port collisions on rerun.** If you Ctrl-C Pane A and immediately relaunch,
  the coordinator's `:5555` may be in TIME_WAIT for ~30s. Pass
  `--inference-coordinator-port 0` to get a random port (then re-export
  `COORD_ADDR`), or wait it out.
- **Stale model in etcd after a hard kill.** If Pane B dies without graceful
  shutdown, its entry can linger in etcd until the etcd lease expires. The
  frontend may continue advertising a model that no longer answers. If a
  curl hangs after a restart, check `etcdctl get --prefix /dynamo` (or wait
  ~30s for the lease).
- **Tokenizer mismatch is silent.** If you pass the wrong `--tokenizer-model`
  to Megatron or `--model` to the worker, generation will be garbage but
  nothing will error. Always pass the same HF id to both.
