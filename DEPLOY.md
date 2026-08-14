# Deploy

Live: **https://goarag.72-61-251-132.sslip.io**

Single VPS, two systemd units behind the existing nginx. No Docker required —
the Dockerfiles exist for `docker compose up` reproducibility, but the live
deployment runs the services directly.

## Why it is shaped this way

The target box is **1 vCPU / 4 GB, already running other services**. Two
consequences drove the design:

- **Ship a prebuilt index.** Building it needs ~11.5k embeddings — 855 s on a
  4-core laptop, far worse on a contended single core. `scripts/build_snapshot.py`
  writes `data/snapshot/` (19.7 MB); the API loads it in **3.4 s**. Boot went
  from *minutes of inference* to *seconds of I/O*.
- **HTTPS is not optional.** `getUserMedia` — the microphone — is refused by
  browsers on plain HTTP. Without TLS the entire voice feature is dead on
  arrival, which is why this uses a real cert rather than `http://IP:port`.
  `sslip.io` resolves `<anything>.72-61-251-132.sslip.io` to the box, so
  Let's Encrypt issues against it with no domain purchase.

`OMP_NUM_THREADS=1` is set deliberately: on one shared core, extra ONNX threads
buy contention, not throughput.

## Ports

`8100` API (localhost only) · `3003` web (localhost only) · nginx terminates TLS
and routes `/` → web, `/api/` → API. 3000/3002 were already taken on this host.

## Redeploy

```bash
# API
tar -czf /tmp/api.tgz --exclude='__pycache__' -C apps/api goarag main.py
scp /tmp/api.tgz root@HOST:/tmp/
ssh root@HOST 'cd /opt/goarag/apps/api && tar -xzf /tmp/api.tgz && systemctl restart goarag-api'

# Frontend (built locally; the VPS never runs npm install or next build)
cd apps/web
NEXT_PUBLIC_GOARAG_API="https://goarag.72-61-251-132.sslip.io/api" npx next build
cp -r public .next/standalone/ && cp -r .next/static .next/standalone/.next/
tar -czf /tmp/web.tgz -C .next/standalone .
scp /tmp/web.tgz root@HOST:/tmp/
ssh root@HOST 'cd /opt/goarag/web && tar -xzf /tmp/web.tgz && systemctl restart goarag-web'
```

`NEXT_PUBLIC_*` is inlined at build time, so the API URL must be set for the
build — it cannot be supplied at runtime.

## Secrets

`/opt/goarag/.env` (mode 600, loaded by the unit). Add `SARVAM_API_KEY=` to
enable voice; everything else already runs without it.
