# systemd templates

These unit files are **examples** for running CharaForge T2I Lab on a Linux host.

- Adjust paths (`WorkingDirectory`, `EnvironmentFile`) to your installation.
- This repo expects a configured `.env` (see `.env.example`) including `AI_CACHE_ROOT`.
- For GPU hosts, make sure your CUDA drivers/container runtime are configured.

## Units

- `charaforge-api.service`: FastAPI server (`uvicorn api.main:app ...`)
- `charaforge-t2i-worker.service`: Redis-backed async T2I worker (`python -m api.t2i_worker`)

## Install

```bash
sudo cp docs/systemd/charaforge-*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now charaforge-api.service charaforge-t2i-worker.service
```

