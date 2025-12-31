# Repository Guidelines

## Project Structure & Module Organization
- `api/`: FastAPI app (`api/main.py`) and routers (`api/routers/*`).
- `core/`: shared logic (config/cache in `core/config.py`, T2I in `core/t2i/`, training helpers in `core/train/`).
- `workers/`: Celery app (`workers/celery_app.py`) and async tasks (`workers/tasks/*`).
- `configs/`: YAML configuration (`configs/app.yaml`, `configs/models.yaml`, `configs/train/*`).
- `frontend/`: React UI (`frontend/react_app/`).
- `scripts/`: setup/start/smoke scripts; `tests/`: pytest suite; `docker/`: compose + Dockerfiles; `docs/`: design notes.

## Build, Test, and Development Commands
- `conda env create -f environment.yml && conda activate ai_env`: create/enter the project env.
- `python scripts/check_ai_env.py`: verify runtime imports (includes `peft`, `redis`, `celery`).
- `cp .env.example .env` then `python scripts/setup.py`: create AI_WAREHOUSE directories + baseline `.env`.
- `bash scripts/start_api.sh` (or `uvicorn api.main:app --reload`): run API at `http://localhost:8000` (`/docs` for Swagger).
- `bash scripts/start_worker.sh`: run Celery worker (requires Redis).
- `python scripts/scan_models.py --replace`: scan `/mnt/c/ai_models` → `registry.json`.
- React UI: `npm --prefix frontend/react_app install`, then `npm --prefix frontend/react_app run dev|lint|test`.
- Tests/lint: `pytest -q`, `ruff check api core workers tests`.

## Coding Style & Naming Conventions
- Python: 4-space indentation, `snake_case` for functions/modules, `PascalCase` for classes; prefer type hints where practical.
- Formatting/linting: `ruff check api core workers tests` (config in `pyproject.toml`).
- Avoid hard-coded paths; rely on `PROJECT_SLUG` + `AI_MODELS_ROOT`, `AI_CACHE_ROOT`, `AI_DATASETS_ROOT`, `AI_TRAINING_ROOT` via `core.config.get_cache_paths()`.

## Testing Guidelines
- Python tests use `pytest` (+ `pytest-asyncio` for async). Test files follow `tests/test_*.py`.
- Mark GPU/model/network-dependent tests as `@pytest.mark.slow`; run fast tests with `pytest -m "not slow"`.
- React tests use `vitest` via `npm --prefix frontend/react_app run test`.
- No coverage threshold is enforced; add focused regression tests for fixes in `api/`, `core/`, or `workers/`.

## Commit & Pull Request Guidelines
- Follow Conventional Commits used in history: `feat(api): ...`, `fix(train): ...`, `refactor(core): ...`, `test(utils): ...`, `chore: ...`.
- PRs: include a short description, how you tested (commands/output), and screenshots for UI changes.
- Keep the repo slim: do not commit weights, datasets, or large outputs—store them under `AI_CACHE_ROOT` instead.
