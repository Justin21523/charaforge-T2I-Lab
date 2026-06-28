# Deployment Notes

## Public Portfolio Demo

Recommended target: GitHub Pages.

The public demo is static and served from `portfolio-web/`. It does not run GPU inference,
Redis, Celery, or the FastAPI backend. This is intentional: the goal is a reliable interview
demo with clear product flows and screenshots.

Deployment is handled by `.github/workflows/pages.yml`.

## Local Full Stack

Use Docker Compose when you want backend services and workers:

```bash
docker compose -f docker/docker-compose.yml up --build
```

Optional profile:

```bash
docker compose -f docker/docker-compose.yml --profile frontend up --build
```

## Live Backend Demo Options

If an interactive public backend is required later, use one of these:

- GPU workstation behind a reverse proxy for real inference.
- Render/Railway/Fly.io for the API only, with mock or CPU-safe endpoints.
- Vercel/Netlify for frontend plus a separate backend host.

GitHub Pages alone cannot host the FastAPI API.
