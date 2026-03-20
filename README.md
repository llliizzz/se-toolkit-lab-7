# Lab 7 — Build a Client with an AI Coding Agent

[Sync your fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork#syncing-a-fork-branch-from-the-command-line) regularly — the lab gets updated.

## Product brief

> Build a Telegram bot that lets users interact with the LMS backend through chat. Users should be able to check system health, browse labs and scores, and ask questions in plain language. The bot should use an LLM to understand what the user wants and fetch the right data. Deploy it alongside the existing backend on the VM.

This is what a customer might tell you. Your job is to turn it into a working product using an AI coding agent (Qwen Code) as your development partner.

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  ┌──────────────┐     ┌──────────────────────────────────┐   │
│  │  Telegram    │────▶│  Your Bot                        │   │
│  │  User        │◀────│  (aiogram / python-telegram-bot) │   │
│  └──────────────┘     └──────┬───────────────────────────┘   │
│                              │                               │
│                              │ slash commands + plain text    │
│                              ├───────▶ /start, /help         │
│                              ├───────▶ /health, /labs        │
│                              ├───────▶ intent router ──▶ LLM │
│                              │                    │          │
│                              │                    ▼          │
│  ┌──────────────┐     ┌──────┴───────┐    tools/actions      │
│  │  Docker      │     │  LMS Backend │◀───── GET /items      │
│  │  Compose     │     │  (FastAPI)   │◀───── GET /analytics  │
│  │              │     │  + PostgreSQL│◀───── POST /sync      │
│  └──────────────┘     └──────────────┘                       │
└──────────────────────────────────────────────────────────────┘
```

## Requirements

### P0 — Must have

1. Testable handler architecture — handlers work without Telegram
2. CLI test mode: `cd bot && uv run bot.py --test "/command"` prints response to stdout
3. `/start` — welcome message
4. `/help` — lists all available commands
5. `/health` — calls backend, reports up/down status
6. `/labs` — lists available labs
7. `/scores <lab>` — per-task pass rates
8. Error handling — backend down produces a friendly message, not a crash

### P1 — Should have

1. Natural language intent routing — plain text interpreted by LLM
2. All 9 backend endpoints wrapped as LLM tools
3. Inline keyboard buttons for common actions
4. Multi-step reasoning (LLM chains multiple API calls)

### P2 — Nice to have

1. Rich formatting (tables, charts as images)
2. Response caching
3. Conversation context (multi-turn)

### P3 — Deployment

1. Bot containerized with Dockerfile
2. Added as service in `docker-compose.yml`
3. Deployed and running on VM
4. README documents deployment

## Learning advice

Notice the progression above: **product brief** (vague customer ask) → **prioritized requirements** (structured) → **task specifications** (precise deliverables + acceptance criteria). This is how engineering work flows.

You are not following step-by-step instructions — you are building a product with an AI coding agent. The learning comes from planning, building, testing, and debugging iteratively.

## Learning outcomes

By the end of this lab, you should be able to say:

1. I turned a vague product brief into a working Telegram bot.
2. I can ask it questions in plain language and it fetches the right data.
3. I used an AI coding agent to plan and build the whole thing.

## Tasks

### Prerequisites

1. Complete the [lab setup](./lab/setup/setup-simple.md#lab-setup)

> **Note**: First time in this course? Do the [full setup](./lab/setup/setup-full.md#lab-setup) instead.

### Required

1. [Plan and Scaffold](./lab/tasks/required/task-1.md) — P0: project structure + `--test` mode
2. [Backend Integration](./lab/tasks/required/task-2.md) — P0: slash commands + real data
3. [Intent-Based Natural Language Routing](./lab/tasks/required/task-3.md) — P1: LLM tool use
4. [Containerize and Document](./lab/tasks/required/task-4.md) — P3: containerize + deploy

## Telegram bot

The repository now includes a `bot/` application that talks to the LMS backend.
It supports both slash commands and natural-language questions:

- `/start`, `/help`, `/health`, `/labs`, `/scores <lab>`
- plain-language requests such as `what labs are available?`
- LLM tool calling across all nine backend endpoints

The bot architecture keeps handlers independent from Telegram, so the same logic
works in CLI test mode and in real polling mode.

## Bot configuration

Create `.env.bot.secret` in the repository root from `.env.bot.example`.

Required variables:

- `BOT_TOKEN` for Telegram polling mode
- `LMS_API_URL` and `LMS_API_KEY` for the LMS backend
- `LLM_API_KEY`, `LLM_API_BASE_URL`, `LLM_API_MODEL` for natural-language routing

`BOT_TOKEN` is optional in `--test` mode, but the backend and LLM variables are
still needed for live integrations.

## Run locally

```bash
cd bot
uv sync
uv run bot.py --test "/start"
uv run bot.py --test "/health"
uv run bot.py --test "what labs are available?"
```

To run the real Telegram bot:

```bash
cd bot
uv run bot.py
```

## Deploy

Prepare the root `.env.docker.secret` with the bot variables as well:

- `BOT_TOKEN`
- `BOT_LMS_API_URL=http://backend:8000`
- `LLM_API_KEY`
- `LLM_API_BASE_URL=http://host.docker.internal:42005/v1`
- `LLM_API_MODEL`

Then build and start the full stack:

```bash
docker compose --env-file .env.docker.secret up --build -d
docker compose --env-file .env.docker.secret ps
docker compose --env-file .env.docker.secret logs bot --tail 50
```

Verify:

- `curl -sf http://localhost:42002/docs`
- `docker compose --env-file .env.docker.secret ps bot`
- send `/start`, `/health`, and a plain-language question in Telegram

## Troubleshooting

- `LLM error: HTTP 401` usually means the Qwen proxy token expired. Restart it with `cd ~/qwen-code-oai-proxy && docker compose restart`.
- If the bot container cannot reach the backend, make sure `BOT_LMS_API_URL` points to `http://backend:8000`, not `localhost`.
- If natural-language queries fail only inside Docker, make sure `LLM_API_BASE_URL` uses `http://host.docker.internal:42005/v1`.
- If Telegram polling times out, first verify the VM can reach `https://api.telegram.org`, then re-check `BOT_TOKEN` in `.env.bot.secret` and `.env.docker.secret`.
