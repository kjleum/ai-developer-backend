# AI Platform — Monorepo (Frontend + Backend)

Это рабочий MVP-бэкенд (FastAPI + Postgres) и статический фронтенд (Telegram Mini App / Web SPA на чистом JS).

## Локальный запуск (Docker)

```bash
cp backend/.env.example backend/.env
# заполни JWT_SECRET и ENCRYPTION_KEY (Fernet)

docker compose up --build
```

- Frontend: http://localhost/
- API docs: http://localhost/api/v1/docs
- Health: http://localhost/api/v1/health

## Основные эндпоинты

- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login` (OAuth2 form)
- `POST /api/v1/auth/login_json`
- `POST /api/v1/auth/telegram` (Telegram Mini App initData)
- `GET /api/v1/users/me`
- `PUT /api/v1/users/me/settings`
- `POST /api/v1/keys` / `GET /api/v1/keys` / `DELETE /api/v1/keys/{provider}`
- `POST /api/v1/ai/generate`
- `WS /api/v1/ai/stream`
- `GET /api/v1/projects`

## Деплой на Render (рекомендуемая схема)

### 1) Postgres
Создай **Render Postgres** и возьми `Internal Database URL` (или стандартный `DATABASE_URL`).

### 2) Backend (Web Service)
- Type: **Web Service**
- Runtime: **Docker**
- Root Directory: `backend`
- Environment variables:
  - `DATABASE_URL` = (из Postgres)
  - `JWT_SECRET` = длинная строка
  - `ENCRYPTION_KEY` = Fernet key
  - `CORS_ORIGINS` = домен фронта (например `https://your-frontend.onrender.com`)
  - (опционально) `OPENAI_API_KEY`, `GROQ_API_KEY`, `TELEGRAM_BOT_TOKEN`

Healthcheck: `/api/v1/health`

### 3) Frontend (Static Site)
- Type: **Static Site**
- Root Directory: `frontend`
- Build Command: *(пусто)*
- Publish Directory: `.`

В `app.js` по умолчанию `apiBase = "/api/v1"`. Если фронт и бэк на разных доменах, укажи в UI **Настройки → API Base** полный URL бэка, например:

`https://your-backend.onrender.com/api/v1`

### 4) Telegram Mini App
В BotFather:
- `/setdomain` → домен фронта (HTTPS)
- `/setmenubutton` → Web App URL (фронт)

И в `backend` переменная:
- `TELEGRAM_BOT_TOKEN`

## Миграции

Для продакшена лучше прогнать alembic:

```bash
cd backend
alembic upgrade head
```

(Но для быстрого старта API сам делает `create_all()` на startup.)
