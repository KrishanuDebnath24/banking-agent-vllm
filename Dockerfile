# One image for the whole app. It runs as two containers:
#   web chat (default)   python -m api                ports 8000
#   bank servers         python -m mcp_servers.run    ports 8101-8105
# On your PC:  docker compose up --build   (docker-compose.yml). On AWS: ECS Fargate (phase 4, step 4).
#
# No secrets inside: .env stays out (.dockerignore); settings and keys arrive as environment
# variables when the container starts. demo_accounts.json and seed_log.jsonl are copied in if
# they exist (account ids and seeding dates, not secrets).

FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so a code change doesn't reinstall them.
COPY requirements.txt .
RUN pip install -r requirements.txt

# The app. Owned by root, so the user that runs it can read it but not change it.
COPY . .
RUN python -m compileall -q /app \
 && useradd --system --uid 10001 --home-dir /app --shell /usr/sbin/nologin bank \
 && mkdir -p /app/data && chown bank /app/data

# Containers listen on all interfaces; the SQLite paths are only used if DATABASE_URL isn't set.
ENV API_HOST=0.0.0.0 PORT=8000 PYTHONDONTWRITEBYTECODE=1 \
    SESSION_DB=/app/data/sessions.db OTP_DB=/app/data/verification.db \
    TELEMETRY_DB=/app/data/telemetry.db SERVICE_REQUESTS_DB=/app/data/service_requests.db \
    GUEST_DB=/app/data/guest_banks.db WEB_DB=/app/data/web.db

USER bank
EXPOSE 8000
CMD ["python", "-m", "api"]
