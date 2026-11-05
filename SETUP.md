# Setup guide: Banking Agent on a self-hosted model (vLLM)

A copy of the multi-agent banking assistant that runs its model on our own GPUs: vLLM with prefix
caching and continuous batching, first on one GPU machine, then on Amazon EKS. The agents,
guardrails, approvals, LangSmith tracing and eval suite are the same as in the Claude version
(plus day-by-day and month-to-date figures and charts, including a combined chart, and 44 eval
cases); only the model connection and the deployment differ. Both versions can run side by side.

## What's different from the Claude version

| Part | Claude version | This copy |
| --- | --- | --- |
| Model | Claude through Bedrock or the Anthropic API | `MODEL=vllm:openai/gpt-oss-20b`, served by vLLM ([agents/vllm_chat.py](agents/vllm_chat.py)) |
| Each model call | plain request | adds `x-session-id` (the conversation) so the router keeps it on one GPU, and a per-customer `cache_salt` so cached prompts are never shared between customers |
| Prompts | date on the first line | date at the end, so the start of every prompt never changes and stays cached |
| Deployment | CDK: CloudFront, API Gateway, Fargate, RDS (`infra/`) | [deploy/phase0](deploy/phase0/README.md): one GPU machine with Docker, Prometheus and Grafana; [deploy/k8s](deploy/k8s/README.md): EKS Auto Mode, vLLM production-stack with session routing, KEDA, Prometheus and Grafana |
| Benchmarks | none | [bench/](bench): throughput and latency sweep, prefix cache on/off, load test with simulated customers, summary tables |
| Evals | as before | the same cases and gates; the report adds how much of the prompt the prefix cache served |
| LangSmith project | banking-agent-demo | banking-agent-vllm (same eval dataset, so experiments compare directly) |

**Start with Phase 0** ([deploy/phase0/README.md](deploy/phase0/README.md)): one GPU machine for a
day answers the model, cache and batching questions before any Kubernetes work. Then the cluster:
[deploy/k8s/README.md](deploy/k8s/README.md).

New offline tests: `python tests/test_vllm_offline.py` (the agents talking to a fake vLLM server:
session header, cache salt, tool calls, cached tokens, prompt stability) and
`python tests/test_bench_offline.py` (the benchmark tools).

**How the app works, with diagrams: [ARCHITECTURE.md](ARCHITECTURE.md).** The sections below are
the original step-by-step setup, which still applies apart from the model and the deployment.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # fill in your consumer key and the two customer logins
```

## 1. Check the OBP connection

```bash
python -m scripts.smoke_test
```

Logs in customers A and B, lists A's accounts, balances and transactions,
then confirms B is refused when reading A's account.

## 2. Pick demo accounts

```bash
python -m scripts.list_accounts           # table of accounts + a suggestion
python -m scripts.list_accounts --write   # save the suggestion to demo_accounts.json
```

Each customer gets four accounts:

| Role | Shown to the agent | Plays |
|---|---|---|
| current | yes | everyday account |
| savings | yes | savings / tax pot |
| income | no | employer or clients |
| merchants | no | shops and billers |

### If customer B has no accounts

Shared sandbox logins are often locked or empty. If B shows no accounts:

```bash
python -m scripts.setup_customer_b            # dry run
python -m scripts.setup_customer_b --apply    # opens 4 accounts for B, funds B's income from A
```

## 3. Seed realistic transactions

```bash
python -m scripts.seed_transactions            # dry run: prints the plan
python -m scripts.seed_transactions --apply    # creates the transfers
```

Options: `--customer A`, `--cycles 2` (months of activity). Every transfer is
logged to `seed_log.jsonl`. OBP dates transactions when they are created, so
seeded data carries today's date.

## 4. Chat with the phase 1 agent

```bash
pip install -r requirements.txt     # adds LangChain, LangGraph, LangSmith
python -m scripts.chat              # as customer A
python -m scripts.chat --customer B
```

Set `MODEL` (and `OLLAMA_BASE_URL` for Ollama) plus the LangSmith settings in `.env`,
then check the model: `python -m scripts.check_llm`.
Try: "what are my balances?", "how much did I spend on groceries?",
"what did I pay in rent?", "show me account r9y4bwhc43woma1gxvg8k4r5q" (should be refused).

Offline test (no keys needed): `python tests/test_agent_offline.py`

## 5. Phase 2: coordinator + 3 subagents over MCP

```
coordinator ──task──> accounts-agent      ── Accounts MCP     :8101  balances, own-account transfers
            ──task──> transactions-agent  ── Transactions MCP :8102  history, totals, monthly totals, statements
            ──task──> service-agent       ── Service MCP      :8103  profile, address, service requests
            ──task──> chart-agent         ── Charts MCP       :8105  draws saved figures (no bank access)
chat app  ──────────────────────────────────── Verify service   :8104  security codes (not MCP, not the AI)
```

Terminal 1 (leave running):

```bash
python -m mcp_servers.run
```

Terminal 2:

```bash
python -m scripts.chat_multi                 # customer A
python -m scripts.chat_multi --customer B
python -m scripts.chat_multi --thread A-1a2b3c4d   # continue a saved conversation
```

- The customer's OBP token travels only in MCP request headers. Each server asks OBP who
  the token belongs to and allows only that customer's current and savings accounts.
- Transfers (up to £850), address changes, cheque books and KYC updates stop and ask you
  to confirm before they run.
- Address changes and transfers from £500 also ask for a security code emailed to you (see 6).
- A new address is saved at the bank, on the customer's own OBP profile, and the agent reads it
  back from there. Transfers are real OBP transactions on both accounts.
- Cheque books and KYC updates get an `SR-` reference, are logged in `service_requests.jsonl`
  (the back office queue) and sent to the bank as a customer message. Address changes are
  logged there too, as completed.
- Conversations and pending confirmations are saved in `sessions.db`.

Try: "move £20 from current to savings", "move £600 from current to savings" (asks for a code),
"what did I spend on groceries this month?", "chart my grocery spending by month" (the web chat
draws a bar chart from the bank's own monthly totals), "give me a statement for my savings",
"I've moved to 12 Main Street, Maynooth, W23 X1Y2" (asks for a code), "what address do you have
for me?", "order a cheque book", "what service requests do I have?", and as Robert ask for the
balance of Anil's current account by its ID from `demo_accounts.json` (refused).

Offline test (no keys, fake bank): `python tests/test_multi_agent_offline.py`

## 6. Security codes

Address changes, and transfers from `OTP_TRANSFER_THRESHOLD` (default £500) up to the £850
limit, need a 6-digit code:

```
you>  move £600 from current to savings
  CONFIRM: Transfer £600.00 from current to savings
  Go ahead? [y/N] y
  We've emailed a 6-digit security code to k***@gmail.com. It expires in 5 minutes.
  Security code: 482913
bank> Done. £600.00 moved; current is now ...
```

- The code goes from your keyboard straight to the verify service (port 8104). It never enters
  the conversation, so the AI model, `sessions.db` and LangSmith never see it.
- Each code is tied to one exact action (same accounts and amount, or same address), works once,
  expires after 5 minutes and dies after 3 wrong tries. The email says what it approves.
- The MCP tools themselves refuse to transfer or change the address without a confirmed code,
  so the step can't be skipped by the model or by calling the servers directly.
- Transfers between your own accounts don't need a code under UK/EU rules; the demo asks for
  larger ones anyway to show the step.

Setup: with `EMAIL_BACKEND=console` (the default) codes are printed in the MCP server window.
For real email, use a Gmail account: turn on 2-Step Verification, create an app password at
https://myaccount.google.com/apppasswords, then in `.env` set `EMAIL_BACKEND=smtp`,
`SMTP_USER`, `SMTP_PASSWORD` (the app password) and `OTP_EMAIL_TO` (the inbox for codes). Check:

```bash
python -m scripts.check_codes     # settings, a test email, and saving to the bank profile
```

## 7. Phase 3: guardrails, costs and evals

### Guardrails: three layers, cheapest first

| Layer | What it does | Cost |
|---|---|---|
| 1. Rules (`guardrails/`, LangChain middleware) | Blocks card numbers, CVVs, PINs, passwords and security codes typed into the chat and removes them from the conversation before any model, store or trace sees them. Masks emails and phone numbers. Masks card/IBAN/account numbers and internal ids in replies. Hides instruction-like text in transaction descriptions at the bank servers. Caps model and tool calls per turn. | free, instant, always on |
| 2. Amazon Bedrock Guardrails | Prompt attacks (jailbreaks, injection), harmful content, denied topics (investment, tax and legal advice), ML-detected card data, and whether the answer is grounded in what the tools returned. Works with Claude on the Anthropic API too. | about $0.10-0.15 per 1,000 short checks |
| 3. Hard controls (phase 2) | Server-side account checks, y/N confirmations, emailed security codes, the £850 limit. The real security boundary: even if 1 and 2 miss something, the agent can't move money or see another customer's data. | |

Bedrock starts in **shadow mode**: it logs what it would block. Run the evals, check it doesn't block
normal questions, then set `GUARDRAIL_MODE=enforce`. To set it up (needs `aws configure`):

```bash
python -m scripts.setup_guardrail --test     # creates it from guardrails/bedrock_policy.py
```

LangSmith traces are redacted before they leave your PC (codes, card data, emails, phones, IBANs,
postcodes, address fields). `LANGSMITH_HIDE_CONTENT=true` sends no message text at all.

### Cost tracker

Every model call (coordinator and subagents) and every Bedrock check is priced and shown after
each turn (`[cost] $0.00093 · 4 model calls · 9.2k tokens · 3.1s`), and saved to `telemetry.db`.

```bash
python -m scripts.cost_report --days 30      # per day, customer, agent, model, and guardrails
```

### Eval suite

44 cases on a fixed fake bank (`evals/fixture.py`), so answers are known and transfers are safe:
balances, spending, statements, follow-ups, write actions, security, prompt injection (including
instructions hidden in a transaction description), PII and scope. Code checks the figures, that
every £ amount in an answer came from the bank's tools (the AI never adds up or subtracts by
itself), the subagents and tools used, proposed actions, leaks, guardrails and efficiency; Claude
Sonnet 5.5 judges refusals and grounding with `--judge`.

```bash
python -m evals.run                          # all cases (+ a LangSmith experiment if a key is set)
python -m evals.run --judge                  # add the judge
python -m evals.run --category security --repeat 3
python -m evals.run --list
```

Gates (exit code 1 if any fails): security, injection and PII cases 100%; overall pass rate 90%;
average cost per case $0.01; p95 latency 45 s; no harness errors. `.github/workflows/ci.yml`
runs the offline tests on every push and the eval suite on pull requests, main and nightly.

Offline tests: `python tests/test_guardrails.py` and `python tests/test_evals_offline.py`

## 8. Phase 4, step 1: web chat

The same agents and safety rules, behind a web API and a chat page.

Terminal 1: `python -m mcp_servers.run`   Terminal 2: `python -m api`   Then open http://127.0.0.1:8000

- Sign in as a demo customer (Robert or Anil). On AWS this becomes the bank's sign-in (Cognito).
- Transfers and service requests appear as a payment slip with Approve and Decline. Address changes and
  transfers from £500 ask for the emailed code on the slip itself.
- "What happened" shows, for each answer, the specialist, the bank tools and arguments, any guardrail
  that stepped in, and the cost.
- "Your conversations" (on the left; on a phone or tablet, behind the menu button) lists your recent
  conversations, newest first, with "What happened" on the right. Each is
  titled with its first message, with card numbers, codes and contact details taken out; open one to
  carry on where you left off, charts and any waiting slip included. It survives signing out.
- The bank token stays on the server: the browser only holds a random session id (HttpOnly,
  SameSite=Strict cookie). Customers can only open their own conversations. Security headers and a
  per-session rate limit are on.

| Endpoint | What it does |
|---|---|
| `POST /api/login` | dev sign-in: `{customer: "A"}` (owner) or `{email}` (visitor, step 4a) |
| `POST /api/chat` | `{message, thread_id?}` -> 202 and a `turn_id`; the answer comes later |
| `GET /api/turns/{turn_id}` | the answer so far: running (with live steps), answered, a slip, or failed |
| `POST /api/confirm` | `{thread_id, approve, code?}` -> code needed, wrong code, next slip, or 202 and a `turn_id` |
| `GET /api/history` | a conversation's messages, any slip still waiting, any answer still on its way |
| `GET /api/conversations` | your recent conversations: title, last used, waiting for an approval or answering |
| `POST /api/guest/reset` | visitors: new balances and activity, conversations cleared |
| `GET /healthz` | health checks (the container's own) |

Offline test (no keys): `python tests/test_api_offline.py`

## 9. Phase 4, step 2: Postgres and Docker

On AWS the app runs in containers, and files inside a container vanish when it restarts. So
everything the app saves now goes into one Postgres database when `DATABASE_URL` is set:
conversations (with any slip waiting for approval), security codes, service requests and the
cost ledger. Without `DATABASE_URL` nothing changes: SQLite files, as before. Requests from the old
`service_requests.jsonl` are copied into the database once, on first use.

**Install Docker Desktop** (free for personal use): https://docs.docker.com/desktop/setup/install/windows-install/
Use the WSL 2 option it suggests, restart, and open Docker Desktop once so it's running.

**Run the whole app in Docker**, the way AWS will run it (Postgres + bank servers + web chat):

```powershell
docker compose up --build          # first build takes a few minutes
# open http://127.0.0.1:8000
docker compose logs -f mcp         # in another terminal: security codes appear here (EMAIL_BACKEND=console)
docker compose down                # stop; conversations stay in the database (down -v deletes it)
```

Stop `python -m api` / `python -m mcp_servers.run` first if they're running (same ports).
Docker reads your `.env`, with three things to know:
- Model: Docker can't see `aws configure` credentials. With `MODEL=bedrock_converse:...`, put a Bedrock
  API key in `AWS_BEARER_TOKEN_BEDROCK`, or use `MODEL=anthropic:...` with `ANTHROPIC_API_KEY`. (On AWS the
  container gets credentials from its role.) Same for the Bedrock guardrail: without credentials it is
  skipped and logged, never blocking.
- A value containing `$` needs single quotes in `.env`: `PASSWORD='ab$cd'`.
- `demo_accounts.json` and `seed_log.jsonl` are copied into the image; `.env` never is.

**Or keep running Python on your PC with Postgres from Docker:**

```powershell
docker compose up -d db
# in .env:  DATABASE_URL=postgresql://bank:bank-local-only@localhost:5432/bank
python -m mcp_servers.run          # both say "Records: Postgres bank@localhost:5432/bank"
python -m api
```

Look inside the database: `docker compose exec db psql -U bank -d bank -c "\dt"`

**Tests on Postgres** (with `docker compose up -d db` running; each run uses its own throwaway schema):

```powershell
python tests/test_storage.py                    # SQLite only
$env:TEST_DATABASE_URL="postgresql://bank:bank-local-only@localhost:5432/bank"
python tests/test_storage.py                    # SQLite, then the same checks on Postgres
python tests/test_api_offline.py                # the web API end to end on Postgres
Remove-Item Env:TEST_DATABASE_URL
```

## 10. Phase 4, step 3: the AWS setup as code

> **Claude version only.** This copy has no `infra/` folder; it deploys with
> [deploy/phase0](deploy/phase0/README.md) and [deploy/k8s](deploy/k8s/README.md). Sections 10 and
> 13 describe the original's Fargate set-up, kept for reference.

`infra/` describes everything the app needs on AWS, in Python (AWS CDK). One command creates it,
one deletes it, so it only exists (and costs) while you're demoing.

| Stack | What it is | Cost |
|---|---|---|
| `BankBudget` | an email at 50%, 80% and 100% of $25 a month, and when the month is heading past it. Counts spending *before* credits, so it warns while credits are paying | free; keep it |
| `BankApp` | the app: CloudFront (https), API Gateway, one small Fargate task (web chat + bank servers), the smallest Postgres, open 09:00-17:00 Irish time. See section 13 | about $14 a month |

Passwords and API keys never go into the templates: `python -m infra.app_secrets push` copies them
from `.env` to AWS Secrets Manager, and the containers read them from there when they start.

**One-time setup** (from the project folder, `.venv` active):

```powershell
pip install -r infra/requirements.txt
npm install -g aws-cdk
aws sts get-caller-identity                              # which AWS login you're using
aws freetier get-account-plan-state --region us-east-1   # FREE or PAID plan
```

**Step 3: the budget alarm only.** Put `BUDGET_EMAIL=you@example.com` in `.env`, then:

```powershell
cd infra
cdk bootstrap            # once per account and region: CDK's own small bucket and image store
cdk deploy BankBudget    # creates the budget alarm
cdk diff BankApp         # lists what step 4 will create; creates nothing
cd ..
```

**Checks without AWS** (also run on GitHub): `python infra/tests/test_stacks.py`. These build the
templates and check, for example: no NAT gateway, database not public, the app only reachable through
API Gateway and CloudFront, the opening hours, no password in any template, and the AWS security checks (cdk-nag), where every
finding is either fixed or accepted with a written reason in `infra/stacks/bank.py`.

## 11. Phase 4, step 4a: open to visitors

Anyone with the link can try the assistant without touching the real sandbox: each visitor gets a
**pretend bank of their own** (`obp/guest_bank.py`), kept in the same database as everything else.
It has a current and a savings account, about two months of everyday activity (salary, rent,
groceries, subscriptions) and an address, and it behaves like the OBP sandbox: same tools, same
limits, same confirmation slips and security codes. It's deleted 24 hours after it was opened, with
the visitor's conversations and requests. "Start over" gives them a fresh one.

Your two demo customers (Robert, Anil) still use the real OBP sandbox, side by side.

**Send, then check back.** An answer takes 5-30 seconds, longer than gateways like to hold a request
open. So the page sends the message, gets a ticket back at once (`turn_id`), and checks every second,
showing each step as it happens ("Accounts specialist", "Using list my accounts"). Answers, waiting
slips and sign-ins are all in the database, so a restart, a deploy or a second copy of the app
signs nobody out, and a reload brings back the conversation, the slip and even the code box.

**Limits for visitors** (you, the owner, have none), set in `.env`:

| Setting | Default | What it does |
|---|---|---|
| `GUEST_DAILY_MESSAGES` | 30 | messages per visitor per 24 hours |
| `GUEST_SIGNUPS_PER_IP_HOUR` | 5 | new demo banks per internet address per hour |
| `DAILY_SPEND_CAP_USD` | 2.00 | what all visitors together may spend on the model per day; then "come back tomorrow" |
| `GUEST_TTL_HOURS` | 24 | how long a demo bank lives |

Also set a monthly spend limit in the Anthropic Console (Settings > Limits) as a last line of defence.

**APP_SECRET.** A visitor's bank token is signed, so nobody can make one up. Put a long random value in
`.env` (the web chat and the bank servers must share it):

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"     # paste the result as APP_SECRET=...
```

Without it the app uses a development secret and logs a warning.

**Security codes for visitors** go to the address they signed in with, once the sign-in has proved
the address is theirs (section 12). With the simple form (`AUTH_MODE=dev`) anyone could type anyone's
address, so there visitors' codes go to `OTP_EMAIL_TO`, like the demo customers' do.

**Try it:** start the app as before (or `docker compose up --build`), open http://127.0.0.1:8000, and
sign in with any email under "Try the banking assistant".

Offline tests (no keys): `python tests/test_guest_bank.py` and `python tests/test_api_offline.py`;
both also run on Postgres with `TEST_DATABASE_URL` set.

## 12. Phase 4, step 4b: real sign-in

`AUTH_MODE=public` replaces the simple form with real sign-in, built into the app (`api/signin.py`):

- **Continue with Google.** Google tells the app the person's verified email address. The app
  checks the reply came from Google, for this app, for this sign-in (state, nonce and PKCE).
- **Or email me a code.** A 6-digit code, valid 10 minutes, three tries, sent the same way as
  security codes. Limits keep the app from flooding anyone's inbox: 3 codes per address per
  15 minutes, 10 per internet address per hour, 300 a day in total (Gmail sends at most 500 a day).

Either way the address is proven, so visitors' security codes go to it. Signing in with an address
in `OWNER_EMAILS` makes you the owner: you pick Robert or Anil (and can switch with "Switch customer").

**Set up Google sign-in** (free, about 10 minutes):

1. Open https://console.cloud.google.com, and create a project (project picker at the top >
   New project), named for example "Demo Bank assistant".
2. Open **Google Auth Platform** (https://console.cloud.google.com/auth/overview) > **Get started**.
   App name: "Demo Bank assistant"; support email: yours; Audience: **External**; contact email:
   yours. Agree and **Create**.
3. **Clients** > **Create client**. Application type **Web application**, name "Demo Bank web".
   Under **Authorized redirect URIs** add `http://127.0.0.1:8000/api/auth/google/callback`. **Create**,
   then copy the **Client ID** and **Client secret** (the secret is only shown once).
4. **Audience** > **Publish app**. Otherwise only test users you list can sign in. The app only asks
   for name and email address, which needs no review by Google.
5. In `.env` (never in chat or GitHub):

```
AUTH_MODE=public
PUBLIC_URL=http://127.0.0.1:8000
GOOGLE_CLIENT_ID=...apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=...
OWNER_EMAILS=you@gmail.com
EMAIL_BACKEND=smtp            # so sign-in codes really arrive (console: they're printed by the web chat)
```

Then `docker compose up --build` (or `python -m api`) and open http://127.0.0.1:8000. On AWS the
app gets an https address in step 5; that address's `/api/auth/google/callback` is then added to the
same Google client, and `PUBLIC_URL` set to it.

Without `GOOGLE_CLIENT_ID` the page offers email codes only. `AUTH_MODE=dev` brings the simple form back.

## 13. Phase 4, step 5: the app on AWS, open 09:00-17:00

```
browser --https--> CloudFront --> API Gateway --VPC link--> Fargate task [ web chat | bank servers ] --> Postgres
                   opening hours,  pay per request           0.25 vCPU / 1 GB, 1-3 copies           isolated subnet
                   secret header
```

- **CloudFront** gives the app an https address (`https://d....cloudfront.net`), which Google sign-in
  needs. A small CloudFront Function shows a "We're closed" page outside opening hours and passes on
  the visitor's address (for the sign-up limits).
- **API Gateway** instead of a load balancer: about $1 per million requests instead of about $18 a
  month. The page's "is my answer ready?" checks are now *long* checks (the server holds each one up to
  20 seconds until there's news), so a whole conversation takes a handful of requests.
- **Only through CloudFront.** CloudFront adds a secret header; the app refuses requests without it.
  `python -m infra.app_secrets push` makes that secret for you (`ORIGIN_SECRET`).
- **Opening hours** (`OPEN_HOURS`, Irish time): the database starts at 08:30 and stops at 17:10, the
  app starts at 08:45 and stops at 17:00. While open: 1 copy, up to 3 when busy (`APP_MAX_COPIES`).
- **Invite-only** if you want it: with `OPEN_TO_PUBLIC=0` only `OWNER_EMAILS` and `INVITED_EMAILS` can sign
  in, and others aren't even sent a sign-in code. `OPEN_TO_PUBLIC=1`: anyone.
- The visitors' bank now uses the same single-transfer limit as yours (`OBP_MAX_TRANSFER`, £800).

**Cost** (eu-west-1, open 8¼ hours a day, every day; checked October 2026):

| Part | About |
|---|---|
| Fargate task, 0.25 vCPU + 1 GB, while open | $3.70 |
| Its public IPv4 address, while open | $1.25 |
| Postgres db.t4g.micro, while running | $4.30 |
| Postgres storage, 20 GB (charged even when stopped) | $2.50 |
| Secrets Manager (app secret + database password) | $0.80 |
| Cloud Map (private DNS zone + 1 entry) | $0.60 |
| API Gateway, CloudFront, CloudFront Function, schedules | $0 at demo traffic (free allowances) |
| CloudWatch logs, image storage | under $0.50 |
| **Total** | **about $14 a month**, plus about $0.015 an hour for each extra copy when busy |

The model (Anthropic or Bedrock) is billed separately; visitors' share is capped by `DAILY_SPEND_CAP_USD`.

**Deploy** (Docker Desktop running; from the project folder, `.venv` active):

1. In `.env`: `OWNER_EMAILS`, `GOOGLE_CLIENT_ID` (+ secret), `EMAIL_BACKEND=smtp` and the `SMTP_*` lines,
   `APP_SECRET`, your `MODEL` and its key. Optional: `INVITED_EMAILS=friend@gmail.com,...`,
   `OPEN_HOURS=09:00-17:00` (or `always`).
2. Copy the secrets to AWS: `python -m infra.app_secrets push` (it warns if one the app needs is empty).
3. `cd infra`, then `cdk deploy BankApp`. The first time takes 15-25 minutes (CloudFront and the
   database are the slow parts). Deploy during opening hours: deployed in the evening, the app and
   database run until the next closing time (about $0.50).
4. It prints `AppUrl` and `GoogleRedirectUri`. Add `GoogleRedirectUri` to your Google client
   (Google Auth Platform > Clients > your client > Authorized redirect URIs > Add URI > Save).
5. Open `AppUrl` and sign in with your owner address.
6. To let anyone use "Continue with Google", Google wants a home page and a privacy notice. In Google
   Auth Platform > **Branding**: App home page = `AppUrl`; Privacy policy = `PrivacyUrl` (the app's
   `/privacy.html`, served even when closed; its contact address is `CONTACT_EMAIL`, else your first
   `OWNER_EMAILS`); Authorized domains = the `d....cloudfront.net` part of `AppUrl`; no logo (a logo
   means a review by Google). Save, then **Audience** > **Publish app**.
7. For everyone, not just invited addresses: `OPEN_TO_PUBLIC=1` in `.env`, then `cdk deploy BankApp`.

Logs: `aws logs tail /banking-agent/app --follow` (app) and the `BankApp-ApiLogs...` group (API Gateway).
Change the hours: edit `OPEN_HOURS`, then `cdk deploy BankApp` again. Remove it all: `cdk destroy BankApp`
(the secret stays, $0.40 a month: `python -m infra.app_secrets delete` removes it too).

## 14. Phase 4, step 6: charts

Ask for a chart in the web chat ("chart my grocery spending by month", "show that as a line",
"where does my money go? as a donut") and it appears under the answer.

```
coordinator --task--> transactions-agent --> Transactions MCP: works out the figures, saves them
                                                 as a dataset (ds-4f2a9c1e), returns only the id
            --task--> chart-agent        --> Charts MCP :8105: opens that id (this customer's only)
                                                 and returns a chart; no bank access
web API     copies the chart from the chart tool's result (never from the AI's words) --> the page draws it
```

- **Two steps, one id.** The figures come from the transactions server's code; only the dataset id
  passes through the AI. "Show that as a line" goes straight to the chart agent with the same id:
  the bank isn't asked again.
- **Forms.** Month by month: `column_chart` (several categories side by side), `line_chart` (up to 3
  lines), `stacked_column_chart` (2-4 categories). A breakdown by category or shop: `bar_chart`
  (ranked; the rest as "Other"), `donut_chart` (5 slices and "Other"). A form that doesn't fit the
  figures is refused with the forms that do.
- **Categories** live in one place, `obp/categories.py` (groceries, eating out, coffee, transport,
  subscriptions, bills, gym, rent, shopping; salary, invoices, interest). The AI names a category;
  the bank servers know its payee words. `tests/test_categories.py` checks that every demo
  transaction (visitors' banks, Robert and Anil, the eval bank, the fake bank) is in exactly one.
- **Totals leave out transfers between your own accounts** (the same amount out of one and into the
  other on the same day): moving money to savings is neither spending nor income. Lists and
  statements still show them. A month that isn't over yet is marked "so far".
- **Saved figures** are in the `chart_datasets` table (`CHART_DB` with SQLite). A visitor's go with
  their bank (24 hours, or "Start over"); others are deleted after 7 days.

No AWS changes: the charts server runs in the same container as the other bank servers.

Offline checks: `python tests/test_categories.py`, and the chart conversation in
`python tests/test_api_offline.py`.

## Layout

```
obp/client.py                  Direct Login, reads, transfers, customer messages
scripts/smoke_test.py          connection and authorisation check
scripts/list_accounts.py       survey accounts, write demo_accounts.json
scripts/setup_customer_b.py    open and fund accounts for B
scripts/probe_limit.py         find the largest transfer that needs no approval
scripts/seed_transactions.py   create realistic activity
agents/context.py              per-customer token + allowed accounts (never shown to the model)
agents/tools.py                read tools with account checks; totals computed in code
agents/llm.py                  model from .env: Bedrock, Ollama or Anthropic
agents/agent.py                phase 1 single agent
scripts/check_llm.py           check the model answers and can call tools
scripts/chat.py                terminal chat with tracing
tests/test_agent_offline.py    fake model + fake bank end-to-end test
obp/fake.py                    in-memory fake bank (OBP_FAKE=1)
mcp_servers/common.py          identity from the token, allowed accounts, filters
mcp_servers/accounts/server.py      Accounts MCP server
mcp_servers/transactions/server.py  Transactions MCP server
mcp_servers/charts/server.py   Charts MCP server: draws saved figures (no bank access)
mcp_servers/datasets.py        figures saved for charts, by id, per customer
obp/categories.py              spending and income categories, and their payee words
tests/test_categories.py       every demo transaction is in exactly one category
api/charts.py                  checks a chart before the page draws it
mcp_servers/services/server.py      Service MCP server
mcp_servers/verification.py    security codes: email, checks, the verify service
mcp_servers/run.py             start the servers
scripts/check_codes.py         check code email + bank profile storage
scripts/probe_challenge.py     showed OBP needs a second approver above the limit
agents/coordinator.py          Deep Agents coordinator + 4 subagents (accounts, transactions, service, charts)
scripts/chat_multi.py          phase 2 terminal chat with confirmations
tests/test_multi_agent_offline.py  phase 2 end-to-end test
guardrails/pii.py              deterministic PII detection and masking
guardrails/middleware.py       input/output guardrails, contact masking, call limits
guardrails/bedrock.py          Amazon Bedrock Guardrails client (shadow/enforce)
guardrails/bedrock_policy.py   the Bedrock guardrail, as code
observability/costs.py         per-turn cost meter and the ledger
observability/tracing.py       PII redaction for LangSmith traces
scripts/setup_guardrail.py     create/update the Bedrock guardrail
scripts/cost_report.py         cost report from the ledger
evals/fixture.py               the fixed eval bank and its oracle
evals/cases.py                 the 44 eval cases
evals/harness.py               runs a case against the real agents
evals/evaluators.py            code scorers + the Claude judge
evals/run.py                   the eval CLI, gates, LangSmith experiments
tests/test_guardrails.py       guardrails, costs, trace redaction
tests/test_evals_offline.py    the eval machinery, with a scripted model
.github/workflows/ci.yml       offline tests + eval gates in CI
api/app.py                     web API: sign-in, send-then-check answers, confirmations with codes, limits
api/state.py                   sessions, answers in progress and waiting slips, in the database
api/signin.py                  real sign-in: Google (OpenID Connect) and emailed sign-in codes
mailer/__init__.py             sending email (console or SMTP), for security and sign-in codes
api/actions.py                 what each action looks like on the confirmation slip
obp/guest_bank.py              visitors' pretend banks (OBP-compatible, in the database)
tests/test_guest_bank.py       visitors' banks on SQLite and Postgres
web/                           the chat page (index.html, app.css, app.js)
tests/test_api_offline.py      the web API end to end, with a scripted model
storage/__init__.py            where records go: Postgres (DATABASE_URL) or SQLite files
tests/test_storage.py          every kind of record, on SQLite and Postgres
tests/pg_scratch.py            a throwaway Postgres schema per test run
Dockerfile                     one image: web chat (default) or bank servers
docker-compose.yml             Postgres + bank servers + web chat on your PC
agents/vllm_chat.py            the vLLM model: session header and per-customer cache salt on every call
tests/fake_vllm.py             a stand-in vLLM server for offline tests (scripted answers, simulated cache)
tests/test_vllm_offline.py     the agents against the fake vLLM server
tests/test_bench_offline.py    the benchmark tools against the fake vLLM server
bench/sweep.sh                 E1/E3: vllm bench serve at 1 to 256 requests at once
bench/cache_on_off.py          E2: a 10-message chat, time to first token and cached tokens per message
bench/locustfile.py            E7: simulated customers through the web API
bench/summarise.py             result tables for the report (also from a Kubernetes Job's log)
deploy/phase0/                 one GPU machine: compose file, set-up script, guide
deploy/k8s/cluster.yaml        the EKS cluster (Auto Mode)
deploy/k8s/platform/           GPU node pool, EBS storage class, network policy controller
deploy/k8s/helm/               vLLM production-stack (vLLM + session router) and monitoring values
deploy/k8s/bank/               Postgres, bank servers, web app, network policies (kustomize)
deploy/k8s/llm/                KEDA autoscaling, network policies, alerts, a round-robin Service for E6
deploy/k8s/loadtest/           benchmark Jobs (E3 sweep, E7 Locust)
deploy/k8s/env_to_secret.py    .env to the cluster's Secret, without writing it to disk
```
