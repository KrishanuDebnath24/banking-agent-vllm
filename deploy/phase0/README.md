# Phase 0: one GPU machine

The whole app (Postgres, bank servers, web chat) plus vLLM on one EC2 GPU machine, started with
Docker Compose. It answers the plan's first questions cheaply, before any Kubernetes work: does
the model pass the evals (E9), what does one request cost (E1), how much does the prefix cache
save (E2), and how many requests fit at once (E3 to E5).

Cost: a g5.xlarge (1× NVIDIA A10G, 24 GB; the results here were measured on one) is about $1.12 an
hour on demand in Ireland. A day of benchmarks is roughly $10. **Stop the machine when you finish.**
(g6e.xlarge, with an L40S 48 GB, is the faster choice where it's offered; it isn't in eu-west-1.)

## 1. Launch the machine (AWS console, eu-west-1)

1. **Check your GPU quota first.** Service Quotas → Amazon EC2 → *Running On-Demand G and VT
   instances* must be at least **4** (vCPUs). New accounts often have 0; request 8 and wait for
   approval (usually hours, sometimes a day).
2. EC2 → **Launch instance**:
   - **AMI:** search *Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023)*
     (it has the NVIDIA driver).
   - **Instance type:** `g5.xlarge` (if there's no capacity, try another subnet or later).
   - **Key pair:** create one, download the `.pem` file.
   - **Network:** allow SSH from **My IP** only. Nothing else is opened; you reach the app
     through an SSH tunnel.
   - **Storage:** 200 GiB gp3 (Docker images and the model download).
3. Note the instance's public IP.

## 2. Copy the project to it (PowerShell on your PC)

`.env`, `demo_accounts.json` and `seed_log.jsonl` aren't in git, so copy the folder itself:

```powershell
cd "D:\banking-agent-vllm"
tar -czf ..\bank.tgz --exclude=.venv --exclude=__pycache__ --exclude=bench/results .
scp -i C:\path\to\key.pem ..\bank.tgz ec2-user@<public-ip>:~
ssh -i C:\path\to\key.pem ec2-user@<public-ip>
```

## 3. Start it (on the GPU machine)

```bash
mkdir -p bank && tar -xzf bank.tgz -C bank && cd bank
bash deploy/phase0/setup-gpu-box.sh        # Docker, GPU access for containers, Compose; once
exit                                        # log out and back in, so docker works without sudo
```

```bash
cd bank
docker compose -f docker-compose.yml -f deploy/phase0/compose.vllm.yml up -d --build
docker compose -f docker-compose.yml -f deploy/phase0/compose.vllm.yml logs -f vllm
```

The first start downloads the model (about 13 GB for gpt-oss-20b) and compiles, so give it 5 to
15 minutes. When the log says the server is up, **write down two lines it printed**: the KV cache
size in tokens, and the maximum concurrency for `--max-model-len`. They set the first numbers to
test in E3 and E4.

If the model fails to start on this GPU, try another candidate: set `VLLM_MODEL` (and the same
name in `MODEL=vllm:...`) and `VLLM_TOOL_PARSER` in `.env`, as vLLM's *Tool Calling* page lists
for that model, then run the `up -d` line again.

## 4. Use it from your PC

Open a tunnel (keep this window open):

```powershell
ssh -i C:\path\to\key.pem -L 8000:127.0.0.1:8000 -L 8001:127.0.0.1:8001 -L 3000:127.0.0.1:3000 -L 9090:127.0.0.1:9090 ec2-user@<public-ip>
```

- **Web chat on vLLM:** http://127.0.0.1:8000, the same page as before.
- **vLLM itself:** http://127.0.0.1:8001/v1 (your PC's `.env` already points `VLLM_BASE_URL` there).
- **Grafana:** http://127.0.0.1:3000, live dashboards (no login needed to view).
- **Prometheus:** http://127.0.0.1:9090, the raw metrics and the alerts (*Alerts* tab).
- **LangSmith:** traces go to the `banking-agent-vllm` project.

### Dashboards

Prometheus reads vLLM's `/metrics` every 5 seconds (`monitoring/prometheus.yml`) and keeps 15
days. Grafana opens on **Banking agent: vLLM at a glance** (`monitoring/grafana/dashboards/bank-llm.json`):

| Panel | Shows | Good |
| --- | --- | --- |
| Requests running / waiting | continuous batching: how many the GPU works on, how many queue | waiting stays 0 |
| KV cache used | conversation memory on the GPU | under 90% |
| Prompt tokens from cache | what the prefix cache saves | above 80% in chats |
| Time to first token, p50 / p95 | how long before an answer starts | p95 under 1 s |
| Time per output token, p50 / p95 | speed of each answer | under 50 ms |
| Tokens per second | generated, prompt computed, prompt from cache | |
| Requests finished / preempted per second | throughput; preempted = KV cache ran out | no preemptions |
| Whole request time, queue time | end to end, and time spent waiting | |

The folder *Banking agent* also has **vLLM (from the vLLM project)**, the dashboard from vLLM's own
examples for this version. The alerts (`monitoring/alerts.yml`) are the cluster's four: requests
waiting, KV cache nearly full, slow first token, low cache hit rate.

Watch Grafana while E3 or E7 runs; set the time range (top right) to the test's window for screenshots.

## 5. The experiments

| # | What | How |
| --- | --- | --- |
| E9 | Quality gate: same 41 cases as the Claude version | On your PC, tunnel open: `python -m evals.run --judge`. Compare the experiment with the Claude one in LangSmith (same dataset). The report also prints how much of the prompt the prefix cache reused. |
| E1 | One request on an idle GPU | On the machine: `CONCURRENCY=1 LABEL=e1 bash bench/sweep.sh` |
| E2 | Prefix cache on vs off | On your PC: `python -m bench.cache_on_off --label cache-on`. Then on the machine add `VLLM_PREFIX_CACHING_FLAG=--no-enable-prefix-caching` to `.env` and run the `up -d` line; on your PC `python -m bench.cache_on_off --label cache-off`. Remove the line and `up -d` again. |
| E3 | Requests at once within the targets | On the machine: `bash bench/sweep.sh` (1 to 256 at once, about 30 minutes) |
| E4 | Batch limits | Set `VLLM_MAX_NUM_SEQS` / `VLLM_MAX_NUM_BATCHED_TOKENS` in `.env`, `up -d`, then `LABEL=e4-seqs64 bash bench/sweep.sh` (one label per setting) |
| E5 | FP8 KV cache | `VLLM_KV_CACHE_DTYPE=fp8` in `.env`, `up -d`, `LABEL=e5-fp8 bash bench/sweep.sh`, and E9 again |
| E7 | Real users through the web app | Test-only settings in the machine's `.env`: `AUTH_MODE=dev`, `GUEST_SIGNUPS_PER_IP_HOUR=10000`, `GUEST_DAILY_MESSAGES=10000`, `DAILY_SPEND_CAP_USD=1000`; `up -d`; then on the machine (Locust's own image, so no Python set-up): `docker run --rm --network host -v "$PWD":/work -w /work locustio/locust -f bench/locustfile.py --host http://127.0.0.1:8000 --headless --users 50 --spawn-rate 1 --run-time 15m --csv bench/results/e7-50users`. Repeat with more users until turns slow past 10 s. |

Summaries: `python3 -m bench.summarise` on the machine prints the tables for the plan's report.
Copy the raw results home with `scp -i C:\path\to\key.pem -r ec2-user@<public-ip>:~/bank/bench/results .`

GuideLLM (optional cross-check for E3), on your PC with the tunnel open:

```powershell
pip install -r bench/requirements.txt
guidellm run --backend kind=openai_http,target=http://127.0.0.1:8001 --profile kind=sweep `
  --constraint kind=max_duration,seconds=60 --data kind=synthetic_text,prompt_tokens=3000,output_tokens=200
```

## 6. Stop paying

EC2 → the instance → **Stop** (the disk stays, with the downloaded model, and costs a little a
month) or **Terminate** (everything goes). Undo the test-only settings before showing anyone.
