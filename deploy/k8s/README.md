# Phases 1–4: the cluster (Amazon EKS)

The same app as Phase 0, on Kubernetes: the web app, bank servers and Postgres on CPU nodes, and
vLLM on A10G GPU nodes (g5.xlarge, the GPU Phase 0 measured) behind a session router that keeps each conversation on one GPU. KEDA
scales vLLM on its queue and KV-cache use; Prometheus and Grafana show what it's doing.

```
your PC ──kubectl port-forward──▶ bank/api ──▶ llm/vllm-router-service ──▶ vLLM pod(s), 1 GPU each
                                     │          (same x-session-id → same pod)
                                     ├──▶ bank/mcp (bank servers) ──▶ Open Bank Project
                                     └──▶ bank/postgres
monitoring: Prometheus + Grafana  ·  keda: autoscaling  ·  loadtest: benchmark Jobs
```

**Cost:** about $1.6 an hour with one GPU and $2.9 with two (EKS control plane $0.10/h, Auto
Mode's node fee, a NAT gateway, small CPU nodes, each g5.xlarge at $1.12/h), plus about $1 of NAT
traffic each time a GPU node downloads the vLLM image and the model. A test afternoon of E6 and E8
is roughly $10-15. The cluster costs money even when idle, so **create it for a test day and delete
it after** (step 9).

## 0. Before you start (PowerShell)

- Signed in to AWS as before (`aws login`, the krish-admin user), region eu-west-1.
- Install [eksctl](https://eksctl.io/installation/), kubectl and Helm, e.g.
  `winget install Kubernetes.kubectl` and `winget install Helm.Helm`.
- GPU quota: Service Quotas → Amazon EC2 → *Running On-Demand G and VT instances*: **4** vCPUs for
  one GPU, **8** for two (E6, E8). **Stop the Phase 0 machine first**: running, it uses 4 of them.
- CPU nodes use *Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances*; check it's at
  least 8: `aws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A --region eu-west-1 --query Quota.Value`
- Phase 0's E9 passed for the model you're deploying (`openai/gpt-oss-20b` here).

All commands run from the project folder, `D:\banking-agent-vllm`.

## 1. Create the cluster (15–20 minutes)

```powershell
eksctl create cluster -f deploy/k8s/cluster.yaml
kubectl apply -f deploy/k8s/platform/
```

EKS Auto Mode starts with no nodes; they appear when pods need them. `platform/` adds the GPU node
pool (A10G nodes, only for vLLM, removed when empty), encrypted EBS volumes and the network policy
controller.

## 2. Monitoring and autoscaling

```powershell
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add kedacore https://kedacore.github.io/charts
helm repo add vllm https://vllm-project.github.io/production-stack
helm repo update
helm upgrade --install monitoring prometheus-community/kube-prometheus-stack -n monitoring --create-namespace -f deploy/k8s/helm/monitoring.values.yaml
helm upgrade --install keda kedacore/keda -n keda --create-namespace
```

## 3. vLLM and the router

```powershell
helm upgrade --install vllm vllm/vllm-stack --version 0.1.13 -n llm --create-namespace -f deploy/k8s/helm/vllm-stack.values.yaml
kubectl apply -k deploy/k8s/llm
kubectl -n llm annotate scaledobject vllm autoscaling.keda.sh/paused-replicas="1" --overwrite
```

The last line holds vLLM at one pod while you test. Without it, KEDA follows opening hours (one
pod 08:40–17:05 Irish time, none outside). Remove it with `paused-replicas-` when you want that.

Watch it start: a GPU node launches (a few minutes), then the image pulls and the model downloads.

```powershell
kubectl -n llm get pods -w
kubectl -n llm logs -f deploy/vllm-gptoss-deployment-vllm
```

As in Phase 0, note the KV cache size and maximum concurrency lines from the log.

## 4. The app's image

```powershell
$ACCOUNT = aws sts get-caller-identity --query Account --output text
$REGISTRY = "$ACCOUNT.dkr.ecr.eu-west-1.amazonaws.com"
aws ecr create-repository --repository-name banking-agent-vllm --region eu-west-1 --image-scanning-configuration scanOnPush=true
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $REGISTRY
docker build -t "$REGISTRY/banking-agent-vllm:v1" .
docker push "$REGISTRY/banking-agent-vllm:v1"
(Get-Content deploy/k8s/bank/kustomization.yaml) -replace 'ACCOUNT_ID', $ACCOUNT | Set-Content deploy/k8s/bank/kustomization.yaml
```

For a new version, build and push with a new tag (`v2`), change `newTag` in
`deploy/k8s/bank/kustomization.yaml`, and apply step 6 again.

## 5. Secrets

Your `.env` becomes a Secret without ever being written to disk; the database password is random.
Neither goes into git or a manifest.

```powershell
kubectl create namespace bank
kubectl -n bank create secret generic postgres --from-literal=password=$([guid]::NewGuid().ToString('N'))
python deploy/k8s/env_to_secret.py | kubectl apply -f -
```

## 6. The app

```powershell
kubectl apply -k deploy/k8s/bank
kubectl -n bank get pods -w                    # postgres, mcp, api all Running and 1/1
kubectl -n bank port-forward svc/api 8000:8000
```

Open http://127.0.0.1:8000: the same web chat, now answered by vLLM on the cluster. Traces go to
the `banking-agent-vllm` project in LangSmith.

Grafana: `kubectl -n monitoring port-forward svc/monitoring-grafana 3000:80`, then
http://127.0.0.1:3000 (user `admin`; the password is in the `monitoring-grafana` secret, see
`helm/monitoring.values.yaml`). The vLLM dashboard shows TTFT, latency, queue, KV use and cache hits.

## 7. The experiments on the cluster

| # | How |
| --- | --- |
| E9 | `kubectl -n llm port-forward svc/vllm-router-service 8001:80`, then on your PC `python -m evals.run --judge` (your `.env` points `VLLM_BASE_URL` at 127.0.0.1:8001). |
| E3 | `kubectl apply -f deploy/k8s/loadtest/namespace.yaml`, `kubectl apply -f deploy/k8s/loadtest/e3-sweep.yaml`, `kubectl -n loadtest logs -f job/e3-sweep`. Afterwards `kubectl -n loadtest logs job/e3-sweep > e3-cluster.log` and `python -m bench.summarise --from-log e3-cluster.log`. |
| E2 | With the router port-forward from E9 open: `python -m bench.cache_on_off --session-id --label cluster-cache-on`. For the cache-off run, set `enablePrefixCaching: false` in `helm/vllm-stack.values.yaml`, run step 3's `helm upgrade` line, wait for the new pod, run again with `--label cluster-cache-off`, then set it back. |
| E6 | Two pods: `kubectl -n llm annotate scaledobject vllm autoscaling.keda.sh/paused-replicas="2" --overwrite`. Then run the E2 conversation from inside the cluster, once through the router and once through a plain round-robin Service (a port-forward always picks one pod, so it can't show this): `kubectl -n loadtest run e6-router --rm -i --restart=Never --image="$REGISTRY/banking-agent-vllm:v1" --env=BENCH_RESULTS=/tmp -- python -m bench.cache_on_off --session-id --salt --base-url http://vllm-router-service.llm.svc.cluster.local/v1 --label router`, and the same with `--base-url http://engines-roundrobin.llm.svc.cluster.local/v1 --label roundrobin`. `--salt` makes each run a new customer, as the app does, so neither run gets the other's cache. Compare the share of prompt tokens from the cache. |
| E7 | Test-only settings: `kubectl -n bank set env deploy/api AUTH_MODE=dev GUEST_SIGNUPS_PER_IP_HOUR=10000 GUEST_DAILY_MESSAGES=10000 DAILY_SPEND_CAP_USD=1000`. Then `kubectl -n loadtest create configmap locustfile --from-file=bench/locustfile.py`, `kubectl apply -f deploy/k8s/loadtest/e7-locust.yaml`, `kubectl -n loadtest logs -f job/e7-locust`. Undo afterwards: `kubectl -n bank set env deploy/api AUTH_MODE- GUEST_SIGNUPS_PER_IP_HOUR- GUEST_DAILY_MESSAGES- DAILY_SPEND_CAP_USD-`. |
| E8 | Scale-out: with one pod Ready, let KEDA decide (`kubectl -n llm patch scaledobject vllm --type merge -p '{"spec":{"minReplicaCount":1}}'`, then `paused-replicas-`), start `kubectl apply -f deploy/k8s/loadtest/e8-scale.yaml` (32 at once through the router) and watch `kubectl -n llm get pods -w` and Grafana: note when the load started, when the second pod appeared, and when it was Ready and taking requests. Cold start: `paused-replicas="0"`, wait until the GPU nodes are gone (`kubectl get nodes -w`), then time 0 → 1 Ready (see the commands below the table). Afterwards set `minReplicaCount` back to 0. |

`kubectl -n loadtest delete job <name>` before running a Job again.

E8 cold start, timed (PowerShell):

```powershell
$t = Get-Date
kubectl -n llm annotate scaledobject vllm autoscaling.keda.sh/paused-replicas="1" --overwrite
do { Start-Sleep 5 } until (kubectl -n llm get pods -l app=bank-llm-engine -o name)
kubectl -n llm wait --for=condition=Ready pod -l app=bank-llm-engine --timeout=30m
(Get-Date) - $t          # node launch + image pull + model download + load and compile
```

## 8. Change settings

- vLLM flags or the model: edit `helm/vllm-stack.values.yaml`, run the `helm upgrade` line in
  step 3 again. Changing the model? Change `modelURL`, `--tool-call-parser`, and `MODEL` in
  `bank/api.yaml` together, then `kubectl apply -k deploy/k8s/bank`.
- App settings from `.env`: run the `env_to_secret.py` line again, then
  `kubectl -n bank rollout restart deploy/api deploy/mcp`.

## 9. Delete everything

```powershell
kubectl delete -k deploy/k8s/bank
kubectl -n bank delete pvc --all                # the database disk isn't deleted with the app
helm -n llm uninstall vllm
eksctl delete cluster -f deploy/k8s/cluster.yaml
```

The ECR image stays (cents a month); `aws ecr delete-repository --repository-name banking-agent-vllm --force` removes it.

## Notes

- **Model download:** each vLLM pod downloads the model from Hugging Face into its own scratch
  disk when it starts. Copying the weights to S3 once and loading from there is faster and is the
  next improvement for cold starts (E8).
- **Nodes and subnets:** with eksctl's default VPC, Auto Mode may place nodes in public subnets.
  No ports are open to them, and nothing in this set-up is exposed to the internet; for
  production, use a NodeClass limited to private subnets.
- **vLLM API key:** off, because only the router and the load-test namespace can reach the vLLM
  pods (`llm/networkpolicies.yaml`).
