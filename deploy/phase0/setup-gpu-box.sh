#!/usr/bin/env bash
# Get a fresh GPU machine (Deep Learning Base OSS Nvidia Driver GPU AMI, Amazon Linux 2023) ready
# for the phase 0 compose file. Safe to run twice: it only installs what is missing.
#
#   bash deploy/phase0/setup-gpu-box.sh
set -euo pipefail

say() { printf '\n== %s\n' "$*"; }

say "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || { echo "No NVIDIA driver: launch the instance from the Deep Learning Base GPU AMI."; exit 1; }

say "Docker"
if ! command -v docker >/dev/null; then
  sudo dnf install -y docker
fi
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER" || true

say "NVIDIA Container Toolkit (lets containers use the GPU)"
if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
    | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo >/dev/null
  sudo dnf install -y nvidia-container-toolkit
fi
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

say "Docker Compose and Buildx plugins"
plugins=/usr/local/lib/docker/cli-plugins
sudo mkdir -p "$plugins"
if ! sudo docker compose version >/dev/null 2>&1; then
  sudo curl -fsSL -o "$plugins/docker-compose" \
    https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64
  sudo chmod +x "$plugins/docker-compose"
fi
# Compose's build needs buildx 0.17 or later; some images ship an older one, so check the version too.
# (/usr/local/lib/docker/cli-plugins is searched before the system's own plugin folder.)
have=$(sudo docker buildx version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)
if [ -z "$have" ] || [ "$(printf '%s\n' 0.17.0 "$have" | sort -V | head -1)" != "0.17.0" ]; then
  echo "buildx ${have:-missing}: installing the latest"
  tag=$(curl -fsSL https://api.github.com/repos/docker/buildx/releases/latest | grep '"tag_name"' | head -1 | cut -d'"' -f4)
  sudo curl -fsSL -o "$plugins/docker-buildx" \
    "https://github.com/docker/buildx/releases/download/${tag}/buildx-${tag}.linux-amd64"
  sudo chmod +x "$plugins/docker-buildx"
fi

say "git"
command -v git >/dev/null || sudo dnf install -y git

say "Check: a container can see the GPU"
sudo docker run --rm --gpus all public.ecr.aws/amazonlinux/amazonlinux:2023-minimal nvidia-smi -L

say "Ready. Log out and back in once (so 'docker' works without sudo), then follow deploy/phase0/README.md"
