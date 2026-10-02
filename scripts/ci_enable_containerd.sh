#!/usr/bin/env bash
# Ephemeral GitHub-hosted Linux runners only; never change a developer daemon.
set -euo pipefail
if [[ "${GITHUB_ACTIONS:-}" != "true" || "${RUNNER_OS:-}" != "Linux" ]]; then
  echo "This setup is only for GitHub Actions Linux runners." >&2
  exit 1
fi

sudo python3 - <<'PYCONFIG'
import json
from pathlib import Path

path = Path("/etc/docker/daemon.json")
config = json.loads(path.read_text()) if path.exists() else {}
config.setdefault("features", {})["containerd-snapshotter"] = True
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(config) + "\n")
PYCONFIG
sudo systemctl restart docker
docker info --format '{{json .DriverStatus}}' | python3 -c '
import json, sys
status = json.load(sys.stdin)
assert ["driver-type", "io.containerd.snapshotter.v1"] in status, status
print("containerd image store enabled")
'
