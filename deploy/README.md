# Deploying to Oracle Cloud

[`cloud-init.yaml`](cloud-init.yaml) provisions the whole app on a fresh Ubuntu instance: uv, Python 3.14, the repo, a systemd service and nginx in front of it. Paste it into **Show advanced options → Management → Cloud-init script** when creating the instance.

## Shape

Measured with [`scripts/measure_memory.py`](../scripts/measure_memory.py) — ~1.5 GB at startup, ~3 GB after moderate use, growth per new query decaying as the allocator arena settles:

| | |
|---|---|
| **OCPUs** | 2 |
| **Memory** | 12 GB (4 GB would work; 12 is generous) |
| **Disk** | 50 GB+ (the checkout is ~420 MB, plus a ~5 GB dependency tree) |
| **Image** | Ubuntu 24.04, x86_64 or Ampere A1 |

Either architecture is fine — torch, faiss-cpu and numpy all publish `manylinux` wheels for `aarch64`, so nothing gets built from source.

## Two things cloud-init cannot do

**1. Open port 80 in the security list.** This is a VCN setting, outside the instance. Networking → Virtual Cloud Networks → your VCN → the subnet's security list → Add Ingress Rule:

- Source CIDR `0.0.0.0/0`, IP protocol TCP, destination port `80`

Without it the instance is provisioned and healthy but unreachable. Cloud-init already handles the *other* firewall — OCI's Ubuntu images carry an iptables policy that rejects everything except port 22, which catches people out because it blocks the port even after the security list allows it.

**2. Set `REPO`.** Edit it under `write_files: /etc/if-recommender.env`. If the repo is private, either make it public or use a deploy key:

```bash
# on the instance, before the clone step succeeds
sudo -u ubuntu ssh-keygen -t ed25519 -N "" -f /home/ubuntu/.ssh/id_ed25519
sudo -u ubuntu cat /home/ubuntu/.ssh/id_ed25519.pub   # add to GitHub deploy keys
```

then set `REPO=git@github.com:ehenestroza/if-recommender.git` and re-run the clone.

## First boot takes a while

Ten to twenty minutes: apt, then uv downloading CPython and ~5 GB of wheels (torch dominates), then the service loading models and ranking tables. Watch it:

```bash
tail -f /var/log/cloud-init-output.log     # provisioning
journalctl -u if-recommender -f            # the app itself
```

The app is up when the log shows Gradio listening on `127.0.0.1:7860`.

## Operating it

```bash
systemctl status if-recommender
systemctl restart if-recommender
journalctl -u if-recommender -n 100

# deploy a new commit
cd /opt/if-recommender && sudo -u ubuntu git pull && \
  sudo -u ubuntu /usr/local/bin/uv sync --frozen && \
  sudo systemctl restart if-recommender
```

Check memory on the real hardware rather than trusting laptop numbers — glibc's allocator fragments differently from macOS's, and that difference is exactly what the growth under load consists of:

```bash
cd /opt/if-recommender && sudo -u ubuntu /usr/local/bin/uv run scripts/measure_memory.py
```

## TLS

Not automated, because it needs a domain name pointed at the instance first. Once you have one:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.example
```

Certbot rewrites the nginx site in place and sets up renewal. Add port 443 to the security list and the iptables rule alongside it:

```bash
sudo iptables -I INPUT 1 -p tcp --dport 443 -m conntrack --ctstate NEW -j ACCEPT
sudo netfilter-persistent save
```

## Notes on the design

**The app binds `127.0.0.1`, not `0.0.0.0`.** Gradio reads `GRADIO_SERVER_NAME`/`GRADIO_SERVER_PORT` from the environment, so the service sets them and `app.py` needs no deployment-specific code. nginx is the only thing that reaches the app, which keeps it off the public interface and gives TLS somewhere to live.

**`uv sync --frozen`, not `uv sync`.** `--frozen` installs exactly what `uv.lock` pins and refuses to re-resolve, so the instance runs the versions that were tested. Without it a deploy months from now would silently pick up newer torch and gradio.

**`MALLOC_ARENA_MAX=2` and `OMP_NUM_THREADS=2`** are set in the service environment. The first bounds glibc's per-thread heap arenas, which is where the app's growth under load actually goes; the second stops torch starting one thread per core when inference is already serialised behind Gradio's queue.

**The clone is shallow.** History carries every superseded copy of the models, FAISS index and ranking tables — none of which delta-compress — so `.git` is several times the size of the checkout. `--depth 1` skips all of it.
