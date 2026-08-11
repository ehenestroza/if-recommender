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

Either architecture *runs*, but prefer **x86_64**. The cross-encoder is dynamically quantized to int8 where it pays off, and that depends on the backend: fbgemm (x86) measured **2.02×**, while ARM's qnnpack measured *slower* and is skipped. `model.quantize_reranker: "auto"` picks correctly on both, so an A1 instance is correct — just roughly half the live throughput on `vibe`.

Both architectures are otherwise fine: torch, faiss-cpu and numpy all publish `manylinux` wheels for `aarch64`, so nothing gets built from source.

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

## Verify the boot before touching DNS

Three checks, ten seconds, all runnable while the old instance is still serving. They catch the failure modes that otherwise surface as a browser error with no clue attached:

```bash
# 1. Both ports opened in the OS. Missing 443 is a *refused* connection later,
#    not a timeout, and only after certbot has run — long after the cause.
sudo iptables -L INPUT -n | grep -E 'dpt:(80|443)'      # expect two ACCEPT lines

# 2. The app answers on the bare IP, before any hostname or TLS is involved.
curl -sI http://localhost/                              # expect 200, text/html

# 3. The provisioned checkout is the commit you meant to deploy.
git -C /opt/if-recommender log --oneline -1
```

**Take `cloud-init.yaml` from the repo every time, and do not reuse a copy.** Two ways a stale one gets used: the OCI console pre-fills the previous instance's script, and Cloud Shell's home directory persists between sessions, so last deploy's upload is still sitting there under the right filename with nothing to suggest its age.

From Cloud Shell, fetch it rather than uploading, which makes staleness impossible:

```bash
curl -fsSLO https://raw.githubusercontent.com/ehenestroza/if-recommender/main/deploy/cloud-init.yaml
oci compute instance launch --user-data-file cloud-init.yaml ...
```

A stale script is how check 1 fails, and it fails quietly: the port-443 rule was added after the copy being reused, so nginx listened on 443 with a valid certificate while the kernel rejected every connection to it — three healthy-looking layers above the actual cause.

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

Same for latency, which is the number that governs what a visitor waits on a `vibe` query. It also reports whether int8 quantization engaged:

```bash
sudo systemctl stop if-recommender     # both tools load a second copy of the models
cd /opt/if-recommender && sudo -u ubuntu /usr/local/bin/uv run scripts/measure_latency.py --json
sudo systemctl start if-recommender
```

`"quantized_engine": "fbgemm"` means the 2× path is active. On the 2-vCPU E-series box this measured `t = 0.82s + pairs/91.5`, against `1.45s + pairs/45.2` without it.

## TLS

Not automated, because it needs a domain name pointed at the instance first. Once you have one:

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.example
```

Add port 443 to the **security list** first — that is the VCN setting, same as port 80 above. The instance-side iptables rule for 443 is already opened by cloud-init, so there is nothing to run for it.

Certbot rewrites the nginx site in place and sets up renewal.

**Point DNS at the instance before running certbot.** The HTTP-01 challenge resolves the name and must reach *this* box; running it early fails, and running it against a stale record issues for the wrong host.

**Change the hostname only by reprovisioning, not in place.** Certbot owns the nginx site after it runs, so editing `server_name` afterwards leaves the `location /` proxy in whichever block you did not edit — nginx then answers the new name from a block with no proxy and returns a bare 404 while the app is running perfectly. Set `DOMAIN` in cloud-init and boot a fresh instance instead.

## Notes on the design

**The app binds `127.0.0.1`, not `0.0.0.0`.** Gradio reads `GRADIO_SERVER_NAME`/`GRADIO_SERVER_PORT` from the environment, so the service sets them and `app.py` needs no deployment-specific code. nginx is the only thing that reaches the app, which keeps it off the public interface and gives TLS somewhere to live.

**`uv sync --frozen`, not `uv sync`.** `--frozen` installs exactly what `uv.lock` pins and refuses to re-resolve, so the instance runs the versions that were tested. Without it a deploy months from now would silently pick up newer torch and gradio.

**`MALLOC_ARENA_MAX=2`** is set in the service environment, and **`OMP_NUM_THREADS`/`MKL_NUM_THREADS`** are appended from `nproc` at first boot. The first bounds glibc's per-thread heap arenas, which is where the app's growth under load actually goes; the second gives torch one thread per vCPU, which is where the cores actually get used since inference is serialised behind Gradio's queue.

**The clone is shallow.** History carries every superseded copy of the models, FAISS index and ranking tables — none of which delta-compress — so `.git` is several times the size of the checkout. `--depth 1` skips all of it.
