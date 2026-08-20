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

The backend is chosen from `platform.machine()`, not from `torch.backends.quantized.supported_engines`, which reports what the wheel was *built* with rather than what the CPU can run. The Linux aarch64 wheel lists `fbgemm`; selecting it on an A1 made the first prepack raise `RuntimeError: unknown architecure` and the service exited 1 on every restart. macOS arm64 lists only `qnnpack`, so development never saw it.

Both architectures are otherwise fine: torch, faiss-cpu and numpy all publish `manylinux` wheels for `aarch64`, so nothing gets built from source.

The Always Free allocation is Ampere A1 only — the free x86 shapes are `VM.Standard.E2.1.Micro`, 1 GB of RAM, which cannot hold the models. So a free-tier deployment gives up the fbgemm path deliberately, and the compensation is that the same allocation affords twice the cores. See [Launching a free-tier instance](#launching-a-free-tier-instance).

## Launching a free-tier instance

A1 capacity is usually exhausted in popular regions, so the launch is a retry loop rather than a single command. Run it in Cloud Shell, which already has the CLI authenticated as you.

Always Free covers **2 OCPUs and 12 GB of A1 total** in your home region, and 200 GB of block storage across at most two volumes. That is the entire allocation, so the shape config below asks for exactly it.

Budget for `vibe` being noticeably slower than it is on the paid x86 box, from two compounding losses: half the cores, and the int8 path skipped because it does not pay off on ARM (`quantize_reranker: "auto"` detects that, so nothing needs configuring). Re-measure once it is up — `scripts/measure_latency.py --json`, per [Operating it](#operating-it) — and if the tail is worse than you want to serve, `rerank_pool_cap` in `config.yaml` is the control: it bounds how many candidates the cross-encoder scores, so halving it roughly halves the worst case.

Collect the OCIDs first. They are exported because the loop below runs as a separate process and would not otherwise see them — under `set -u` that surfaces as `C: unbound variable` on the first line, which reads like a shell problem rather than a missing export.

```bash
export C=$OCI_TENANCY        # Cloud Shell presets this to the tenancy root

# Filtering by --shape returns only images that boot on A1, which is what stops
# an x86 OCID carried over from the old instance reaching an ARM shape.
export IMAGE=$(oci compute image list --compartment-id "$C" \
  --operating-system "Canonical Ubuntu" --operating-system-version "24.04" \
  --shape VM.Standard.A1.Flex \
  --query 'data[0].id' --raw-output)

# The *public* subnet, which is not reliably the first one.
export SUBNET=$(oci network subnet list --compartment-id "$C" \
  --query 'data[?"prohibit-public-ip-on-vnic"==`false`] | [0].id' --raw-output)

# Everything on offer, and which of them that picked.
oci network subnet list --compartment-id "$C" \
  --query 'data[].{name:"display-name",id:id,private:"prohibit-public-ip-on-vnic"}' \
  --output table
echo "picked: ${SUBNET:-NONE}"
```

A VCN made by the console wizard contains both a public and a private subnet, and the private one is often listed first — so `data[0].id` selects the subnet that cannot carry `--assign-public-ip true`, and the instance either fails to launch or comes up unreachable. `prohibit-public-ip-on-vnic` is the field that decides it, and `false` means public: it is the same property the launch flag needs, rather than a guess from the name, which is only a label and can say anything.

If `picked:` prints `NONE`, this compartment has no public subnet and one has to be created before launching — the loop would otherwise fail on every attempt with an error that says nothing about subnets.

Then write the loop to a file and run it — pasting it straight into the shell would take an `exit` with it, and a file can be re-run unchanged after Cloud Shell disconnects:

```bash
cat > launch.sh <<'EOF'
#!/usr/bin/env bash
set -u
NAME=if-recs
KEY=$HOME/.ssh/if-recs.pub        # the public half of the key you will connect with

# Fetched, never uploaded — see "Verify the boot" below for why.
curl -fsSL -o cloud-init.yaml \
  https://raw.githubusercontent.com/ehenestroza/if-recommender/main/deploy/cloud-init.yaml

ADS=$(oci iam availability-domain list --compartment-id "$C" | jq -r '.data[].name')

DELAY=60           # between sweeps of every AD
MAX_DELAY=900      # ceiling, used only when OCI says we are asking too often
wait=$DELAY

while :; do
  # Re-runnable and disconnect-proof: never launch a second instance, which
  # would put the account over the free allocation and start billing.
  # `.data[]?` and the default cover the first run: with nothing matching, the
  # CLI prints nothing at all, jq then yields nothing, and an empty string is a
  # syntax error to `-gt` — failing on the one path that is entirely normal.
  live=$(oci compute instance list --compartment-id "$C" --display-name "$NAME" 2>/dev/null \
         | jq '[.data[]? | select(."lifecycle-state"
                | test("RUNNING|PROVISIONING|STARTING"))] | length' 2>/dev/null || true)
  if [ "${live:-0}" -gt 0 ]; then echo "$NAME already exists — nothing to do."; break; fi

  for AD in $ADS; do
    echo "$(date '+%H:%M:%S') trying $AD"
    if oci compute instance launch \
        --availability-domain "$AD" \
        --compartment-id "$C" \
        --display-name "$NAME" \
        --shape VM.Standard.A1.Flex \
        --shape-config '{"ocpus":2,"memoryInGBs":12}' \
        --image-id "$IMAGE" \
        --subnet-id "$SUBNET" \
        --assign-public-ip true \
        --boot-volume-size-in-gbs 100 \
        --ssh-authorized-keys-file "$KEY" \
        --user-data-file cloud-init.yaml \
        --wait-for-state RUNNING 2>launch.err
    then
      echo "LAUNCHED in $AD"; exit 0
    fi
    # The CLI reports a ServiceError as a JSON blob whose useful half — the
    # message — sits past character 200, behind the SDK version string and the
    # logging tips. Pull out the code and the message; anything that is not
    # JSON (a usage error, say) falls back to the head of the raw text.
    raw=$(tr '\n' ' ' < launch.err)
    err=$(sed 's/^[^{]*//' <<<"$raw" \
          | jq -r 'select(.code) | "\(.code): \(.message // "?")"' 2>/dev/null)
    [ -n "$err" ] || err=$(cut -c1-200 <<<"$raw")
    echo "$(date '+%H:%M:%S') $AD: $err"

    # Classified on the whole error, never on the line printed above: truncate
    # first and "Out of host capacity" is cut away before it can be matched,
    # leaving the loop to run on `InternalError` alone — which it happens to
    # survive, until a capacity error arrives carrying some other code.
    case "$raw" in
      *"Out of host capacity"*|*InternalError*|*"Internal Server Error"*)
        wait=$DELAY ;;                      # keep asking at a steady rate
      *TooManyRequests*|*"429"*)            # asking too often *is* the problem,
        wait=$(( wait * 2 ))                # so retrying at the same rate is
        [ "$wait" -gt "$MAX_DELAY" ] && wait=$MAX_DELAY
        echo "throttled — backing off to ${wait}s" ;;
      *) echo "not a capacity error, stopping:"; cat launch.err; exit 1 ;;
    esac
  done
  # Jittered, so a fleet of loops all polling on the minute do not converge on
  # the same instant — including whatever else is chasing the same capacity.
  nap=$(( wait + RANDOM % 15 ))
  echo "$(date '+%H:%M:%S') nothing free in any AD, next sweep in ${nap}s"
  sleep "$nap"
done
EOF
bash launch.sh 2>&1 | tee -a launch.log
```

**On the interval.** A sweep tries every AD, so the rate is one launch call per AD per `DELAY` — three ADs at 60s is three calls a minute, not one. 60s is a sensible floor. Capacity, when it appears, is usually taken within seconds by everyone else polling for it, so the odds are governed far more by happening to be mid-sweep than by shaving the interval, while each halving multiplies the request rate against a limit Oracle does not publish a figure for. Below about 30s there is little left to win.

The interval no longer has to be picked conservatively as insurance, because the loop distinguishes the two failures rather than lumping them together: out-of-capacity holds the steady rate, while a 429 doubles the wait up to fifteen minutes and then relaxes once capacity errors resume. That is the one case where retrying at an unchanged cadence is precisely wrong, since the request rate *is* the complaint.

Since every attempt's error is logged, the question is also answerable from evidence rather than from documentation — `grep -c TooManyRequests launch.log` says whether this tenancy minds the current cadence. If it never appears, the rate is fine.

One caveat worth checking before going faster: the CLI retries some 5xx responses internally, which would make a single loop attempt several API calls and quietly multiply the rate. `oci --help` lists the global retry options; pinning `--max-retries 0` on the launch call makes the loop's interval mean what it says.

Cloud Shell ends with the browser tab, so expect to restart the loop; the existence check at the top makes that safe.

`RUNNING` means the hypervisor started the VM, not that the app is up — cloud-init still has ten to twenty minutes of work ahead of it. Carry on with [First boot takes a while](#first-boot-takes-a-while), then the boot checks, and note that the new instance gets a new public IP: update the A record and re-issue the certificate before cutting over.

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

The app is up when the log shows Gradio listening on `127.0.0.1:7860`. That line is a `print()` rather than a log call, so it only appears because the service sets `PYTHONUNBUFFERED=1`; on an instance provisioned before that was added, expect the log to go quiet after the last INFO line and check with `curl -sI http://localhost/` instead. Silence there means running, not stuck.

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
sudo certbot --nginx -d your-domain.example
```

Cloud-init already installs certbot from snap and links it onto the path, so there is nothing to install first — and `apt install certbot` would put a second, older copy alongside it.

Add port 443 to the **security list** first — that is the VCN setting, same as port 80 above. The instance-side iptables rule for 443 is already opened by cloud-init, so there is nothing to run for it.

Certbot rewrites the nginx site in place and sets up renewal. Check both, rather than taking the second on trust — a certificate that silently stops renewing fails in ninety days, long after anyone is still watching this deploy:

```bash
curl -sI https://your-domain.example/          # 200, and the cert validates
sudo certbot renew --dry-run                   # a real renewal, against staging
systemctl list-timers | grep -i certbot        # and something scheduled to do it
```

**Every certbot command needs root**, including the read-only-looking ones: it takes a lock in `/var/log/letsencrypt` before doing anything, so a bare `certbot renew --dry-run` stops at `[Errno 13] Permission denied` on the lock file rather than telling you to use sudo.

The snap registers an ordinary systemd timer, so the third check reports a next-run time:

```
snap.certbot.renew.timer   snap.certbot.renew.service
```

`snap services certbot` says the same thing from the other side — `enabled`, `timer-activated`, and `inactive` between runs, which is what a timer-driven unit looks like when it is working rather than a sign that anything is wrong.

**Point DNS at the instance before running certbot.** The HTTP-01 challenge resolves the name and must reach *this* box; running it early fails, and running it against a stale record issues for the wrong host.

**Change the hostname only by reprovisioning, not in place.** Certbot owns the nginx site after it runs, so editing `server_name` afterwards leaves the `location /` proxy in whichever block you did not edit — nginx then answers the new name from a block with no proxy and returns a bare 404 while the app is running perfectly. Set `DOMAIN` in cloud-init and boot a fresh instance instead.

## Notes on the design

**The app binds `127.0.0.1`, not `0.0.0.0`.** Gradio reads `GRADIO_SERVER_NAME`/`GRADIO_SERVER_PORT` from the environment, so the service sets them and `app.py` needs no deployment-specific code. nginx is the only thing that reaches the app, which keeps it off the public interface and gives TLS somewhere to live.

**`uv sync --frozen`, not `uv sync`.** `--frozen` installs exactly what `uv.lock` pins and refuses to re-resolve, so the instance runs the versions that were tested. Without it a deploy months from now would silently pick up newer torch and gradio.

**`MALLOC_ARENA_MAX=2`** is set in the service environment, and **`OMP_NUM_THREADS`/`MKL_NUM_THREADS`** are appended from `nproc` at first boot. The first bounds glibc's per-thread heap arenas, which is where the app's growth under load actually goes; the second gives torch one thread per vCPU, which is where the cores actually get used since inference is serialised behind Gradio's queue.

**The clone is shallow.** History carries every superseded copy of the models, FAISS index and ranking tables — none of which delta-compress — so `.git` is several times the size of the checkout. `--depth 1` skips all of it.
