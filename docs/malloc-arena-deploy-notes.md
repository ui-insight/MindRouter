# 2.9.101 (MALLOC_ARENA_MAX) Deploy Notes

No `.env` change and no migration. The image sets `MALLOC_ARENA_MAX=2`, and it
takes effect when the app is rebuilt and recreated (a normal deploy). The `mcp`
container is built from the same Dockerfile, so it gets the setting once it is
rebuilt and recreated too (remember that `mcp` needs its explicit
`docker compose restart mcp` on code deploys); only the app's value can be
changed from `.env`.

## Why

On 2026-11-06 the app container used 33 GB of the host's 62 GB, leaving 12 GB:

| Workers | Age | Resident memory each |
|---|---|---|
| 5 | 23 days (since the 2.9.94 deploy) | 3.9–4.7 GB |
| 3 (restarted during the 11-05 OCR outage) | ~12 hours | 2.2–5.9 GB |

glibc gives allocating threads their own malloc arenas (up to 8 per CPU), and
memory freed inside an arena is seldom returned to the OS. OCR page conversion
runs in threads, and a 200-dpi page is about 11 MB as an RGB image. Once such a
buffer is freed, glibc raises its "allocate large blocks directly from the OS"
threshold (up to 32 MiB), so later page buffers are carved from an arena and
stay there. With the default, each of dozens of pool threads can keep its own
copy of a busy moment; with 2 arenas, at most two. This is a cap, not a
guarantee: one large conversion can still pin its peak inside an arena.

## Before deploying: confirm the diagnosis

On the host, for one or two large workers (host PIDs from `docker top`):

```bash
docker top mindrouter-app-1 -eo pid,rss,etime,args | sort -k2 -rn | head -10
sudo pmap -x <pid> | sort -k3 -n | tail -40
```

Many anonymous regions of about 64 MiB (in `pmap -x` a writable `rw---` part
followed by an unreadable `-----` reserve) are malloc heaps belonging to
thread arenas; one arena can chain several of them. Lots of them is the
pattern this change addresses. A few very large regions, or steady growth of
the main heap (the region right after the program's data), point to live
objects or a leak, which it will not fix. Record the table above (or a fresh
one) as the baseline.

## After deploying

Check that the setting reached the workers. This lists processes in the
container that do NOT have it, so it should print nothing:

```bash
docker exec mindrouter-app-1 sh -c 'grep -L MALLOC_ARENA_MAX=2 /proc/[0-9]*/environ'
docker exec mindrouter-app-1 ps -eo pid,rss,args   # names the PIDs, if anything printed
```

Or for one worker from the host, using a host PID from `docker top`:

```bash
sudo tr '\0' '\n' < /proc/<pid>/environ | grep ^MALLOC_ARENA_MAX=
```

Don't judge it by counting 64 MiB regions in `pmap`: with 2 arenas every
thread shares them, and a shared arena can still chain several heaps after a
busy day. Whether it helps shows in resident memory over time (below).

A deploy restarts every worker, so memory drops whether or not this helps.
Compare like with like: record each worker's resident memory at about 1 hour,
1 day and 7 days after the deploy, and compare with workers of the same age
before it (the 12-hour-old workers above reached 2.2–5.9 GB; part of a
worker's size is legitimate, such as the TrustMark watermark model loaded on
first use). Success is a curve that flattens instead of stepping up after
every busy OCR day.

## If memory still climbs

1. `MALLOC_MMAP_THRESHOLD_=4194304` (note the trailing underscore) fixes the
   large-block threshold at 4 MiB and turns off its automatic rise, so OCR
   page-image buffers (4 MiB and up) are always mapped from the OS and
   returned to it when freed. The cost: every such buffer becomes its own
   mmap/munmap with fresh zero-filled pages (about 2,800 page faults for an
   11 MB page image), and the heap trim threshold stays at 128 KiB, so the
   heap is trimmed more often. Add it to the app's `environment:` in compose.
2. Worker recycling: `--limit-max-requests N` with
   `--limit-max-requests-jitter M` on the compose `command:` (uvicorn 0.54).
   A recycling worker finishes its requests for up to
   `--timeout-graceful-shutdown` (60 s); only longer ones, such as long
   streams, are cut off.

## Tune or roll back

Set `MALLOC_ARENA_MAX` in `/opt/mindrouter/.env` and run
`docker compose up -d app` (no rebuild). That recreates the container: all
eight workers restart, and requests in flight get up to the container's stop
grace period (75 s) to finish, so do it at a quiet time. `0` restores glibc's default (8 per
CPU). If request latency rises, which would mean allocator contention between
threads, try 4.
