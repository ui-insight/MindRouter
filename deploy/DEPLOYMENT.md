# MindRouter Deployment Guide - Rocky Linux 8

## Prerequisites

- Rocky Linux 8 VM with root/sudo access
- Docker and Docker Compose installed
- Git installed
- SSL certificate (self-signed for testing, or real cert for production)

> **Reverse proxy:** The production Docker Compose stack includes an nginx container that handles TLS termination and reverse proxying on ports 80/443. You do **not** need to install a separate web server (Apache, nginx, etc.) on the host. If you prefer to use an external reverse proxy (e.g., Apache httpd), see [Alternative: External Apache Reverse Proxy](#alternative-external-apache-reverse-proxy) below and remove the `nginx` service from `docker-compose.prod.yml`.

> **Request body size limits come from the proxies, not from the app.** The application enforces no body-size ceiling of its own: `MAX_REQUEST_SIZE` exists in `settings.py` and `.env.example`, but nothing in `backend/` reads it, so changing it has no effect. The effective limiter is the gateway proxy — `client_max_body_size 50m` in this repo's `nginx/nginx.conf` (or `LimitRequestBody 52428800` in `deploy/apache-mindrouter.conf` if you run the external-Apache variant) — and an oversized request gets that proxy's opaque `413 Request Entity Too Large` HTML, not a JSON error. Easy to forget: the **nginx TLS proxies on each GPU inference node** count too, and get a raw nginx default of 1 MB unless configured. Full-transcript agent clients (Codex/Responses API) exceed 1 MB routinely and will see a 413 from the node proxy. Each inference node carries `/etc/nginx/conf.d/00-body-size.conf` with `client_max_body_size 64m;` (http-level, covers all server blocks). Raise the gateway and the node configs together — every hop in the chain must allow the largest legitimate request.

> **Which Compose file this guide uses.** Every command below drives **`docker-compose.prod.yml`** — the hardened stack (nginx TLS terminator, no host-exposed database or Redis, configured through `env_file: .env.prod`). The flag is not optional; each `docker compose` invocation must name the file:
>
> ```bash
> docker compose -f docker-compose.prod.yml <command>
> ```
>
> The repository ships a **second, different stack** in `docker-compose.yml`, and that is what a bare `docker compose up -d` starts — host networking, an `environment:` passthrough block instead of `env_file:`, and its own volume layout. On a host deployed with this guide, a bare `docker compose` command does **not** restart your stack; it brings up the other one alongside it. If you want the bare commands to resolve correctly anyway, export `COMPOSE_FILE=docker-compose.prod.yml` in the deployment shell.

## Step 1: Install Dependencies (if needed)

```bash
# Install Docker
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER

# Install Git
sudo dnf install -y git
```

## Step 2: Clone Repository

```bash
# Create deployment directory and clone
sudo mkdir -p /opt/mindrouter
sudo chown $USER:$USER /opt/mindrouter
cd /opt/mindrouter

# Clone from GitHub
git clone https://github.com/ui-insight/MindRouter.git .
```

## Step 3: Configure Environment

```bash
# Copy and edit production environment file
cp .env.prod.example .env.prod

# Generate secure passwords and secret key
python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_hex(32))"
python3 -c "import secrets; print('DB_PASSWORD=' + secrets.token_urlsafe(24))"
python3 -c "import secrets; print('REDIS_PASSWORD=' + secrets.token_urlsafe(24))"

# Edit .env.prod with your values
nano .env.prod
```

**Important: Update these values in .env.prod:**
- `SECRET_KEY` - Generated secret
- `MYSQL_ROOT_PASSWORD` - Secure root password
- `MYSQL_PASSWORD` and in `DATABASE_URL` - Same secure password
- `REDIS_PASSWORD` and in `REDIS_URL` - Same secure password
- `CORS_ORIGINS` - Your actual domain
- `APP_BASE_URL` - Your own public HTTPS origin, e.g. `https://mindrouter.example.com` (scheme + host, no trailing slash or path)
- `ARCHIVE_DB_HOST_PATH`, `ARTIFACT_HOST_PATH`, `VIDEO_HOST_PATH` - Host directories for persistent data (see below)

> **`APP_BASE_URL` must be set to *your* hostname before enabling SSO.** Every OIDC and Google redirect URI, and the SAML `Destination`/`Recipient` check, is derived from this value rather than from request headers. Nothing validates it at startup, so a wrong value never fails loudly — you get a working-looking deployment whose only symptom is a `redirect_uri_mismatch` at the identity provider, which reads like an IdP misconfiguration. Leaving it blank is equally wrong: the code then falls back to the request's scheme and `Host` header. Set it in Step 3, before you configure any provider.

**Host data paths.** Several data sets live in directories on the host rather than in named volumes. Point each variable at an absolute path that exists on *this* machine:

| Variable | Holds |
|----------|-------|
| `ARCHIVE_DB_HOST_PATH` | Archive MariaDB data directory (the `mariadb-archive` service) |
| `ARTIFACT_HOST_PATH` | Generated image artifacts served back to users |
| `VIDEO_HOST_PATH` | Generated video files |

Create them, and hand the app's directories to uid 1000, **before** the first start:

```bash
sudo mkdir -p /srv/mindrouter/{archivedb,artifacts,video}
sudo chown -R 1000:1000 /srv/mindrouter/artifacts /srv/mindrouter/video
```

The `chown` is the part that bites. Docker silently creates a missing bind-mount source as a **root-owned** directory, while the application container runs as **uid 1000**. A root-owned artifact directory does not stop anything: the container starts, passes its health check, serves traffic, and then every artifact and upload write fails at runtime, long after the deploy looked successful. Leave the archive database directory alone — the MariaDB image initializes and takes ownership of its own data directory on first start.

## Step 4: Configure SSL Certificate

The Docker nginx container expects SSL certificates at `./nginx/ssl/`.

```bash
cd /opt/mindrouter
mkdir -p nginx/ssl
```

### Option A: Self-signed certificate (testing only)
```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/server.key \
  -out nginx/ssl/fullchain.crt \
  -subj "/CN=mindrouter.example.com"
```

### Option B: Let's Encrypt (production)
```bash
sudo dnf install -y certbot
sudo certbot certonly --standalone -d mindrouter.example.com

# Copy certs to nginx/ssl (or symlink)
sudo cp /etc/letsencrypt/live/mindrouter.example.com/fullchain.pem nginx/ssl/fullchain.crt
sudo cp /etc/letsencrypt/live/mindrouter.example.com/privkey.pem nginx/ssl/server.key
```

## Step 5: Configure Nginx

```bash
# Edit nginx config to match your domain
nano nginx/nginx.conf
# Update server_name to your actual domain
# Verify ssl_certificate and ssl_certificate_key paths match nginx/ssl/ filenames
```

## Step 6: Configure Firewall

```bash
# Open HTTP and HTTPS ports
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

## Step 7: Configure SELinux (if enabled)

```bash
# If you have issues with Docker networking, you may need:
sudo setsebool -P container_manage_cgroup 1
```

## Step 8: Start the Application

```bash
cd /opt/mindrouter

# Build and start containers
docker compose -f docker-compose.prod.yml up -d --build

# Check status
docker compose -f docker-compose.prod.yml ps

# View logs
docker compose -f docker-compose.prod.yml logs -f app
```

## Step 9: Run Database Migrations

```bash
# Run Alembic migrations
docker compose -f docker-compose.prod.yml exec app alembic upgrade head
```

## Step 9b: Bootstrap the First Admin Account

A fresh database has **no users**. Until an admin exists you cannot reach the
admin dashboard at all — including **Admin → Branding**, which supplies the
institution name used as the SSO button label, and **Admin → Groups**, which
must contain any group you name in an SSO `*_DEFAULT_GROUP` variable. Treat this
step as required, not optional.

The bootstrap tool is `scripts/seed_dev_data.py`. Its banner says
"MindRouter Development Data Seeder", but it is the only first-admin path
shipped, and it is safe to run in production **provided you set a password**:

```bash
docker compose -f docker-compose.prod.yml exec \
  -e ADMIN_PASSWORD='<strong-unique-password>' \
  app python scripts/seed_dev_data.py
```

What it creates:

- The seven default groups (`students`, `staff`, `faculty`, `researchers`, `admin`, `nerds`, `other`) if the migrations have not already created them.
- Exactly **one** user: `admin` / `admin@mindrouter.local`, role `ADMIN`, in the `admin` group, with a quota row. It creates no other accounts.
- One API key for that user, printed once as a single parseable line, `ADMIN_API_KEY=mr2_...`. Capture it now — it cannot be redisplayed.

Environment overrides it honors:

| Variable | Effect |
|----------|--------|
| `ADMIN_PASSWORD` | Password for the admin account. **Defaults to `admin123` if unset** — never accept that default on an internet-reachable host. |
| `ADMIN_API_KEY` | Use this exact key instead of minting a random one. **Must start with `mr2_`** — authentication rejects any other prefix before it even looks the key up, so a key without it would silently 401 forever. Since 2.9.8 the script refuses to run rather than store an unusable key. |
| `MINT_ADMIN_KEY=1` | Issue an additional API key for an admin that already exists. |

Two behaviors worth knowing before you run it:

- **It is idempotent, and that cuts both ways.** If `admin` already exists the script leaves it untouched and exits without minting a key (unless `ADMIN_API_KEY` or `MINT_ADMIN_KEY` is set). `ADMIN_PASSWORD` therefore applies **only at creation** — it will not reset a forgotten password.
- **The closing banner always prints `admin / admin123` verbatim**, even when you supplied `ADMIN_PASSWORD`. It is a hardcoded string, not a readback of what was set. Ignore it and use the password you passed.

Then, immediately:

1. Log in at `https://your-domain/login` as `admin`.
2. Change the password from the account page (do this even if you set a strong `ADMIN_PASSWORD`, since it was visible to your shell history and to anyone reading the deploy transcript).
3. Set **Admin → Branding → Institution / organization name** if you plan to enable SSO — it is what the primary SSO button is labeled with.

### Ordering on a truly fresh database

The app reads the `backends` table during startup, so on an unmigrated
database it exits before it can serve — and the `docker compose exec` you need
in order to migrate then races the restart loop. Bring the schema up **before**
the app serves traffic, either way round:

```bash
# Option A — migrate out of band (recommended, and required for multi-worker)
docker compose -f docker-compose.prod.yml up -d mariadb
docker compose -f docker-compose.prod.yml run --rm app alembic upgrade head
docker compose -f docker-compose.prod.yml up -d

# Option B — let the app migrate itself on first boot
RUN_MIGRATIONS=1 docker compose -f docker-compose.prod.yml up -d
```

`RUN_MIGRATIONS=1` runs `alembic upgrade head` inside the app before anything
reads the schema. Concurrent workers are serialized on a database advisory lock,
so one applies the DDL and the others wait — but Option A remains the
recommendation, because a migration failure there is a plain command failure
instead of a container that won't start. Unset the flag once the schema is at
head.

Two upgrade notes: this flag was inert-and-then-fatal before 2.9.8, so don't
carry a `RUN_MIGRATIONS=1` value forward from an old `.env` without re-reading
this. And when you finish with it, **delete the line rather than blanking it** —
`env_file` passes `RUN_MIGRATIONS=` through as an empty string, which fails
boolean validation and stops the app from starting.

### Making an SSO identity the admin

SSO **cannot** produce the first admin on its own: newly provisioned SSO users
land in a **non-admin** group chosen at provision time — the provider's
`*_DEFAULT_GROUP` (default `other`), or for Azure the `jobTitle` mapping
(`students`/`faculty`/`staff`, falling back to `AZURE_AD_DEFAULT_GROUP`) —
nothing promotes a first user, and creating an API key to drive the admin API
requires a principal that does not exist yet. Bootstrap the local `admin` above
first, then use **one** of these:

1. **Email pre-linking (cleanest — do it before your first SSO login).** As the
   local `admin`, create an account (Admin → Users → Create Local User) whose
   email exactly matches your institutional SSO email, and put it in the `admin`
   group. Your first SSO login then *links to that existing account* and keeps
   its group. Order matters: if you log in via SSO first, you get a separate
   non-admin account instead, and you must use option 2.
2. **Promote after the fact.** Log in via SSO once so the account is
   provisioned, then sign in as the local `admin`, open Admin → Users → *your
   SSO user* → Edit, and set **Group** to `admin`. It takes effect on your next
   page load.

Do **not** set a provider's `*_DEFAULT_GROUP` to `admin` as a shortcut: it makes
**every** user from that provider an admin for as long as it is set, and the
group is fixed at provision time, so accounts created meanwhile keep admin after
you change it back. (For Azure it is not even reliable — the `jobTitle` mapping
is consulted first and silently overrides `AZURE_AD_DEFAULT_GROUP`.)

Keep the local `admin` account. It is your way back in if SSO breaks, and an
SSO-provisioned account cannot be given a local password afterwards.

## Single sign-on (optional)

SSO is **entirely environment-variable driven** — there is no admin-UI toggle. A
provider is on when its required variables are present in `.env.prod` and off
when they are unset.

- **Azure AD / Entra ID, Google, generic OIDC, and SAML 2.0 are independent**, and any subset can run at once. The login page renders one button per enabled provider.
- **Local username/password login never goes away.** SSO buttons appear alongside the local form, so the admin account from Step 9b remains usable if an IdP breaks.
- **`APP_BASE_URL` must already be correct** (Step 3). Redirect URIs and the SAML `Destination` check are built from it.
- **SAML requires HTTPS.** Its SP-initiated handshake relies on a `SameSite=None; Secure` cookie, which browsers drop over plain HTTP.
- **Button labels come from Admin → Branding**, so complete Step 9b first.
- **Restart after changing any SSO variable** — settings are cached per worker process at startup:
  ```bash
  docker compose -f docker-compose.prod.yml up -d
  ```

Per-provider walkthroughs — app registration, claim/attribute mapping, IdP
metadata, JIT provisioning, account-linking rules, and the full variable
reference — are in **[../docs/sso-configuration.md](../docs/sso-configuration.md)**.
Follow that guide for the provider details; this section only covers how SSO
fits into the deployment.

## Step 10: Deploy GPU Sidecar Agents

Each GPU inference node needs a sidecar agent running to report GPU metrics back to MindRouter.

### 10a. Install NVIDIA Container Toolkit

The sidecar container requires GPU access via `--gpus all`. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on each GPU node:

```bash
# RHEL/Rocky Linux (aspen nodes)
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
sudo dnf install -y nvidia-container-toolkit

# Debian/Ubuntu
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

# Configure and restart Docker
sudo nvidia-ctk runtime configure --driver=docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### 10b. Configure Docker daemon network on each GPU node

Docker's default bridge network (`172.17.0.0/16`) can collide with campus or institutional routing. Configure each GPU node to use `10.x.x.x` address space:

```bash
# Create /etc/docker/daemon.json on each GPU node
sudo tee /etc/docker/daemon.json <<'EOF'
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "bip": "10.77.0.1/16",
    "default-address-pools": [
        { "base": "10.78.0.0/16", "size": 24 }
    ]
}
EOF

sudo systemctl restart docker
```

### 10c. Deploy the sidecar container

The sidecar requires a `SIDECAR_SECRET_KEY` for authentication. Generate one per node:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Build and deploy the sidecar directly from GitHub on each GPU server. In production, bind the container to localhost only and use nginx as a reverse proxy.

Create the sidecar configuration directory and env file (once per node):

```bash
ssh user@gpu-server

sudo mkdir -p /etc/mindrouter
python3 -c "import secrets; print('SIDECAR_SECRET_KEY=' + secrets.token_hex(32))" | sudo tee /etc/mindrouter/sidecar.env
sudo chmod 600 /etc/mindrouter/sidecar.env
```

Build and run:

```bash
# Build a specific release tag directly from GitHub (no clone needed)
docker build -t mindrouter-sidecar:v2.0.0 \
  -f Dockerfile.sidecar \
  https://github.com/ui-insight/MindRouter.git#v2.0.0:sidecar

# Or build latest from master
docker build -t mindrouter-sidecar:latest \
  -f Dockerfile.sidecar \
  https://github.com/ui-insight/MindRouter.git:sidecar

# Run bound to localhost only (nginx will proxy external traffic)
docker run -d --name gpu-sidecar \
  --gpus all \
  -p 127.0.0.1:18007:8007 \
  --env-file /etc/mindrouter/sidecar.env \
  --restart unless-stopped \
  mindrouter-sidecar:v2.0.0
```

To upgrade an existing sidecar to a new version:

```bash
docker build -t mindrouter-sidecar:v2.0.0 \
  -f Dockerfile.sidecar \
  https://github.com/ui-insight/MindRouter.git#v2.0.0:sidecar
docker stop gpu-sidecar && docker rm gpu-sidecar
docker run -d --name gpu-sidecar \
  --gpus all \
  -p 127.0.0.1:18007:8007 \
  --env-file /etc/mindrouter/sidecar.env \
  --restart unless-stopped \
  mindrouter-sidecar:v2.0.0
```

### 10d. Configure nginx reverse proxy

Install nginx and create a proxy config so MindRouter can reach the sidecar on port 8007:

```bash
# Install nginx (Rocky Linux / RHEL)
sudo dnf install -y nginx
sudo systemctl enable --now nginx

# Create sidecar proxy config
sudo tee /etc/nginx/conf.d/sidecar-proxy.conf <<'EOF'
server {
    listen 8007;
    listen [::]:8007;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:18007;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Sidecar-Key $http_x_sidecar_key;
        proxy_connect_timeout 5s;
        proxy_read_timeout 10s;
    }
}
EOF

sudo systemctl reload nginx
```

Verify the sidecar is reachable:

```bash
# Locally
curl -H "X-Sidecar-Key: your-generated-key" http://localhost:8007/health

# From MindRouter server
curl -H "X-Sidecar-Key: your-generated-key" http://gpu-server.example.com:8007/health
```

### 10e. Register the node in MindRouter

Include the same key that was set as `SIDECAR_SECRET_KEY` on the sidecar:

```bash
curl -X POST https://mindrouter.example.com/api/admin/nodes/register \
  -H "Authorization: Bearer admin-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpu-server-1",
    "hostname": "gpu1.example.com",
    "sidecar_url": "http://gpu1.example.com:8007",
    "sidecar_key": "your-generated-key"
  }'
```

Or use the admin dashboard at `/admin/nodes`.

### Register backends on the node:

```bash
# Backend using all GPUs on the node
curl -X POST https://mindrouter.example.com/api/admin/backends/register \
  -H "Authorization: Bearer admin-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ollama-gpu1",
    "url": "http://gpu1.example.com:11434",
    "engine": "ollama",
    "max_concurrent": 4,
    "node_id": 1
  }'

# Backend using specific GPUs (for multi-backend nodes)
curl -X POST https://mindrouter.example.com/api/admin/backends/register \
  -H "Authorization: Bearer admin-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "vllm-gpu1-01",
    "url": "http://gpu1.example.com:8000",
    "engine": "vllm",
    "max_concurrent": 16,
    "node_id": 1,
    "gpu_indices": [0, 1]
  }'
```

**Note:** `gpu_indices` is optional. Omit it to assign all GPUs on the node to the backend. Use it when multiple backends share the same physical server and you want each to report telemetry only for its assigned GPUs.

## Step 11: Verify Deployment

```bash
# Test health endpoint directly
curl http://127.0.0.1:8000/healthz

# Test through Apache (replace with your domain)
curl -k https://mindrouter.example.com/healthz

# Check all services are healthy
docker compose -f docker-compose.prod.yml ps

# Test OpenAI-compatible endpoint
curl -X POST https://mindrouter.example.com/v1/chat/completions \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "max_tokens": 16, "messages": [{"role": "user", "content": "Say ok."}]}'

# Test Anthropic-compatible endpoint
curl -X POST https://mindrouter.example.com/anthropic/v1/messages \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "max_tokens": 16, "messages": [{"role": "user", "content": "Say ok."}]}'

# Verify sidecar connectivity (from MindRouter server)
curl http://gpu1.example.com:8007/health

# Verify node appears in telemetry
curl -H "Authorization: Bearer admin-api-key" \
  https://mindrouter.example.com/api/admin/telemetry/overview
```

**Firewall note:** The MindRouter server needs network access to each GPU node's sidecar port (default 8007). Ensure firewall rules allow this traffic between the gateway and GPU nodes.

## Uvicorn Worker Count

The Dockerfile runs `uvicorn` with `--workers 8` by default. This should be adjusted based on available CPU cores in production. A common starting point is 2-4 workers per CPU core. Too many workers on a small VM will cause memory pressure; too few will underutilize the CPU.

## MariaDB Tuning

The file `mariadb/custom.cnf` contains production tuning parameters that are mounted into the MariaDB container:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `innodb_buffer_pool_size` | `2G` | Adjust based on available RAM (typically 50-70% of total) |
| `innodb_log_file_size` | `512M` | Larger log files improve write performance |
| `innodb_io_capacity` | `2000` | Tune for SSD-backed storage |
| `max_connections` | `512` | Each uvicorn worker has its own connection pool, so keep `UVICORN_WORKERS × (DATABASE_POOL_SIZE + DATABASE_MAX_OVERFLOW)`, plus the `mcp` container's pool (default 30 + 20) and admin clients, well under this. Production: 8 × (20 + 25) + 50 ≈ 410. Exceeding it turns one overloaded worker into "Too many connections" (error 1040) for everyone — see `docs/ocr-load-guard-deploy-notes.md` |
| `slow_query_log` | `1` | Logs queries taking longer than 1 second |

Operators should adjust `innodb_buffer_pool_size` based on available RAM on the database host. On a dedicated MariaDB server, 50-70% of total RAM is a reasonable target.

## Nginx Configuration

Key nginx parameters in the production config that operators may need to adjust:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `client_max_body_size` | `50m` | Maximum upload/request body size. This is the deployment's effective limit — the app enforces none of its own |
| `proxy_read_timeout` | `720s` | Must exceed the app's worst-case request lifetime for long-running LLM requests |
| `proxy_send_timeout` | `720s` | Timeout for sending data to the proxied server |
| `proxy_connect_timeout` | `60s` | Timeout for establishing a connection to the backend |

The shipped `720s` read/send timeouts are sized to cover the app's worst case: routing wait plus `BACKEND_RETRY_MAX_ATTEMPTS` (default 3) attempts of `BACKEND_REQUEST_TIMEOUT_PER_ATTEMPT` (default 180s) each. If you raise those settings, or `BACKEND_REQUEST_TIMEOUT` (default 300s), raise `proxy_read_timeout` to stay ahead of them — otherwise nginx cuts the connection before the backend answers.

## Docker Network Subnet

The production `docker-compose.prod.yml` defines a custom bridge network with subnet `10.101.0.0/16`. This is used to avoid collisions with institutional/campus network routing that may overlap with Docker's default `172.17.0.0/16` range. Operators should adjust this subnet if `10.101.0.0/16` conflicts with their network.

## Docker Compose Profiles

The Docker Compose configuration supports profiles for optional services:

- **Default (no profile)**: Runs the core services — `app`, `mariadb`, `redis`, `nginx`
- **`--profile dev`**: Adds development tools
- **`--profile gpu`**: Adds the `gpu-agent` service for GPU telemetry collection on sidecar nodes

```bash
# Start core services only
docker compose -f docker-compose.prod.yml up -d

# Start with GPU sidecar agent
docker compose -f docker-compose.prod.yml --profile gpu up -d

# Start with development tools
docker compose -f docker-compose.prod.yml --profile dev up -d
```

## Ongoing Operations

### View Logs
```bash
# Application logs
docker compose -f docker-compose.prod.yml logs -f app

# Apache logs
sudo tail -f /var/log/httpd/mindrouter_error.log
sudo tail -f /var/log/httpd/mindrouter_access.log
```

### Restart Services
```bash
# Restart app only
docker compose -f docker-compose.prod.yml restart app

# Restart everything
docker compose -f docker-compose.prod.yml restart
```

### Update Application
```bash
cd /opt/mindrouter
git pull origin master

# Rebuild and restart
docker compose -f docker-compose.prod.yml up -d --build

# Run any new migrations
docker compose -f docker-compose.prod.yml exec app alembic upgrade head
```

### Backup Database
```bash
# Source the env file to get the root password
source .env.prod

# Backup
docker compose -f docker-compose.prod.yml exec mariadb \
  mysqldump -u root -p"${MYSQL_ROOT_PASSWORD}" mindrouter > backup_$(date +%Y%m%d).sql

# Restore
docker compose -f docker-compose.prod.yml exec -T mariadb \
  mysql -u root -p"${MYSQL_ROOT_PASSWORD}" mindrouter < backup.sql
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker compose -f docker-compose.prod.yml logs app

# Check if ports are in use
sudo ss -tlnp | grep 8000
```

### Nginx 502 Bad Gateway
```bash
# Check if app container is running
docker compose -f docker-compose.prod.yml ps app

# Check app health from inside the Docker network
docker compose -f docker-compose.prod.yml exec nginx curl http://app:8000/healthz
```

### Database connection issues
```bash
# Check MariaDB is healthy
docker compose -f docker-compose.prod.yml ps mariadb

# Check connection from app container
docker compose -f docker-compose.prod.yml exec app \
  python -c "from backend.app.db.session import engine; print('OK')"
```

## Security Checklist

- [ ] Changed all default passwords in .env.prod
- [ ] Generated unique SECRET_KEY
- [ ] First admin seeded with a strong `ADMIN_PASSWORD` — **not** the built-in `admin123` default — and its password changed after first login
- [ ] Admin API key printed by the seeder stored in a secret manager, not left in shell history or deploy logs
- [ ] `APP_BASE_URL` points at this deployment's own hostname (wrong values fail silently and break SSO)
- [ ] SSL certificate installed and working
- [ ] Firewall configured (only 80/443 open)
- [ ] SELinux properly configured
- [ ] Database not exposed externally
- [ ] Redis not exposed externally
- [ ] CORS_ORIGINS set to actual domain
- [ ] Disabled DEBUG mode
- [ ] GPU sidecar containers bound to localhost only (127.0.0.1:18007)
- [ ] Nginx reverse proxy configured on each GPU node (port 8007 → 127.0.0.1:18007)
- [ ] Sidecar agents running on all GPU nodes with unique `SIDECAR_SECRET_KEY`
- [ ] Docker daemon configured with 10.x.x.x address space on all GPU nodes

## Production Security Hardening

### CORS

By default, MindRouter's CORS middleware is configured with `allow_methods=["*"]` and `allow_headers=["*"]`. For tighter security in production, customize `CORS_ORIGINS` in `.env.prod` to restrict allowed origins to your actual domain(s) only.

### Security Headers

MindRouter does not set HSTS, X-Frame-Options, or CSP headers at the application level. These should be added at the reverse proxy layer (nginx or Apache). For example, in your nginx config:

```
add_header Strict-Transport-Security "max-age=63072000; includeSubDomains" always;
add_header X-Frame-Options "DENY" always;
add_header Content-Security-Policy "default-src 'self'" always;
```

### Session Cookies

Set `SESSION_COOKIE_SECURE=True` in `.env.prod` for HTTPS deployments to ensure session cookies are only transmitted over TLS. This prevents cookies from being sent over unencrypted HTTP connections.

---

## Alternative: External Apache Reverse Proxy

If you prefer to use Apache httpd instead of the Docker nginx container:

1. **Remove the nginx service** from `docker-compose.prod.yml` and expose the app port directly:
   ```yaml
   app:
     ports:
       - "127.0.0.1:8000:8000"
   ```

2. **Install and configure Apache:**
   ```bash
   sudo dnf install -y httpd mod_ssl mod_proxy_html
   sudo cp deploy/apache-mindrouter.conf /etc/httpd/conf.d/mindrouter.conf
   # Edit ServerName, SSL cert paths, and ProxyPass target
   sudo nano /etc/httpd/conf.d/mindrouter.conf
   sudo setsebool -P httpd_can_network_connect 1
   sudo systemctl enable --now httpd
   ```

3. Place SSL certificates in the paths specified in the Apache config (typically `/etc/pki/tls/`).

> **Do not run both** the Docker nginx service and a host Apache on ports 80/443 — they will conflict.
